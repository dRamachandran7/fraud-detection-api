"""Tests for the score consumer."""
from __future__ import annotations

import json
from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np
import pytest
from kafka.errors import CommitFailedError, KafkaTimeoutError
from kafka.structs import TopicPartition

from src import config
from src.consumer.batching import commit_batch, rewind
from src.consumer.score_consumer import _parse, _process_batch
from src.inference import predict, predict_batch
from tests.test_inference import make_artifact


def record(payload, offset: int = 0) -> SimpleNamespace:
    """A stand-in for a ConsumerRecord: the loop only reads .value/.offset."""
    raw = payload if isinstance(payload, bytes) else json.dumps(payload).encode()
    return SimpleNamespace(value=raw, offset=offset)


def txn(tid: str) -> dict:
    return {"transaction_id": tid, "features": [0.1] * 29}


def producer() -> MagicMock:
    p = MagicMock()
    p.send.return_value.get.return_value = None
    return p


def sent_to(p: MagicMock, topic: str) -> list[dict]:
    return [c.kwargs["value"] for c in p.send.call_args_list if c.args[0] == topic]


class TestConsumerScoring:
    def test_predict_returns_expected_keys(self):
        result = predict(make_artifact(0.5), [0.1] * 29)
        assert {"is_fraud", "score", "model_version"} <= result.keys()

    def test_fraud_flagged_at_or_above_threshold(self):
        assert predict(make_artifact(0.9), [0.1] * 29)["is_fraud"] is True

    def test_clean_transaction_not_flagged(self):
        assert predict(make_artifact(0.01), [0.1] * 29)["is_fraud"] is False

    def test_predict_batch_preserves_order_in_one_model_call(self):
        artifact = make_artifact(0.1)
        artifact["pipeline"].predict_proba.side_effect = lambda X: np.column_stack(
            [1 - X["V1"], X["V1"]]
        )
        rows = [[p] + [0.0] * 28 for p in (0.9, 0.1, 0.7)]

        results = predict_batch(artifact, rows)

        assert [r["score"] for r in results] == pytest.approx([0.9, 0.1, 0.7])
        assert artifact["pipeline"].predict_proba.call_count == 1


class TestParse:
    def _payload(self, **overrides) -> bytes:
        event = {"transaction_id": "txn-1", "features": [0.1] * 29}
        event.update(overrides)
        return json.dumps(event).encode("utf-8")

    def test_valid_message_parses(self):
        transaction_id, features = _parse(self._payload())
        assert transaction_id == "txn-1"
        assert len(features) == 29

    def test_malformed_json_rejected(self):
        with pytest.raises(ValueError, match="malformed JSON"):
            _parse(b"{not json")

    def test_missing_features_rejected(self):
        with pytest.raises(ValueError, match="no features"):
            _parse(self._payload(features=[]))

    def test_wrong_feature_count_rejected(self):
        with pytest.raises(ValueError, match="expected 29 features, got 3"):
            _parse(self._payload(features=[0.1, 0.2, 0.3]))

    def test_non_numeric_feature_rejected(self):
        with pytest.raises(ValueError, match="non-numeric"):
            _parse(self._payload(features=["a"] * 29))

    def test_non_object_payload_rejected(self):
        with pytest.raises(ValueError, match="not an object"):
            _parse(b"[1, 2, 3]")


class TestProcessBatch:
    def test_every_score_published_keyed_by_transaction_id(self):
        p = producer()
        _process_batch(make_artifact(0.1), p, [record(txn("a")), record(txn("b"))])

        scores = sent_to(p, config.KAFKA_SCORES_TOPIC)
        assert [s["transaction_id"] for s in scores] == ["a", "b"]
        keys = [c.kwargs["key"] for c in p.send.call_args_list]
        assert keys == ["a", "b"]

    def test_batch_is_scored_in_one_model_call(self):
        artifact = make_artifact(0.1)
        _process_batch(artifact, producer(), [record(txn(str(i))) for i in range(10)])
        assert artifact["pipeline"].predict_proba.call_count == 1

    def test_all_sends_issued_before_waiting_on_acks(self):
        """One ack round trip per batch, not one per message."""
        p = producer()
        order = []
        p.send.side_effect = lambda *a, **k: order.append("send") or SimpleNamespace(
            get=lambda timeout: order.append("ack")
        )
        _process_batch(make_artifact(0.1), p, [record(txn("a")), record(txn("b"))])
        assert order == ["send", "send", "ack", "ack"]

    def test_model_version_travels_on_the_score_message(self):
        p = producer()
        _process_batch(make_artifact(0.1), p, [record(txn("a"))])
        assert sent_to(p, config.KAFKA_SCORES_TOPIC)[0]["model_version"] == "test-version"

    def test_result_is_json_serializable(self):
        """Regression: numpy bool_ from predict() broke json.dumps."""
        p = producer()
        _process_batch(make_artifact(0.1), p, [record(txn("a"))])
        json.dumps(sent_to(p, config.KAFKA_SCORES_TOPIC)[0])

    def test_poison_message_dead_lettered_and_rest_of_batch_scored(self):
        p = producer()
        _process_batch(make_artifact(0.1), p, [record(b"{not json"), record(txn("good"))])

        assert len(sent_to(p, config.KAFKA_DLQ_TOPIC)) == 1
        assert [s["transaction_id"] for s in sent_to(p, config.KAFKA_SCORES_TOPIC)] == ["good"]

    def test_one_unscoreable_row_does_not_dead_letter_the_batch(self):
        """Batch call fails -> row-by-row fallback isolates the bad row."""
        artifact = make_artifact(0.1)

        def proba(X):
            if (X["V1"] == 999).any():
                raise ValueError("model rejected input")
            return np.tile([0.9, 0.1], (len(X), 1))

        artifact["pipeline"].predict_proba.side_effect = proba
        bad = {"transaction_id": "bad", "features": [999] + [0.1] * 28}
        p = producer()

        _process_batch(artifact, p, [record(txn("a")), record(bad), record(txn("b"))])

        assert [s["transaction_id"] for s in sent_to(p, config.KAFKA_SCORES_TOPIC)] == ["a", "b"]
        dlq = sent_to(p, config.KAFKA_DLQ_TOPIC)
        assert len(dlq) == 1 and "scoring failed" in dlq[0]["reason"]

    def test_broker_error_propagates_so_the_batch_is_rewound(self):
        p = producer()
        p.send.return_value.get.side_effect = KafkaTimeoutError("no ack")
        with pytest.raises(KafkaTimeoutError):
            _process_batch(make_artifact(0.1), p, [record(txn("a"))])


class TestOffsets:
    def test_rewind_seeks_each_partition_to_first_offset_of_batch(self):
        """Regression: skipping the commit alone let poll() move past a failed
        batch, and the next commit skipped it for good."""
        tp0, tp1 = TopicPartition("t", 0), TopicPartition("t", 1)
        consumer = MagicMock()
        consumer.assignment.return_value = {tp0, tp1}

        rewind(consumer, {tp0: [record(txn("a"), 40), record(txn("b"), 41)],
                          tp1: [record(txn("c"), 7)]})

        consumer.seek.assert_any_call(tp0, 40)
        consumer.seek.assert_any_call(tp1, 7)

    def test_rewind_skips_partitions_lost_in_a_rebalance(self):
        tp0, tp1 = TopicPartition("t", 0), TopicPartition("t", 1)
        consumer = MagicMock()
        consumer.assignment.return_value = {tp0}

        rewind(consumer, {tp0: [record(txn("a"), 1)], tp1: [record(txn("b"), 2)]})

        consumer.seek.assert_called_once_with(tp0, 1)

    def test_failed_commit_does_not_crash_the_consumer(self):
        consumer = MagicMock()
        consumer.commit.side_effect = CommitFailedError("rebalanced")
        assert commit_batch(consumer) is False
