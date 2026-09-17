"""Tests for the score consumer."""
from __future__ import annotations

import json
from unittest.mock import MagicMock

import numpy as np
import pytest

from src import config
from src.consumer.score_consumer import _handle_message, _parse
from src.inference import predict
from tests.test_inference import make_artifact


class TestConsumerScoring:
    def test_predict_returns_expected_keys(self):
        result = predict(make_artifact(0.5), [0.1] * 29)
        assert "is_fraud" in result
        assert "score" in result
        assert "model_version" in result

    def test_fraud_flagged_below_threshold(self):
        result = predict(make_artifact(-0.5), [0.1] * 29)
        assert result["is_fraud"] is True

    def test_clean_transaction_not_flagged(self):
        result = predict(make_artifact(0.3), [0.1] * 29)
        assert result["is_fraud"] is False


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


class TestHandleMessage:
    def _model(self):
        return make_artifact(0.5)

    def _producer(self):
        producer = MagicMock()
        producer.send.return_value.get.return_value = None
        return producer

    def test_score_published_with_transaction_id_key(self):
        producer = self._producer()
        raw = json.dumps({"transaction_id": "txn-1", "features": [0.1] * 29}).encode()

        _handle_message(self._model(), producer, raw)

        producer.send.assert_called_once()
        args, kwargs = producer.send.call_args
        assert args[0] == config.KAFKA_SCORES_TOPIC
        assert kwargs["key"] == "txn-1"
        assert kwargs["value"]["transaction_id"] == "txn-1"

    def test_model_version_travels_on_the_score_message(self):
        producer = self._producer()
        raw = json.dumps({"transaction_id": "txn-1", "features": [0.1] * 29}).encode()

        _handle_message(self._model(), producer, raw)

        assert producer.send.call_args.kwargs["value"]["model_version"] == "test-version"

    def test_result_is_json_serializable(self):
        """Regression: numpy bool_ from predict() broke json.dumps."""
        producer = self._producer()
        raw = json.dumps({"transaction_id": "txn-1", "features": [0.1] * 29}).encode()

        _handle_message(self._model(), producer, raw)

        json.dumps(producer.send.call_args.kwargs["value"])

    def test_poison_message_is_dead_lettered_not_raised(self):
        producer = self._producer()

        _handle_message(self._model(), producer, b"{not json")

        assert producer.send.call_args.args[0] == config.KAFKA_DLQ_TOPIC

    def test_scoring_failure_is_dead_lettered(self):
        producer = self._producer()
        model = self._model()
        model["pipeline"].decision_function.side_effect = ValueError("feature mismatch")
        raw = json.dumps({"transaction_id": "txn-1", "features": [0.1] * 29}).encode()

        _handle_message(model, producer, raw)

        assert producer.send.call_args.args[0] == config.KAFKA_DLQ_TOPIC
