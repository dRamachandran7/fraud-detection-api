"""Tests for the results consumer."""
from __future__ import annotations

import json
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from src.consumer.results_consumer import _parse, _process_batch


def _payload(**overrides) -> bytes:
    event = {
        "transaction_id": "txn-1",
        "is_fraud": True,
        "score": 0.9,
        "model_version": "v1",
    }
    event.update(overrides)
    return json.dumps(event).encode("utf-8")


def record(raw: bytes) -> SimpleNamespace:
    return SimpleNamespace(value=raw, offset=0)


class TestParse:
    def test_valid_score_parses(self):
        assert _parse(_payload())["transaction_id"] == "txn-1"

    def test_malformed_json_rejected(self):
        with pytest.raises(ValueError, match="malformed JSON"):
            _parse(b"{nope")

    def test_missing_model_version_rejected(self):
        with pytest.raises(ValueError, match="model_version"):
            _parse(_payload(model_version=None))

    def test_is_fraud_false_is_not_treated_as_missing(self):
        """Regression guard: a falsy-but-present verdict must be kept."""
        assert _parse(_payload(is_fraud=False))["is_fraud"] is False


class TestProcessBatch:
    @patch("src.consumer.results_consumer.store")
    def test_batch_persisted_in_one_write(self, store):
        _process_batch([record(_payload(transaction_id="a")), record(_payload(transaction_id="b"))])

        store.save_scores.assert_called_once()
        rows = store.save_scores.call_args.args[0]
        assert [r["transaction_id"] for r in rows] == ["a", "b"]

    @patch("src.consumer.results_consumer.store")
    def test_malformed_message_skipped_rest_persisted(self, store):
        _process_batch([record(b"{nope"), record(_payload())])

        rows = store.save_scores.call_args.args[0]
        assert [r["transaction_id"] for r in rows] == ["txn-1"]

    @patch("src.consumer.results_consumer.store")
    def test_all_malformed_writes_nothing(self, store):
        _process_batch([record(b"{nope")])
        store.save_scores.assert_not_called()

    @patch("src.consumer.results_consumer.store")
    def test_database_error_propagates_so_batch_is_rewound(self, store):
        store.save_scores.side_effect = RuntimeError("db down")
        with pytest.raises(RuntimeError):
            _process_batch([record(_payload())])
