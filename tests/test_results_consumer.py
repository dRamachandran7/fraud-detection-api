"""Tests for the results consumer."""
from __future__ import annotations

import json
from unittest.mock import patch

import pytest

from src.consumer.results_consumer import _handle_message, _parse


def _payload(**overrides) -> bytes:
    event = {
        "transaction_id": "txn-1",
        "is_fraud": True,
        "score": -0.2,
        "model_version": "v1",
    }
    event.update(overrides)
    return json.dumps(event).encode("utf-8")


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


class TestHandleMessage:
    @patch("src.consumer.results_consumer.store")
    def test_score_is_persisted(self, store):
        _handle_message(_payload())

        store.save_score.assert_called_once_with(
            transaction_id="txn-1", is_fraud=True, score=-0.2, model_version="v1"
        )

    @patch("src.consumer.results_consumer.store")
    def test_malformed_message_is_skipped_not_raised(self, store):
        _handle_message(b"{nope")
        store.save_score.assert_not_called()

    @patch("src.consumer.results_consumer.store")
    def test_database_error_propagates_so_batch_is_not_committed(self, store):
        store.save_score.side_effect = RuntimeError("db down")
        with pytest.raises(RuntimeError):
            _handle_message(_payload())
