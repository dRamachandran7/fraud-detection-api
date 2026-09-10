"""Tests for the score consumer."""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np

from src.inference import predict


class TestConsumerScoring:
    def test_predict_returns_expected_keys(self):
        model = MagicMock()
        model.decision_function.return_value = np.array([0.5])
        model.predict.return_value = np.array([1])

        result = predict(model, [0.1] * 29)
        assert "is_fraud" in result
        assert "score" in result

    def test_fraud_flagged_on_negative_prediction(self):
        model = MagicMock()
        model.decision_function.return_value = np.array([-0.5])
        model.predict.return_value = np.array([-1])

        result = predict(model, [0.1] * 29)
        assert result["is_fraud"] is True

    def test_clean_transaction_not_flagged(self):
        model = MagicMock()
        model.decision_function.return_value = np.array([0.3])
        model.predict.return_value = np.array([1])

        result = predict(model, [0.1] * 29)
        assert result["is_fraud"] is False
