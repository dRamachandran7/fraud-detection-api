"""Tests for the inference module."""
from __future__ import annotations

from unittest.mock import MagicMock

import numpy as np
import pytest

from src.inference import predict


class TestPredict:
    def _make_model(self, decision_val: float, predict_val: int) -> MagicMock:
        model = MagicMock()
        model.decision_function.return_value = np.array([decision_val])
        model.predict.return_value = np.array([predict_val])
        return model

    def test_normal_transaction_not_fraud(self):
        model = self._make_model(0.5, 1)
        result = predict(model, [0.1] * 29)
        assert not result["is_fraud"]

    def test_anomalous_transaction_is_fraud(self):
        model = self._make_model(-0.3, -1)
        result = predict(model, [0.1] * 29)
        assert result["is_fraud"]

    def test_score_is_float(self):
        model = self._make_model(0.42, 1)
        result = predict(model, [0.1] * 29)
        assert isinstance(result["score"], float)
        assert result["score"] == pytest.approx(0.42)
