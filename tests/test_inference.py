"""Tests for the inference module."""
from __future__ import annotations

from unittest.mock import MagicMock

import numpy as np
import pytest

from src.inference import predict


def make_artifact(fraud_probability: float, threshold: float = 0.5) -> dict:
    """An artifact whose pipeline returns a fixed probability of fraud."""
    pipeline = MagicMock()
    # One row of output per input row, so batch scoring works against the mock.
    pipeline.predict_proba.side_effect = lambda X: np.tile(
        [1 - fraud_probability, fraud_probability], (len(X), 1)
    )
    return {
        "pipeline": pipeline,
        "threshold": threshold,
        "score_type": "fraud_probability",
        "feature_order": [f"V{i}" for i in range(1, 29)] + ["Amount"],
        "version": "test-version",
    }


class TestPredict:
    def test_low_probability_not_fraud(self):
        result = predict(make_artifact(0.02), [0.1] * 29)
        assert not result["is_fraud"]

    def test_high_probability_is_fraud(self):
        result = predict(make_artifact(0.9), [0.1] * 29)
        assert result["is_fraud"]

    def test_score_is_float(self):
        result = predict(make_artifact(0.42), [0.1] * 29)
        assert isinstance(result["score"], float)
        assert result["score"] == pytest.approx(0.42)

    def test_model_version_is_stamped_on_result(self):
        result = predict(make_artifact(0.42), [0.1] * 29)
        assert result["model_version"] == "test-version"

    def test_threshold_from_artifact_decides_not_half(self):
        """The same probability flips with the tuned threshold, not at 0.5."""
        assert predict(make_artifact(0.3, threshold=0.4), [0.1] * 29)["is_fraud"] is False
        assert predict(make_artifact(0.3, threshold=0.2), [0.1] * 29)["is_fraud"] is True

    def test_wrong_feature_count_rejected(self):
        with pytest.raises(ValueError, match="expected 29 features, got 5"):
            predict(make_artifact(0.5), [0.1] * 5)

    def test_features_reach_pipeline_as_named_ordered_frame(self):
        """Regression: a bare ndarray let column order drift from training."""
        artifact = make_artifact(0.5)
        predict(artifact, list(range(29)))

        frame = artifact["pipeline"].predict_proba.call_args.args[0]
        assert list(frame.columns) == artifact["feature_order"]
        assert frame.iloc[0]["V1"] == 0
        assert frame.iloc[0]["Amount"] == 28
