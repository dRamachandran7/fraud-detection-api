"""End-to-end test of training through to scoring, on a synthetic dataset."""
from __future__ import annotations

from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from src.inference import ArtifactError, load_model, predict
from src.model.preprocess import FEATURE_ORDER
from src.model.train import train


@pytest.fixture()
def dataset(tmp_path):
    """A frame shaped like creditcard.csv: separable, low-value frauds."""
    rng = np.random.default_rng(0)
    n, n_fraud = 4000, 40

    V = rng.normal(0, 1, size=(n, 28))
    amount = rng.lognormal(3, 1.2, size=n)
    y = np.zeros(n, dtype=int)
    fraud_idx = rng.choice(n, n_fraud, replace=False)
    y[fraud_idx] = 1
    V[fraud_idx] += rng.normal(4.0, 1.0, size=(n_fraud, 28))
    amount[fraud_idx] = rng.lognormal(1.0, 0.8, size=n_fraud)

    df = pd.DataFrame(V, columns=[f"V{i}" for i in range(1, 29)])
    df["Time"] = np.arange(n)
    df["Amount"] = amount
    df["Class"] = y

    path = tmp_path / "creditcard.csv"
    df.to_csv(path, index=False)
    return path, df


class TestTrain:
    def test_artifact_has_everything_inference_needs(self, dataset, tmp_path):
        data_path, _ = dataset
        out = tmp_path / "model.joblib"

        artifact = train(str(data_path), str(out), alert_rate=0.01, cv_folds=0)

        assert out.exists()
        assert artifact["feature_order"] == FEATURE_ORDER
        assert isinstance(artifact["threshold"], float)
        assert artifact["version"]
        assert artifact["metrics"]["n_val_fraud"] > 0

    def test_separable_fraud_is_detected(self, dataset, tmp_path):
        """Guards the pipeline wiring, not real-world accuracy."""
        data_path, _ = dataset
        artifact = train(str(data_path), str(tmp_path / "m.joblib"), alert_rate=0.01, cv_folds=0)

        assert artifact["metrics"]["average_precision"] > 0.5
        assert artifact["metrics"]["recall_at_threshold"] > 0.5

    def test_missing_dataset_fails_with_a_useful_message(self, tmp_path):
        with pytest.raises(FileNotFoundError, match="Download the Kaggle"):
            train(str(tmp_path / "nope.csv"), str(tmp_path / "m.joblib"))

    def test_time_column_is_excluded_from_features(self, dataset, tmp_path):
        data_path, _ = dataset
        artifact = train(str(data_path), str(tmp_path / "m.joblib"), alert_rate=0.01, cv_folds=0)
        assert "Time" not in artifact["feature_order"]


class TestRoundTrip:
    def test_trained_artifact_loads_and_scores(self, dataset, tmp_path):
        """train -> joblib -> load_model -> predict, with no shape drift."""
        data_path, df = dataset
        out = tmp_path / "model.joblib"
        train(str(data_path), str(out), alert_rate=0.01, cv_folds=0)

        with patch("src.config.MODEL_PATH", str(out)):
            artifact = load_model()

        clean = df[df.Class == 0].iloc[0][FEATURE_ORDER].tolist()
        fraud = df[df.Class == 1].iloc[0][FEATURE_ORDER].tolist()

        assert predict(artifact, clean)["is_fraud"] is False
        assert predict(artifact, fraud)["is_fraud"] is True

    def test_stale_bare_estimator_artifact_is_rejected(self, tmp_path):
        """A pre-artifact model.joblib must fail loudly, not score garbage."""
        import joblib
        from sklearn.ensemble import IsolationForest

        stale = tmp_path / "stale.joblib"
        joblib.dump(IsolationForest().fit(np.random.rand(50, 29)), stale)

        with patch("src.config.MODEL_PATH", str(stale)):
            with pytest.raises(ArtifactError, match="Retrain"):
                load_model()


class TestHybrid:
    def test_anomaly_score_is_a_classifier_input(self, dataset, tmp_path):
        data_path, _ = dataset
        artifact = train(str(data_path), str(tmp_path / "m.joblib"), alert_rate=0.01, cv_folds=0)

        steps = dict(artifact["pipeline"].named_steps)
        assert "anomaly_score" in steps
        assert "anomaly_score" in steps["classifier"].feature_names_in_
        assert artifact["score_type"] == "fraud_probability"

    def test_supervised_only_omits_anomaly_score(self, dataset, tmp_path):
        data_path, _ = dataset
        artifact = train(
            str(data_path), str(tmp_path / "m.joblib"), alert_rate=0.01,
            include_anomaly_score=False, cv_folds=0,
        )
        assert "anomaly_score" not in artifact["pipeline"].named_steps
        assert artifact["metrics"]["model"] == "supervised"

    def test_artifact_with_unknown_score_type_is_rejected(self, dataset, tmp_path):
        """An old Isolation-Forest-only artifact must not be thresholded as a probability."""
        import joblib

        data_path, _ = dataset
        out = tmp_path / "m.joblib"
        artifact = train(str(data_path), str(out), alert_rate=0.01, cv_folds=0)
        artifact["score_type"] = "anomaly_decision_function"
        joblib.dump(artifact, out)

        with patch("src.config.MODEL_PATH", str(out)):
            with pytest.raises(ArtifactError, match="score_type"):
                load_model()


class TestStrata:
    def test_strata_combine_label_and_time_bucket(self):
        from src.model.train import _strata

        df = pd.DataFrame({"Time": [0, 3600 * 5, 3600 * 5], "Class": [0, 0, 1]})
        assert list(_strata(df, bucket_hours=4)) == ["0_0", "1_0", "1_1"]

    def test_artifact_records_cross_validation(self, dataset, tmp_path):
        data_path, _ = dataset
        artifact = train(str(data_path), str(tmp_path / "m.joblib"), alert_rate=0.01, cv_folds=3)

        cv = artifact["metrics"]["cross_validation"]
        assert cv["folds"] == 3
        assert 0 <= cv["average_precision"]["mean"] <= 1
