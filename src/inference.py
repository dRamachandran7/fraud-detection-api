"""Load the trained model artifact and run predictions."""
from __future__ import annotations

import joblib

from src import config
from src.model.preprocess import features_to_frame, preprocess

REQUIRED_ARTIFACT_KEYS = {"pipeline", "threshold", "feature_order", "version"}


class ArtifactError(RuntimeError):
    """Raised when the serialized artifact is missing or malformed."""


def load_model() -> dict:
    """Load the serialized model artifact.

    Returns the full artifact dict, not a bare estimator: the decision
    threshold and model version travel with the pipeline so scores stay
    traceable to the artifact that produced them.
    """
    artifact = joblib.load(config.MODEL_PATH)

    if not isinstance(artifact, dict) or not REQUIRED_ARTIFACT_KEYS <= artifact.keys():
        raise ArtifactError(
            f"{config.MODEL_PATH} is not a current model artifact. "
            "Retrain with `python -m src.model.train`."
        )

    expected = len(artifact["feature_order"])
    if expected != config.EXPECTED_FEATURE_COUNT:
        raise ArtifactError(
            f"artifact expects {expected} features but EXPECTED_FEATURE_COUNT "
            f"is {config.EXPECTED_FEATURE_COUNT}; the API would accept vectors "
            "the model cannot score."
        )

    return artifact


def predict(artifact: dict, features: list[float]) -> dict:
    """Score a single transaction. Returns dict with is_fraud, score, version."""
    X = preprocess(features_to_frame(features))

    # decision_function is signed: lower is more anomalous. The threshold was
    # chosen on held-out data at train time, so we compare against it directly
    # rather than calling .predict(), whose boundary comes from `contamination`
    # and is not tuned to the alert budget.
    raw_score = artifact["pipeline"].decision_function(X)[0]

    return {
        "is_fraud": bool(raw_score < artifact["threshold"]),
        "score": float(raw_score),
        "model_version": artifact["version"],
    }
