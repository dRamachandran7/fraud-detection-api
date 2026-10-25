"""Load the trained model artifact and run predictions."""
from __future__ import annotations

import joblib

from src import config
from src.model.preprocess import features_to_frame, preprocess

REQUIRED_ARTIFACT_KEYS = {"pipeline", "threshold", "score_type", "feature_order", "version"}
SUPPORTED_SCORE_TYPE = "fraud_probability"


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

    if artifact["score_type"] != SUPPORTED_SCORE_TYPE:
        raise ArtifactError(
            f"artifact score_type is {artifact['score_type']!r}; this service "
            f"only knows how to threshold {SUPPORTED_SCORE_TYPE!r}."
        )

    expected = len(artifact["feature_order"])
    if expected != config.EXPECTED_FEATURE_COUNT:
        raise ArtifactError(
            f"artifact expects {expected} features but EXPECTED_FEATURE_COUNT "
            f"is {config.EXPECTED_FEATURE_COUNT}; the API would accept vectors "
            "the model cannot score."
        )

    return artifact


def predict_batch(artifact: dict, rows: list[list[float]]) -> list[dict]:
    """Score many transactions in one model call, in input order.

    Each result has is_fraud, score and model_version. `score` is the model's
    probability of fraud, 0 to 1. `is_fraud` compares it against the
    artifact's threshold, which was set from the alert budget at train time —
    not against 0.5.
    """
    if not rows:
        return []

    X = preprocess(features_to_frame(rows))
    scores = artifact["pipeline"].predict_proba(X)[:, 1]
    if len(scores) != len(rows):
        raise RuntimeError(f"model returned {len(scores)} scores for {len(rows)} rows")

    threshold = artifact["threshold"]
    version = artifact["version"]
    return [
        {"is_fraud": bool(s >= threshold), "score": float(s), "model_version": version}
        for s in scores
    ]


def predict(artifact: dict, features: list[float]) -> dict:
    """Score a single transaction. See predict_batch."""
    return predict_batch(artifact, [features])[0]
