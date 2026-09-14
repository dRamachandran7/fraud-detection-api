"""Load the trained model and run predictions."""
from __future__ import annotations

import joblib
import numpy as np

from src import config


def load_model():
    """Load the serialized Isolation Forest model."""
    return joblib.load(config.MODEL_PATH)


def predict(model, features: list[float]) -> dict:
    """Score a single transaction. Returns dict with is_fraud and score."""
    arr = np.array(features).reshape(1, -1)
    raw_score = model.decision_function(arr)[0]
    prediction = model.predict(arr)[0]

    return {
        "is_fraud": bool(prediction == -1),
        "score": float(raw_score),
    }
