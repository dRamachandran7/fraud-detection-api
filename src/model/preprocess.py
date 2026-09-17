"""Shared feature pipeline used by both training and inference.

Both paths must go through `preprocess` so the column set and ordering the
model was fitted on is the column set and ordering it is scored on. The
numeric transforms live inside the serialized sklearn Pipeline rather than
here, so they cannot drift between training and serving.
"""
from __future__ import annotations

import pandas as pd

# V1-V28 are the dataset's PCA components; Amount is raw currency. Time is
# excluded deliberately: it is seconds-since-first-transaction, an artifact of
# how the dataset was assembled rather than a signal that generalizes.
FEATURE_ORDER: list[str] = [f"V{i}" for i in range(1, 29)] + ["Amount"]

LABEL_COLUMN = "Class"


def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    """Select and order the feature columns the model expects.

    Selection rather than dropping: dropping known non-features silently
    tolerates any unexpected column, which is how train/serve skew gets in.

    Raises:
        ValueError: if any expected feature column is missing.
    """
    missing = [col for col in FEATURE_ORDER if col not in df.columns]
    if missing:
        raise ValueError(f"missing feature columns: {missing}")
    return df[FEATURE_ORDER]


def features_to_frame(features: list[float]) -> pd.DataFrame:
    """Build a one-row frame from an ordered feature vector, as sent to /ingest.

    Callers send a bare list, so the ordering contract is positional. This is
    the single place that contract is applied.

    Raises:
        ValueError: if the vector length does not match FEATURE_ORDER.
    """
    if len(features) != len(FEATURE_ORDER):
        raise ValueError(
            f"expected {len(FEATURE_ORDER)} features, got {len(features)}"
        )
    return pd.DataFrame([features], columns=FEATURE_ORDER)
