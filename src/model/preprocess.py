"""Shared feature pipeline used by both training and inference."""
from __future__ import annotations

import pandas as pd


def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    """Return the feature matrix expected by the Isolation Forest model."""
    # Drop non-feature columns if present
    drop_cols = [col for col in ["Class", "Time"] if col in df.columns]
    X = df.drop(columns=drop_cols)
    return X
