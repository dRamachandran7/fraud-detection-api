"""Model components that must be importable wherever the artifact is loaded.

joblib pickles classes by reference, so anything used inside the serialized
pipeline has to live in the package, not in a training script or notebook.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import IsolationForest
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer

ANOMALY_SCORE_COLUMN = "anomaly_score"


def build_isolation_forest(random_state: int | None = None) -> Pipeline:
    """Isolation Forest with log1p on Amount.

    The forest draws splits uniformly between a feature's min and max, so it is
    scale-invariant per feature but not skew-invariant. Raw Amount spans four
    orders of magnitude; log1p stops one large legitimate purchase from being
    isolated in a split or two and dominating the score.
    """
    return Pipeline(
        [
            (
                "amount_log",
                ColumnTransformer(
                    [("log", FunctionTransformer(np.log1p), ["Amount"])],
                    remainder="passthrough",
                    verbose_feature_names_out=False,
                ),
            ),
            (
                "iforest",
                IsolationForest(
                    n_estimators=200,
                    max_samples=256,  # the iForest paper's recommendation
                    random_state=random_state,
                    n_jobs=-1,
                ),
            ),
        ]
    )


class AnomalyScoreFeature(BaseEstimator, TransformerMixin):
    """Appends an Isolation Forest anomaly score as an extra feature column.

    The forest is fitted without labels, so scoring the same rows it was fitted
    on leaks nothing about the target. Higher means more anomalous.
    """

    def __init__(self, random_state: int | None = None):
        self.random_state = random_state

    def fit(self, X: pd.DataFrame, y=None) -> "AnomalyScoreFeature":
        self.forest_ = build_isolation_forest(self.random_state).fit(X)
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        out = X.copy()
        out[ANOMALY_SCORE_COLUMN] = -self.forest_.decision_function(X)
        return out
