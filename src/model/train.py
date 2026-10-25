"""Train the hybrid fraud model and serialize it as a versioned artifact.

Hybrid: a gradient-boosted classifier trained on the fraud labels, with an
Isolation Forest anomaly score added as one more input feature. The classifier
learns which fraud patterns the labels show; the anomaly score gives it a
signal for transactions that are unusual in ways the labels may not cover.
"""
from __future__ import annotations

import argparse
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline

from src.model.features import AnomalyScoreFeature
from src.model.preprocess import FEATURE_ORDER, LABEL_COLUMN, preprocess

DEFAULT_DATA_PATH = "data/creditcard.csv"
DEFAULT_OUTPUT_PATH = "src/model/artifacts/model.joblib"

# Share of scored traffic to flag. This is an alert budget, not an estimate of
# the fraud rate: it should reflect how many reviews per day the team can
# actually absorb. The dataset's true positive rate is ~0.17%, so precision at
# this budget can never exceed ~35%.
DEFAULT_ALERT_RATE = 0.005

# Splits are stratified on label x time bucket, not label alone. The fraud rate
# swings ~8x across the dataset's 48 hours (0.07% to 0.57% per 4h bucket, with
# low-traffic overnight hours carrying the most fraud), so a label-only split
# can over- or under-sample the night and move the measured precision.
TIME_COLUMN = "Time"
DEFAULT_TIME_BUCKET_HOURS = 4
DEFAULT_CV_FOLDS = 5

# Inference reads this to know how to interpret `score` and the threshold.
SCORE_TYPE = "fraud_probability"


def build_pipeline(random_state: int, include_anomaly_score: bool = True) -> Pipeline:
    """The shipped model. `include_anomaly_score=False` is supervised-only.

    No class weighting: it would distort predict_proba, which is served as a
    probability. Ranking — all the alert-budget threshold needs — is fine
    without it.
    """
    steps = []
    if include_anomaly_score:
        steps.append(("anomaly_score", AnomalyScoreFeature(random_state=random_state)))
    steps.append(
        (
            "classifier",
            HistGradientBoostingClassifier(
                max_iter=300, learning_rate=0.05, random_state=random_state
            ),
        )
    )
    return Pipeline(steps)


def fraud_scores(pipeline: Pipeline, X: pd.DataFrame) -> np.ndarray:
    """Probability of fraud per row. Higher = more likely fraud."""
    return pipeline.predict_proba(X)[:, 1]


def _strata(df: pd.DataFrame, bucket_hours: int) -> np.ndarray:
    """Label x time-bucket strata, so every split covers the whole timeline.

    Time is used only to shape the split; it is never a model feature.
    """
    buckets = (df[TIME_COLUMN] // (bucket_hours * 3600)).astype(int)
    return (buckets.astype(str) + "_" + df[LABEL_COLUMN].astype(str)).to_numpy()


def _evaluate(scores: np.ndarray, y: np.ndarray, alert_rate: float) -> dict:
    """Metrics for one scored split. Higher score = more likely fraud.

    The threshold is a score percentile and never looks at labels, so choosing
    it and measuring precision/recall on the same rows is not biased.
    """
    threshold = float(np.percentile(scores, 100 - alert_rate * 100))
    flagged = scores >= threshold
    n_flagged = int(flagged.sum())
    n_fraud = int((y == 1).sum())
    caught = int((flagged & (y == 1)).sum())
    return {
        "average_precision": float(average_precision_score(y, scores)),
        "roc_auc": float(roc_auc_score(y, scores)),
        "threshold": threshold,
        "precision_at_threshold": caught / n_flagged if n_flagged else 0.0,
        "recall_at_threshold": caught / n_fraud if n_fraud else 0.0,
        "caught": caught,
        "n_fraud": n_fraud,
    }


def _cross_validate(
    X: pd.DataFrame,
    y: np.ndarray,
    strata: np.ndarray,
    alert_rate: float,
    folds: int,
    include_anomaly_score: bool,
    random_state: int,
) -> dict:
    """Mean and spread of the metrics across stratified folds.

    A single holdout's average precision moves by several points with the
    split alone, so one number is not enough to compare two configurations.
    """
    per_fold = []
    splitter = StratifiedKFold(n_splits=folds, shuffle=True, random_state=random_state)
    for train_idx, test_idx in splitter.split(X, strata):
        pipeline = build_pipeline(random_state, include_anomaly_score)
        pipeline.fit(X.iloc[train_idx], y[train_idx])
        per_fold.append(
            _evaluate(fraud_scores(pipeline, X.iloc[test_idx]), y[test_idx], alert_rate)
        )

    summary = {"folds": folds}
    for key in ("average_precision", "roc_auc", "precision_at_threshold", "recall_at_threshold"):
        values = [m[key] for m in per_fold]
        summary[key] = {"mean": float(np.mean(values)), "std": float(np.std(values))}
    return summary


def _time_forward(
    df: pd.DataFrame,
    X: pd.DataFrame,
    y: np.ndarray,
    alert_rate: float,
    include_anomaly_score: bool,
    random_state: int,
) -> dict:
    """Fit on the first half of the timeline, test on the second.

    The closest this dataset gets to production, where the model only ever
    scores transactions from after it was trained. Expect it below the CV mean.
    """
    cutoff = df[TIME_COLUMN].max() / 2
    past = (df[TIME_COLUMN] < cutoff).to_numpy()
    pipeline = build_pipeline(random_state, include_anomaly_score)
    pipeline.fit(X[past], y[past])
    return _evaluate(fraud_scores(pipeline, X[~past]), y[~past], alert_rate)


def _resolve_version() -> str:
    """Version the artifact by git sha so a score traces back to its model."""
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    try:
        sha = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return stamp
    return f"{stamp}-{sha}"


def train(
    data_path: str = DEFAULT_DATA_PATH,
    output_path: str = DEFAULT_OUTPUT_PATH,
    alert_rate: float = DEFAULT_ALERT_RATE,
    include_anomaly_score: bool = True,
    random_state: int = 42,
    time_bucket_hours: int = DEFAULT_TIME_BUCKET_HOURS,
    cv_folds: int = DEFAULT_CV_FOLDS,
) -> dict:
    """Fit, evaluate, pick a threshold, and serialize."""
    if not Path(data_path).exists():
        raise FileNotFoundError(
            f"{data_path} not found. Download the Kaggle credit card fraud "
            "dataset and place it there before training."
        )

    df = pd.read_csv(data_path)
    for required in (LABEL_COLUMN, TIME_COLUMN):
        if required not in df.columns:
            raise ValueError(f"{data_path} has no {required} column")

    X = preprocess(df)
    y = df[LABEL_COLUMN].to_numpy()
    strata = _strata(df, time_bucket_hours)

    cv = None
    if cv_folds > 1:
        cv = _cross_validate(
            X, y, strata, alert_rate, cv_folds, include_anomaly_score, random_state
        )

    forward = _time_forward(df, X, y, alert_rate, include_anomaly_score, random_state)

    # The shipped model: fit on the training share, threshold from the holdout.
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.3, stratify=strata, random_state=random_state
    )
    pipeline = build_pipeline(random_state, include_anomaly_score)
    pipeline.fit(X_train, y_train)

    holdout = _evaluate(fraud_scores(pipeline, X_val), y_val, alert_rate)

    metrics = {
        "average_precision": holdout["average_precision"],
        "roc_auc": holdout["roc_auc"],
        "precision_at_threshold": holdout["precision_at_threshold"],
        "recall_at_threshold": holdout["recall_at_threshold"],
        "alert_rate": alert_rate,
        "model": "hybrid" if include_anomaly_score else "supervised",
        "split": f"stratified on label x {time_bucket_hours}h time bucket",
        "n_train": int(len(X_train)),
        "n_val": int(len(X_val)),
        "n_val_fraud": holdout["n_fraud"],
        "cross_validation": cv,
        "time_forward": {
            k: forward[k]
            for k in ("average_precision", "precision_at_threshold", "recall_at_threshold")
        },
    }

    artifact = {
        "pipeline": pipeline,
        "threshold": holdout["threshold"],
        "score_type": SCORE_TYPE,
        "feature_order": FEATURE_ORDER,
        "version": _resolve_version(),
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "metrics": metrics,
    }

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(artifact, output_path)

    # Average precision is the number to watch. ROC-AUC looks flattering at
    # 0.17% positives and says very little about review workload.
    print(f"Model saved to {output_path} (version {artifact['version']}, {metrics['model']})")
    print(f"  holdout ({metrics['split']}):")
    print(f"    average precision : {holdout['average_precision']:.4f}")
    print(f"    roc auc           : {holdout['roc_auc']:.4f}")
    print(
        f"    at {alert_rate:.2%} alert rate: precision {holdout['precision_at_threshold']:.4f}, "
        f"recall {holdout['recall_at_threshold']:.4f} "
        f"({holdout['caught']}/{holdout['n_fraud']} frauds caught)"
    )
    if cv:
        print(f"  {cv['folds']}-fold cross-validation (mean +/- std):")
        for key, label in (
            ("average_precision", "average precision"),
            ("precision_at_threshold", "precision at alert"),
            ("recall_at_threshold", "recall at alert"),
        ):
            print(f"    {label:<18}: {cv[key]['mean']:.4f} +/- {cv[key]['std']:.4f}")
    print("  time-forward (fit first 24h, test last 24h):")
    print(
        f"    average precision {forward['average_precision']:.4f}, "
        f"precision {forward['precision_at_threshold']:.4f}, "
        f"recall {forward['recall_at_threshold']:.4f}"
    )
    return artifact


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-path", default=DEFAULT_DATA_PATH)
    parser.add_argument("--output-path", default=DEFAULT_OUTPUT_PATH)
    parser.add_argument(
        "--alert-rate",
        type=float,
        default=DEFAULT_ALERT_RATE,
        help="share of traffic to flag; set from review capacity",
    )
    parser.add_argument(
        "--no-anomaly-score",
        action="store_true",
        help="supervised classifier only, without the Isolation Forest feature",
    )
    parser.add_argument(
        "--time-bucket-hours",
        type=int,
        default=DEFAULT_TIME_BUCKET_HOURS,
        help="width of the time buckets the split is stratified on",
    )
    parser.add_argument(
        "--cv-folds",
        type=int,
        default=DEFAULT_CV_FOLDS,
        help="cross-validation folds for the metric spread; 0 or 1 to skip",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    train(
        data_path=args.data_path,
        output_path=args.output_path,
        alert_rate=args.alert_rate,
        include_anomaly_score=not args.no_anomaly_score,
        time_bucket_hours=args.time_bucket_hours,
        cv_folds=args.cv_folds,
    )
