# Experiment script backing MODEL_IMPROVEMENTS.md (2026-10-26). Not shipped code.
# Run from the repo root: PYTHONPATH=. python notebooks/experiments/experiments.py

"""Quick experiments to rank improvement ideas. Not shipped code."""
import time, warnings
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from src.model.features import AnomalyScoreFeature
from src.model.preprocess import FEATURE_ORDER
from src.model.train import _strata

df = pd.read_csv("data/creditcard.csv")
y = df.Class.to_numpy()
strata = _strata(df, 4)
base = df[FEATURE_ORDER].copy()
with_hour = base.copy()
with_hour["hour"] = (df.Time % 86400) // 3600   # time of day, not the raw dataset clock
past = (df.Time < df.Time.max() / 2).to_numpy()

def hgb(**kw):
    p = dict(max_iter=300, learning_rate=0.05, random_state=42); p.update(kw)
    return HistGradientBoostingClassifier(**p)

def hybrid(**kw):
    return Pipeline([("a", AnomalyScoreFeature(random_state=42)), ("c", hgb(**kw))])

configs = {
    "hybrid (shipped)":              (base, lambda: hybrid()),
    "supervised only":               (base, lambda: hgb()),
    "logistic regression baseline":  (base, lambda: Pipeline([("s", StandardScaler()), ("c", LogisticRegression(max_iter=2000, C=0.1))])),
    "hybrid + class_weight=balanced":(base, lambda: hybrid(class_weight="balanced")),
    "hybrid, stronger regularization":(base, lambda: hybrid(max_iter=500, learning_rate=0.03, max_leaf_nodes=15, min_samples_leaf=40, l2_regularization=1.0)),
    "hybrid + hour-of-day feature":  (with_hour, lambda: hybrid()),
}

oof = {}
print(f"{'config':34} {'CV AP':>15} {'time-fwd AP':>12} {'secs':>5}")
for name, (X, make) in configs.items():
    t = time.time(); aps = []; o = np.zeros(len(y))
    for tr, te in StratifiedKFold(5, shuffle=True, random_state=42).split(X, strata):
        m = make().fit(X.iloc[tr], y[tr]); s = m.predict_proba(X.iloc[te])[:, 1]
        o[te] = s; aps.append(average_precision_score(y[te], s))
    m = make().fit(X[past], y[past]); fwd = average_precision_score(y[~past], m.predict_proba(X[~past])[:, 1])
    oof[name] = o
    print(f"{name:34} {np.mean(aps):.3f} +/- {np.std(aps):.3f} {fwd:12.3f} {time.time()-t:5.0f}")

