# Experiment script backing MODEL_IMPROVEMENTS.md (2026-10-26). Not shipped code.
# Run from the repo root: PYTHONPATH=. python notebooks/experiments/ablation.py

import time, warnings
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import average_precision_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from src.model.features import AnomalyScoreFeature
from src.model.preprocess import FEATURE_ORDER
from src.model.train import _strata

df = pd.read_csv("data/creditcard.csv")
X, y, strata = df[FEATURE_ORDER], df.Class.to_numpy(), _strata(df, 4)
past = (df.Time < df.Time.max() / 2).to_numpy()
REG = dict(max_iter=500, learning_rate=0.03, max_leaf_nodes=15, min_samples_leaf=40, l2_regularization=1.0)

def hgb(**kw):
    p = dict(max_iter=300, learning_rate=0.05, random_state=42); p.update(kw)
    return HistGradientBoostingClassifier(**p)
def hybrid(**kw): return Pipeline([("a", AnomalyScoreFeature(random_state=42)), ("c", hgb(**kw))])

configs = {
    "default, early stopping off":    lambda: hybrid(early_stopping=False),
    "only min_samples_leaf=40":       lambda: hybrid(min_samples_leaf=40),
    "only l2_regularization=1.0":     lambda: hybrid(l2_regularization=1.0),
    "only max_leaf_nodes=15":         lambda: hybrid(max_leaf_nodes=15),
    "only lr 0.03 x 500 iters":       lambda: hybrid(max_iter=500, learning_rate=0.03),
    "regularized, supervised only":   lambda: hgb(**REG),
    "regularized + balanced":         lambda: hybrid(**REG, class_weight="balanced"),
}
print(f"{'config':32} {'CV AP':>15} {'time-fwd AP':>12} {'iters used':>11}")
for name, make in configs.items():
    aps, iters = [], []
    for tr, te in StratifiedKFold(5, shuffle=True, random_state=42).split(X, strata):
        m = make().fit(X.iloc[tr], y[tr]); aps.append(average_precision_score(y[te], m.predict_proba(X.iloc[te])[:, 1]))
        clf = m[-1] if isinstance(m, Pipeline) else m; iters.append(clf.n_iter_)
    m = make().fit(X[past], y[past]); fwd = average_precision_score(y[~past], m.predict_proba(X[~past])[:, 1])
    print(f"{name:32} {np.mean(aps):.3f} +/- {np.std(aps):.3f} {fwd:12.3f} {int(np.mean(iters)):>11}")

# the shipped config's early-stopping behaviour, for reference
iters = [hybrid().fit(X.iloc[tr], y[tr])[-1].n_iter_ for tr, _ in StratifiedKFold(5, shuffle=True, random_state=42).split(X, strata)]
print(f"\nshipped config stops after {iters} iterations (of 300)")
