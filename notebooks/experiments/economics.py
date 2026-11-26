# Experiment script backing MODEL_IMPROVEMENTS.md (2026-10-26). Not shipped code.
# Run from the repo root: PYTHONPATH=. python notebooks/experiments/economics.py

import warnings
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.pipeline import Pipeline
from src.model.features import AnomalyScoreFeature
from src.model.preprocess import FEATURE_ORDER

df = pd.read_csv("data/creditcard.csv")
past = (df.Time < df.Time.max() / 2).to_numpy()
m = Pipeline([("a", AnomalyScoreFeature(random_state=42)),
              ("c", HistGradientBoostingClassifier(max_iter=500, learning_rate=0.03, max_leaf_nodes=15,
                    min_samples_leaf=40, l2_regularization=1.0, random_state=42))])
m.fit(df.loc[past, FEATURE_ORDER], df.Class[past])
d2 = df[~past].reset_index(drop=True)
p = m.predict_proba(d2[FEATURE_ORDER])[:, 1]
y, amt = d2.Class.to_numpy(), d2.Amount.to_numpy()
fraud_usd = amt[y == 1].sum()

print(f"Day 2: {len(d2):,} transactions, {y.sum()} frauds totalling ${fraud_usd:,.0f} "
      f"(median ${np.median(amt[y==1]):.0f}, mean ${amt[y==1].mean():.0f}, max ${amt[y==1].max():,.0f})")
print(f"Calibration: sum of predicted probabilities {p.sum():.0f} vs {y.sum()} actual frauds\n")

print(f"{'alert rate':>10} {'alerts':>7} {'caught':>7} {'precision':>9} {'recall':>7} {'$ recall':>8}")
order = np.argsort(-p)
for rate in (0.0005, 0.001, 0.0015, 0.002, 0.003, 0.005, 0.01, 0.02):
    k = int(round(rate * len(p))); f = np.zeros(len(p), bool); f[order[:k]] = True
    c = (f & (y == 1))
    print(f"{rate:>10.2%} {k:>7} {c.sum():>7} {c.sum()/k:>9.2f} {c.sum()/y.sum():>7.2f} {amt[c].sum()/fraud_usd:>8.2f}")

print("\nDaily cost = missed fraud $ + alerts x review cost   (no model: all fraud missed)")
def cost(f, rc): return amt[(~f) & (y == 1)].sum() + f.sum() * rc
for rc in (2, 10, 25):
    best = min(((cost(np.isin(np.arange(len(p)), order[:k]), rc), k) for k in range(5, 3000, 5)))
    el = p * amt > rc                     # expected-loss rule: flag if P(fraud) x amount > review cost
    c_el = (el & (y == 1))
    print(f"  review ${rc:>2}: no model ${fraud_usd:,.0f} | best fixed alert rate {best[1]/len(p):.2%} "
          f"({best[1]} alerts) costs ${best[0]:,.0f} | expected-loss rule {el.sum()} alerts, "
          f"{c_el.sum()}/{y.sum()} caught, {amt[c_el].sum()/fraud_usd:.0%} of $, costs ${cost(el, rc):,.0f}")
