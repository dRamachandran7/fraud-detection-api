"""Train an Isolation Forest model and serialize it to disk."""
from __future__ import annotations

import joblib
import pandas as pd
from sklearn.ensemble import IsolationForest

from src.model.preprocess import preprocess


def train(data_path: str = "data/creditcard.csv", output_path: str = "src/model/artifacts/model.joblib") -> None:
    df = pd.read_csv(data_path)
    X = preprocess(df)

    model = IsolationForest(n_estimators=100, contamination="auto", random_state=42)
    model.fit(X)

    joblib.dump(model, output_path)
    print(f"Model saved to {output_path}")


if __name__ == "__main__":
    train()
