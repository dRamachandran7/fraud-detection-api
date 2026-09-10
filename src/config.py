from __future__ import annotations

import os

from dotenv import load_dotenv

load_dotenv()

KAFKA_BOOTSTRAP_SERVERS = os.getenv("KAFKA_BOOTSTRAP_SERVERS", "localhost:9092")
KAFKA_INGEST_TOPIC = os.getenv("KAFKA_INGEST_TOPIC", "transactions")
KAFKA_SCORES_TOPIC = os.getenv("KAFKA_SCORES_TOPIC", "fraud-scores")
MODEL_PATH = os.getenv("MODEL_PATH", "src/model/artifacts/model.joblib")
