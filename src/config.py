from __future__ import annotations

import os

from dotenv import load_dotenv

load_dotenv()

KAFKA_BOOTSTRAP_SERVERS = os.getenv("KAFKA_BOOTSTRAP_SERVERS", "localhost:9092")
KAFKA_INGEST_TOPIC = os.getenv("KAFKA_INGEST_TOPIC", "transactions")
KAFKA_SCORES_TOPIC = os.getenv("KAFKA_SCORES_TOPIC", "fraud-scores")
KAFKA_DLQ_TOPIC = os.getenv("KAFKA_DLQ_TOPIC", "transactions-dlq")
MODEL_PATH = os.getenv("MODEL_PATH", "src/model/artifacts/model.joblib")

# Number of features the model expects: V1-V28 plus Amount. `preprocess` drops
# Class and Time, so anything else is a client error worth rejecting at ingest.
EXPECTED_FEATURE_COUNT = int(os.getenv("EXPECTED_FEATURE_COUNT", "29"))

# How long /ingest waits for the broker to acknowledge a write before giving up.
PRODUCER_ACK_TIMEOUT_SECONDS = float(os.getenv("PRODUCER_ACK_TIMEOUT_SECONDS", "5"))

# Score store. Read by the results consumer and the read API; deliberately not
# touched by /ingest, so ingestion stays dependent on Kafka alone.
DATABASE_URL = os.getenv(
    "DATABASE_URL", "postgresql://fraud:fraud@localhost:5432/fraud"
)
DB_POOL_MAX_SIZE = int(os.getenv("DB_POOL_MAX_SIZE", "10"))
DB_CONNECT_TIMEOUT_SECONDS = float(os.getenv("DB_CONNECT_TIMEOUT_SECONDS", "10"))

# Cap on rows returned by the flagged-transactions listing.
MAX_LIST_LIMIT = int(os.getenv("MAX_LIST_LIMIT", "1000"))
