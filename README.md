# Fraud Detection API

A fraud detection API backed by Apache Kafka for high-throughput transaction ingestion and an Isolation Forest model trained on a Kaggle credit card dataset.

## Architecture

Client → FastAPI `/ingest` → Kafka `transactions` topic → Consumer scores with model → Kafka `fraud-scores` topic

The API never blocks on inference. It accepts and enqueues. Kafka absorbs traffic spikes. Consumers scale independently.

## Setup

```bash
pip install -r requirements-dev.txt
python -m src.model.train          # requires data/creditcard.csv
docker-compose up -d               # kafka + topic creation + api + consumer
```

To run the API and consumer on the host instead of in containers, keep
`KAFKA_BOOTSTRAP_SERVERS=localhost:9092` in `.env` and start them directly:

```bash
uvicorn src.api.main:app --reload
python -m src.consumer.score_consumer  # in a separate terminal
```

Compose overrides that variable to `kafka:29092`, since `localhost` inside a
container is the container itself.

## Delivery guarantees

`/ingest` blocks until the broker acknowledges the write with `acks=all`, so a
returned `transaction_id` means the transaction is durably queued; a broker
failure returns 503 rather than a false acknowledgement. The consumer commits
offsets only after a score is published, making the pipeline at-least-once —
downstream consumers should dedupe on `transaction_id`. Messages that can never
be scored (malformed JSON, wrong feature count) go to `transactions-dlq`.

## Tests

```bash
pytest
```
