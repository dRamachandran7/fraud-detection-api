# Fraud Detection API

A fraud detection API backed by Apache Kafka for high-throughput transaction ingestion and an Isolation Forest model trained on a Kaggle credit card dataset.

## Architecture

Client → FastAPI `/ingest` → Kafka `transactions` topic → Consumer scores with model → Kafka `fraud-scores` topic

The API never blocks on inference. It accepts and enqueues. Kafka absorbs traffic spikes. Consumers scale independently.

## Setup

```bash
pip install -r requirements.txt
docker-compose up -d
uvicorn src.api.main:app --reload
python -m src.consumer.score_consumer  # in a separate terminal
```
