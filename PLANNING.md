# Overview

This repo is a fraud detection API, backed by Apache Kafka for high-throughput transaction ingestion, and an Isolation Forest model to detect fraudulent charges trained on a Kaggle dataset.

The architecture is as follows:

Client → FastAPI `/ingest` endpoint → publishes transaction to Kafka `transactions` topic → returns transaction ID immediately → Consumer reads from Kafka, loads model, scores transaction → publishes result to `fraud-scores` topic.

Kafka is the ingestion buffer. The API never blocks on inference — it accepts transactions as fast as they arrive. Scoring happens asynchronously in the consumer, which can be scaled independently.

## Folder layout

fraud-detection-api/
├── docker-compose.yml          # kafka, zookeeper, api, consumer
├── requirements.txt
├── .env
├── data/
│   └── creditcard.csv          # kaggle dataset
├── notebooks/
│   └── train_model.ipynb       # EDA + training
├── src/
│   ├── model/
│   │   ├── train.py            # trains + serializes model
│   │   ├── preprocess.py       # shared feature pipeline (train + inference)
│   │   └── artifacts/
│   │       └── model.joblib
│   ├── api/
│   │   ├── main.py             # FastAPI app — ingest only, no scoring
│   │   ├── schemas.py          # pydantic request/response models
│   │   └── kafka_producer.py   # publishes raw transactions to Kafka
│   ├── consumer/
│   │   └── score_consumer.py   # reads transactions from Kafka, scores with model, publishes results
│   ├── inference.py            # loads model, runs prediction
│   └── config.py
├── tests/
│   ├── test_api.py
│   ├── test_inference.py
│   └── test_consumer.py
└── README.md