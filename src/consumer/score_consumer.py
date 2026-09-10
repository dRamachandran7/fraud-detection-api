"""Consumer: reads transactions from Kafka, scores them, publishes results."""
from __future__ import annotations

import json
import logging

from kafka import KafkaConsumer, KafkaProducer

from src import config
from src.inference import load_model, predict

logger = logging.getLogger(__name__)


def run_consumer() -> None:
    """Consume raw transactions, score with the model, publish results."""
    model = load_model()

    consumer = KafkaConsumer(
        config.KAFKA_INGEST_TOPIC,
        bootstrap_servers=config.KAFKA_BOOTSTRAP_SERVERS,
        value_deserializer=lambda m: json.loads(m.decode("utf-8")),
        auto_offset_reset="earliest",
        group_id="fraud-scorer",
    )

    producer = KafkaProducer(
        bootstrap_servers=config.KAFKA_BOOTSTRAP_SERVERS,
        value_serializer=lambda v: json.dumps(v).encode("utf-8"),
    )

    for message in consumer:
        event = message.value
        transaction_id = event.get("transaction_id")
        features = event.get("features")

        if not features:
            logger.warning("Skipping message with no features: %s", transaction_id)
            continue

        result = predict(model, features)
        result["transaction_id"] = transaction_id

        producer.send(config.KAFKA_SCORES_TOPIC, value=result)
        logger.info("Scored transaction %s: is_fraud=%s", transaction_id, result["is_fraud"])


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    run_consumer()
