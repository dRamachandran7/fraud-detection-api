"""Publishes raw transactions to Kafka for async scoring."""
from __future__ import annotations

import json
import logging
import uuid

from src import config
from src.api.schemas import IngestRequest

logger = logging.getLogger(__name__)

_producer = None


def _get_producer():
    global _producer
    if _producer is None:
        from kafka import KafkaProducer

        _producer = KafkaProducer(
            bootstrap_servers=config.KAFKA_BOOTSTRAP_SERVERS,
            value_serializer=lambda v: json.dumps(v).encode("utf-8"),
        )
    return _producer


def publish_transaction(request: IngestRequest) -> str:
    """Publish a transaction to the ingest topic. Returns a transaction ID."""
    transaction_id = str(uuid.uuid4())
    event = {
        "transaction_id": transaction_id,
        "features": request.features,
    }
    _get_producer().send(config.KAFKA_INGEST_TOPIC, value=event)
    return transaction_id
