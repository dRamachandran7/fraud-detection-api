"""Publishes raw transactions to Kafka for async scoring."""
from __future__ import annotations

import logging
import threading
import time
import uuid

from prometheus_client import Histogram

from src import config
from src.kafka_serde import JsonSerializer, StringSerializer
from src.api.schemas import IngestRequest

logger = logging.getLogger(__name__)


PUBLISH_SECONDS = Histogram(
    "fraud_ingest_publish_seconds",
    "Time from send to broker ack on /ingest, by outcome",
    ["outcome"],
)


class PublishError(RuntimeError):
    """Raised when a transaction could not be durably written to Kafka."""


_producer = None
_producer_lock = threading.Lock()


def _build_producer():
    from kafka import KafkaProducer

    return KafkaProducer(
        bootstrap_servers=config.KAFKA_BOOTSTRAP_SERVERS,
        value_serializer=JsonSerializer(),
        key_serializer=StringSerializer(),
        # Durability: a transaction we have already acknowledged to the caller
        # must not be lost to a leader election or a transient broker error.
        acks="all",
        retries=5,
        # Retries can reorder messages unless only one request is in flight.
        max_in_flight_requests_per_connection=1,
    )


def _get_producer():
    global _producer
    if _producer is None:
        with _producer_lock:
            if _producer is None:
                _producer = _build_producer()
    return _producer


def publish_transaction(request: IngestRequest) -> str:
    """Publish a transaction to the ingest topic. Returns a transaction ID.

    Blocks until the broker acknowledges the write, so a returned ID always
    means the transaction is durably queued. Raises PublishError otherwise.

    This is a blocking call: callers on the event loop must not await it
    directly (see the sync endpoint in main.py, which FastAPI runs in a
    threadpool).
    """
    transaction_id = str(uuid.uuid4())
    event = {
        "transaction_id": transaction_id,
        "features": request.features,
    }

    from kafka.errors import KafkaError

    started = time.perf_counter()
    try:
        future = _get_producer().send(
            config.KAFKA_INGEST_TOPIC,
            key=transaction_id,
            value=event,
        )
        future.get(timeout=config.PRODUCER_ACK_TIMEOUT_SECONDS)
    except KafkaError as exc:
        PUBLISH_SECONDS.labels("failed").observe(time.perf_counter() - started)
        logger.error("Failed to publish transaction %s: %s", transaction_id, exc)
        raise PublishError(str(exc)) from exc
    PUBLISH_SECONDS.labels("acked").observe(time.perf_counter() - started)

    return transaction_id


def shutdown_producer() -> None:
    """Flush buffered records and close the producer. Safe to call twice."""
    global _producer
    with _producer_lock:
        if _producer is None:
            return
        try:
            _producer.flush(timeout=config.PRODUCER_ACK_TIMEOUT_SECONDS)
        finally:
            _producer.close(timeout=config.PRODUCER_ACK_TIMEOUT_SECONDS)
            _producer = None
