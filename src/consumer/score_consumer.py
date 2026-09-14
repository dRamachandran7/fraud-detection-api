"""Consumer: reads transactions from Kafka, scores them, publishes results."""
from __future__ import annotations

import json
import logging
import signal
from types import FrameType

from kafka import KafkaConsumer, KafkaProducer
from kafka.errors import KafkaError

from src import config
from src.inference import load_model, predict

logger = logging.getLogger(__name__)

POLL_TIMEOUT_MS = 1000

_shutdown = False


def _request_shutdown(signum: int, frame: FrameType | None) -> None:
    global _shutdown
    logger.info("Received signal %s, shutting down after current batch", signum)
    _shutdown = True


def _install_signal_handlers() -> None:
    signal.signal(signal.SIGTERM, _request_shutdown)
    signal.signal(signal.SIGINT, _request_shutdown)


def _build_consumer() -> KafkaConsumer:
    return KafkaConsumer(
        config.KAFKA_INGEST_TOPIC,
        bootstrap_servers=config.KAFKA_BOOTSTRAP_SERVERS,
        # Deserialize by hand in the loop so a malformed payload is a
        # dead-letterable message rather than an exception that kills the
        # iterator and takes the process down.
        value_deserializer=None,
        auto_offset_reset="earliest",
        group_id="fraud-scorer",
        # Offsets advance only after a score is durably published, making this
        # at-least-once. With the default timed auto-commit, a crash between
        # commit and publish would silently drop transactions.
        enable_auto_commit=False,
    )


def _build_producer() -> KafkaProducer:
    return KafkaProducer(
        bootstrap_servers=config.KAFKA_BOOTSTRAP_SERVERS,
        value_serializer=lambda v: json.dumps(v).encode("utf-8"),
        key_serializer=lambda k: k.encode("utf-8") if k is not None else None,
        acks="all",
        retries=5,
        max_in_flight_requests_per_connection=1,
    )


def _dead_letter(producer: KafkaProducer, raw: bytes, reason: str) -> None:
    """Park a message we can never score, so the loop is not a crash loop."""
    payload = {
        "reason": reason,
        "raw": raw.decode("utf-8", errors="replace") if raw is not None else None,
    }
    producer.send(config.KAFKA_DLQ_TOPIC, value=payload).get(
        timeout=config.PRODUCER_ACK_TIMEOUT_SECONDS
    )
    logger.warning("Dead-lettered message: %s", reason)


def _parse(raw: bytes) -> tuple[str | None, list[float]]:
    """Decode and validate one message. Raises ValueError if unscoreable."""
    try:
        event = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"malformed JSON: {exc}") from exc

    if not isinstance(event, dict):
        raise ValueError("payload is not an object")

    transaction_id = event.get("transaction_id")
    features = event.get("features")

    if not features:
        raise ValueError("no features")
    if not isinstance(features, list):
        raise ValueError("features is not a list")
    if len(features) != config.EXPECTED_FEATURE_COUNT:
        raise ValueError(
            f"expected {config.EXPECTED_FEATURE_COUNT} features, got {len(features)}"
        )
    if not all(isinstance(f, (int, float)) and not isinstance(f, bool) for f in features):
        raise ValueError("features contains a non-numeric value")

    return transaction_id, features


def _handle_message(model, producer: KafkaProducer, raw: bytes) -> None:
    """Score one message and publish the result.

    Message-level problems are dead-lettered. Broker-level problems propagate
    so the batch goes uncommitted and is retried.
    """
    try:
        transaction_id, features = _parse(raw)
    except ValueError as exc:
        _dead_letter(producer, raw, str(exc))
        return

    try:
        result = predict(model, features)
    except Exception as exc:  # model rejected input we could not anticipate
        _dead_letter(producer, raw, f"scoring failed: {exc}")
        return

    result["transaction_id"] = transaction_id
    producer.send(
        config.KAFKA_SCORES_TOPIC, key=transaction_id, value=result
    ).get(timeout=config.PRODUCER_ACK_TIMEOUT_SECONDS)
    logger.info(
        "Scored transaction %s: is_fraud=%s", transaction_id, result["is_fraud"]
    )


def run_consumer() -> None:
    """Consume raw transactions, score with the model, publish results."""
    _install_signal_handlers()

    model = load_model()
    consumer = _build_consumer()
    producer = _build_producer()

    try:
        while not _shutdown:
            batches = consumer.poll(timeout_ms=POLL_TIMEOUT_MS)
            if not batches:
                continue

            try:
                for records in batches.values():
                    for message in records:
                        _handle_message(model, producer, message.value)
            except KafkaError:
                # Leave the batch uncommitted; it will be redelivered.
                logger.exception("Broker error while processing batch, not committing")
                continue

            consumer.commit()
    finally:
        producer.flush(timeout=config.PRODUCER_ACK_TIMEOUT_SECONDS)
        producer.close(timeout=config.PRODUCER_ACK_TIMEOUT_SECONDS)
        consumer.close()
        logger.info("Consumer shut down cleanly")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    run_consumer()
