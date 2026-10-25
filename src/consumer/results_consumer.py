"""Consumer: reads scores from Kafka and persists them to the score store.

Separate from the scoring consumer on purpose: persistence failures must not
stall scoring, and the two scale on different pressures (one is CPU-bound on
the model, this one is bound on database writes).
"""
from __future__ import annotations

import json
import logging
import signal
from types import FrameType

from kafka import KafkaConsumer

from src import config
from src.store import scores as store

logger = logging.getLogger(__name__)

POLL_TIMEOUT_MS = 1000
REQUIRED_FIELDS = ("transaction_id", "is_fraud", "score", "model_version")

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
        config.KAFKA_SCORES_TOPIC,
        bootstrap_servers=config.KAFKA_BOOTSTRAP_SERVERS,
        value_deserializer=None,
        auto_offset_reset="earliest",
        group_id="score-persister",
        # Commit only after the write lands, so a crash replays rather than
        # silently dropping a score. Upserts make the replay harmless.
        enable_auto_commit=False,
    )


def _parse(raw: bytes) -> dict:
    """Decode one score message. Raises ValueError if unusable."""
    try:
        event = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"malformed JSON: {exc}") from exc

    if not isinstance(event, dict):
        raise ValueError("payload is not an object")

    missing = [f for f in REQUIRED_FIELDS if event.get(f) is None]
    if missing:
        raise ValueError(f"missing fields: {missing}")

    return event


def _handle_message(raw: bytes) -> None:
    """Persist one score. Malformed messages are logged and skipped.

    Database errors are left to propagate: unlike a bad message, they are
    transient, and the batch should go uncommitted so the score is retried.
    """
    try:
        event = _parse(raw)
    except ValueError as exc:
        # No DLQ here: these messages came from our own scoring consumer, so a
        # malformed one is a bug on our side worth alerting on, not client input.
        logger.error("Unusable score message, skipping: %s", exc)
        return

    store.save_score(
        transaction_id=event["transaction_id"],
        is_fraud=bool(event["is_fraud"]),
        score=float(event["score"]),
        model_version=event["model_version"],
    )
    logger.debug("Persisted score for %s", event["transaction_id"])


def run_consumer() -> None:
    """Consume scored transactions and write them to the store."""
    _install_signal_handlers()

    store.init_schema()
    consumer = _build_consumer()

    try:
        while not _shutdown:
            batches = consumer.poll(timeout_ms=POLL_TIMEOUT_MS)
            if not batches:
                continue

            try:
                for records in batches.values():
                    for message in records:
                        _handle_message(message.value)
            except Exception as exc:
                logger.exception("Failed to persist batch, not committing: %s", exc)
                continue

            consumer.commit()
    finally:
        consumer.close()
        store.close_pool()
        logger.info("Results consumer shut down cleanly")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    run_consumer()
