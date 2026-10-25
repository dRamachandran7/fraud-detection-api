"""Consumer: reads scores from Kafka and persists them to the score store.

Separate from the scoring consumer on purpose: persistence failures must not
stall scoring, and the two scale on different pressures (one is CPU-bound on
the model, this one is bound on database writes).
"""
from __future__ import annotations

import json
import logging
import signal
import time
from types import FrameType

from kafka import KafkaConsumer
from kafka.consumer.fetcher import ConsumerRecord
from prometheus_client import Counter, Histogram

from src import config
from src.consumer.batching import commit_batch, rewind
from src.metrics import LagReporter, start_metrics_server
from src.store import scores as store

logger = logging.getLogger(__name__)

GROUP_ID = "score-persister"
POLL_TIMEOUT_MS = 1000
FAILURE_BACKOFF_SECONDS = 1.0
REQUIRED_FIELDS = ("transaction_id", "is_fraud", "score", "model_version")

PERSISTED = Counter("fraud_scores_persisted_total", "Scores written to the store")
SKIPPED = Counter(
    "fraud_scores_skipped_total", "Unusable score messages (a bug upstream if nonzero)"
)
BATCH_FAILURES = Counter(
    "fraud_persist_batch_failures_total", "Batches left uncommitted for redelivery"
)
DB_WRITE_SECONDS = Histogram("fraud_db_write_seconds", "Time per batched upsert")

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
        group_id=GROUP_ID,
        # Commit only after the write lands, so a crash replays rather than
        # silently dropping a score. Upserts make the replay harmless.
        enable_auto_commit=False,
        session_timeout_ms=config.KAFKA_SESSION_TIMEOUT_MS,
        heartbeat_interval_ms=config.KAFKA_HEARTBEAT_INTERVAL_MS,
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


def _process_batch(messages: list[ConsumerRecord]) -> None:
    """Persist a polled batch in one transaction. Malformed messages are skipped.

    Database errors are left to propagate: unlike a bad message, they are
    transient, and the batch should be rewound so every score is retried.
    """
    rows = []
    for message in messages:
        try:
            event = _parse(message.value)
        except ValueError as exc:
            # No DLQ here: these messages came from our own scoring consumer, so
            # a malformed one is a bug on our side worth alerting on, not client
            # input. SKIPPED makes it visible.
            SKIPPED.inc()
            logger.error("Unusable score message, skipping: %s", exc)
            continue
        rows.append(
            {
                "transaction_id": event["transaction_id"],
                "is_fraud": bool(event["is_fraud"]),
                "score": float(event["score"]),
                "model_version": event["model_version"],
            }
        )

    if not rows:
        return

    started = time.perf_counter()
    store.save_scores(rows)
    DB_WRITE_SECONDS.observe(time.perf_counter() - started)
    PERSISTED.inc(len(rows))
    logger.info("Persisted %d scores", len(rows))


def run_consumer() -> None:
    """Consume scored transactions and write them to the store."""
    _install_signal_handlers()
    start_metrics_server()

    store.init_schema()
    consumer = _build_consumer()
    lag = LagReporter(consumer, GROUP_ID)

    try:
        while not _shutdown:
            lag.maybe_report()
            batches = consumer.poll(timeout_ms=POLL_TIMEOUT_MS)
            if not batches:
                continue

            try:
                _process_batch([m for records in batches.values() for m in records])
            except Exception:
                BATCH_FAILURES.inc()
                logger.exception("Failed to persist batch; rewinding")
                rewind(consumer, batches)
                time.sleep(FAILURE_BACKOFF_SECONDS)
                continue

            commit_batch(consumer)
    finally:
        consumer.close()
        store.close_pool()
        logger.info("Results consumer shut down cleanly")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    run_consumer()
