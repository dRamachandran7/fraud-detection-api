"""Consumer: reads transactions from Kafka, scores them, publishes results."""
from __future__ import annotations

import json
import logging
import signal
import time
from types import FrameType

from kafka import KafkaConsumer, KafkaProducer
from kafka.consumer.fetcher import ConsumerRecord
from kafka.errors import KafkaError
from prometheus_client import Counter, Histogram

from src import config
from src.consumer.batching import commit_batch, rewind
from src.inference import load_model, predict, predict_batch
from src.kafka_serde import JsonSerializer, StringSerializer
from src.metrics import LagReporter, start_metrics_server

logger = logging.getLogger(__name__)

GROUP_ID = "fraud-scorer"
POLL_TIMEOUT_MS = 1000
# Pause after a failed batch so a broker outage is a slow retry loop, not a
# hot one.
FAILURE_BACKOFF_SECONDS = 1.0

SCORED = Counter(
    "fraud_transactions_scored_total", "Transactions scored", ["is_fraud"]
)
DEAD_LETTERED = Counter(
    "fraud_transactions_dead_lettered_total",
    "Messages sent to the DLQ",
    ["reason"],  # fixed categories only; free-text reasons go in the payload
)
BATCH_FAILURES = Counter(
    "fraud_score_batch_failures_total", "Batches left uncommitted for redelivery"
)
BATCH_SIZE = Histogram(
    "fraud_score_batch_size",
    "Messages per polled batch",
    buckets=(1, 5, 10, 25, 50, 100, 250, 500),
)
SCORING_SECONDS = Histogram(
    "fraud_scoring_seconds", "Model time per batch, excluding Kafka I/O"
)
BATCH_SECONDS = Histogram(
    "fraud_score_batch_seconds", "Wall time per batch, from poll to commit"
)

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
        group_id=GROUP_ID,
        # Offsets advance only after a score is durably published, making this
        # at-least-once. With the default timed auto-commit, a crash between
        # commit and publish would silently drop transactions.
        enable_auto_commit=False,
        session_timeout_ms=config.KAFKA_SESSION_TIMEOUT_MS,
        heartbeat_interval_ms=config.KAFKA_HEARTBEAT_INTERVAL_MS,
    )


def _build_producer() -> KafkaProducer:
    return KafkaProducer(
        bootstrap_servers=config.KAFKA_BOOTSTRAP_SERVERS,
        value_serializer=JsonSerializer(),
        key_serializer=StringSerializer(),
        acks="all",
        retries=5,
        max_in_flight_requests_per_connection=1,
    )


def _dead_letter(producer: KafkaProducer, raw: bytes, category: str, reason: str) -> None:
    """Park a message we can never score, so the loop is not a crash loop."""
    payload = {
        "reason": reason,
        "raw": raw.decode("utf-8", errors="replace") if raw is not None else None,
    }
    producer.send(config.KAFKA_DLQ_TOPIC, value=payload).get(
        timeout=config.PRODUCER_ACK_TIMEOUT_SECONDS
    )
    DEAD_LETTERED.labels(category).inc()
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


def _score(
    artifact: dict, producer: KafkaProducer, parsed: list[tuple[bytes, str | None, list[float]]]
) -> list[tuple[str | None, dict]]:
    """Score parsed messages in one model call.

    If the batch call fails, one bad row is the likely cause, so fall back to
    row-by-row scoring to dead-letter just that row instead of the batch.
    """
    started = time.perf_counter()
    try:
        results = predict_batch(artifact, [features for _, _, features in parsed])
        scored = [(tid, result) for (_, tid, _), result in zip(parsed, results)]
    except Exception:
        logger.warning("Batch scoring failed, retrying row by row", exc_info=True)
        scored = []
        for raw, tid, features in parsed:
            try:
                scored.append((tid, predict(artifact, features)))
            except Exception as exc:  # model rejected input we could not anticipate
                _dead_letter(producer, raw, "scoring_failed", f"scoring failed: {exc}")
    SCORING_SECONDS.observe(time.perf_counter() - started)
    return scored


def _process_batch(
    artifact: dict, producer: KafkaProducer, messages: list[ConsumerRecord]
) -> None:
    """Score a polled batch and publish every result.

    Message-level problems are dead-lettered. Broker-level problems propagate
    so the caller can rewind and the batch is redelivered.
    """
    parsed = []
    for message in messages:
        try:
            transaction_id, features = _parse(message.value)
        except ValueError as exc:
            _dead_letter(producer, message.value, "invalid", str(exc))
            continue
        parsed.append((message.value, transaction_id, features))

    if not parsed:
        return

    scored = _score(artifact, producer, parsed)

    # Send everything, then wait on the acks together: one round trip per
    # batch rather than one per message.
    futures = []
    for transaction_id, result in scored:
        result["transaction_id"] = transaction_id
        futures.append(
            producer.send(config.KAFKA_SCORES_TOPIC, key=transaction_id, value=result)
        )
    for future in futures:
        future.get(timeout=config.PRODUCER_ACK_TIMEOUT_SECONDS)

    for _, result in scored:
        SCORED.labels(str(result["is_fraud"]).lower()).inc()
    logger.info(
        "Scored %d transactions (%d flagged)",
        len(scored),
        sum(result["is_fraud"] for _, result in scored),
    )


def run_consumer() -> None:
    """Consume raw transactions, score with the model, publish results."""
    _install_signal_handlers()
    start_metrics_server()

    artifact = load_model()
    logger.info(
        "Loaded model version %s (trained_at=%s, average_precision=%s)",
        artifact["version"],
        artifact.get("trained_at"),
        artifact.get("metrics", {}).get("average_precision"),
    )

    consumer = _build_consumer()
    producer = _build_producer()
    lag = LagReporter(consumer, GROUP_ID)

    try:
        while not _shutdown:
            lag.maybe_report()
            batches = consumer.poll(timeout_ms=POLL_TIMEOUT_MS)
            if not batches:
                continue

            started = time.perf_counter()
            messages = [m for records in batches.values() for m in records]
            BATCH_SIZE.observe(len(messages))
            try:
                _process_batch(artifact, producer, messages)
            except KafkaError:
                BATCH_FAILURES.inc()
                logger.exception("Broker error while processing batch; rewinding")
                rewind(consumer, batches)
                time.sleep(FAILURE_BACKOFF_SECONDS)
                continue

            commit_batch(consumer)
            BATCH_SECONDS.observe(time.perf_counter() - started)
    finally:
        producer.flush(timeout=config.PRODUCER_ACK_TIMEOUT_SECONDS)
        producer.close(timeout=config.PRODUCER_ACK_TIMEOUT_SECONDS)
        consumer.close()
        logger.info("Consumer shut down cleanly")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    run_consumer()
