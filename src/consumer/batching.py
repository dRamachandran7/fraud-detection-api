"""Offset handling shared by both consumers.

Both process a polled batch as a unit: commit after the whole batch succeeds,
rewind if any of it fails. Getting either wrong loses messages silently.
"""
from __future__ import annotations

import logging

from kafka import KafkaConsumer
from kafka.consumer.fetcher import ConsumerRecord
from kafka.errors import KafkaError
from kafka.structs import TopicPartition

logger = logging.getLogger(__name__)


def rewind(consumer: KafkaConsumer, batches: dict[TopicPartition, list[ConsumerRecord]]) -> None:
    """Seek back to the start of a failed batch so it is redelivered.

    Skipping the commit is not enough: `poll()` has already advanced the
    in-memory position, so the next poll would fetch the messages *after* the
    failed batch, and the next successful commit would move the committed
    offset past it — losing the failed messages without a trace.

    Partitions revoked by a rebalance are skipped; their new owner resumes from
    the last committed offset, which is still before this batch.
    """
    assigned = consumer.assignment()
    for tp, records in batches.items():
        if records and tp in assigned:
            consumer.seek(tp, records[0].offset)


def commit_batch(consumer: KafkaConsumer) -> bool:
    """Commit consumed offsets. Returns False if the commit did not land.

    A failed commit after a successful batch is not data loss: the batch is
    already published, and at worst it is redelivered and absorbed by the
    idempotent downstream. The usual cause is a rebalance taking the partitions
    away mid-batch, which must not crash the process.
    """
    try:
        consumer.commit()
    except KafkaError:
        logger.warning("Offset commit failed; batch may be redelivered", exc_info=True)
        return False
    return True
