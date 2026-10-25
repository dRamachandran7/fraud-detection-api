"""Prometheus helpers shared by the consumer processes.

Each process defines its own counters next to the code that increments them;
this module holds only what every consumer needs: the metrics endpoint and
consumer-lag reporting.
"""
from __future__ import annotations

import logging
import time

from kafka import KafkaConsumer
from prometheus_client import Gauge, start_http_server

from src import config

logger = logging.getLogger(__name__)

# Lag is the one signal that a consumer group has fallen behind ingestion, and
# the only way to know when to add replicas. Each replica reports lag for the
# partitions it currently owns; sum across replicas for the group total.
CONSUMER_LAG = Gauge(
    "fraud_consumer_lag_messages",
    "Messages in the topic not yet consumed, per owned partition",
    ["group", "topic", "partition"],
)

LAG_REPORT_INTERVAL_SECONDS = 10


def start_metrics_server() -> None:
    """Expose /metrics for Prometheus. METRICS_PORT=0 disables it."""
    if config.METRICS_PORT:
        start_http_server(config.METRICS_PORT)
        logger.info("Metrics on :%s/metrics", config.METRICS_PORT)


class LagReporter:
    """Refreshes the lag gauge at most every LAG_REPORT_INTERVAL_SECONDS.

    `end_offsets` is a broker round trip, so it is throttled rather than run
    on every poll.
    """

    def __init__(self, consumer: KafkaConsumer, group: str):
        self._consumer = consumer
        self._group = group
        self._last_report = 0.0

    def maybe_report(self) -> None:
        now = time.monotonic()
        if now - self._last_report < LAG_REPORT_INTERVAL_SECONDS:
            return
        self._last_report = now

        assigned = list(self._consumer.assignment())
        try:
            end_offsets = self._consumer.end_offsets(assigned) if assigned else {}
            lags = {
                tp: max(end_offsets[tp] - self._consumer.position(tp), 0) for tp in assigned
            }
        except Exception:  # never let metrics take down the consumer
            logger.warning("Could not refresh consumer lag", exc_info=True)
            return

        # Clear first: after a rebalance, partitions this replica no longer owns
        # would otherwise keep reporting their last value forever.
        CONSUMER_LAG.clear()
        for tp, lag in lags.items():
            CONSUMER_LAG.labels(self._group, tp.topic, str(tp.partition)).set(lag)
