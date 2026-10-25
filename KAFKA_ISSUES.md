# Kafka Issues

Tracking file from the 2026-09-13 review of the producer and consumers. Every
item from that review is resolved; the few open items at the bottom were found
while closing them out.

---

## Resolved

**Nothing consumed `fraud-scores`** (2026-10-18). A results consumer persists
scores to Postgres; `GET /transactions/{id}` and `GET /transactions` serve
them.

**Scoring was one row at a time** (2026-10-25). Both consumers now process a
polled batch as a unit: one model call per batch in the scoring consumer, one
transaction per batch in the results consumer, and all score sends issued
before waiting on their acks. Measured locally: ~8x cheaper per transaction at
the ~10-message batches seen under steady load, ~290x at a full 500-message
batch. Batches grow on their own when a backlog builds, which is when it
matters. If the batch model call fails, the consumer retries row by row so one
bad row is dead-lettered instead of the batch.

**No observability** (2026-10-25). Prometheus metrics on the API (`/metrics`)
and both consumers (`:9100/metrics`), scraped by a `prometheus` service on
`localhost:9090`. Consumer lag per partition is the headline metric:
`sum by (group) (fraud_consumer_lag_messages)`. Also: scored/flagged counts,
DLQ count by reason, batch size, model time, DB write time, batch failures,
ingest publish latency, HTTP requests by route and status.

**Failed batches were silently lost** (2026-10-25, found during this work). On
a failed batch both consumers skipped the commit, but `poll()` had already
moved the read position past it, so the next successful commit skipped the
failed messages for good. Both now seek back to the start of a failed batch.
A commit that fails after a rebalance no longer crashes the process.

**Untested under replicas and load** (2026-10-25). `scripts/load_test.py`
sends real transactions through `/ingest` and checks Postgres that every
acknowledged one was persisted. Results with 3 scoring replicas, 5,000
transactions at concurrency 32:

| Scenario | Acked | Persisted | End to end |
|---|---|---|---|
| Steady load | 5000 | 5000 | 15.1s |
| One replica SIGKILLed mid-run | 5000 | 5000 | 15.5s |
| Postgres stopped for 10s mid-run | 5000 | 5000 | 21.5s |

The replica-kill run first took 50.5s: kafka-python's 45s default session
timeout left the dead replica's partitions idle for ~35s. Lowered to 10s
(`KAFKA_SESSION_TIMEOUT_MS`).

**Serializer deprecation warning** (2026-10-25). Replaced lambda serializers
with `kafka.serializer.Serializer` subclasses; `kafka-python` pinned.

---

## Open

- **Ingest throughput tops out around 350–400 req/s** on the single API
  process. Scoring keeps up with that easily, so the API is the bottleneck.
  The likely limit is each request blocking on its broker ack with
  `max_in_flight_requests_per_connection=1`. Not a problem at current volume;
  more uvicorn workers or API replicas are the first step if it becomes one.
- **Nothing alerts.** Prometheus collects the metrics but no rule fires on
  rising lag, DLQ growth, or batch failures.
- **The restart policy is unverified.** Consumers have
  `restart: unless-stopped`, but the kill test used `docker kill`, which
  Docker treats as a manual stop, so automatic restart was not exercised.
