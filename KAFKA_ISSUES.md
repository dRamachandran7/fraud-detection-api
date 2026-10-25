# Open Kafka Issues

Deferred items from the 2026-09-13 review of the producer and consumer. The
blockers, producer durability, and consumer at-least-once work are already
done; what follows is what was consciously left for later.

---

## ~~1. Nothing consumes `fraud-scores`~~ — done 2026-10-18

Results consumer persists scores to Postgres (upsert on `transaction_id`);
`GET /transactions/{id}` and `GET /transactions` serve them. Verified on the
live stack, including no loss and no duplicates across a SIGKILL of the
scoring consumer mid-stream.

---

## 2. Scoring is one row at a time

**Priority: medium.** Cheap throughput win, low risk.

`_handle_message` reshapes a single transaction into `(1, 29)` and calls
`predict` per message. The consumer already uses `poll()`, so records arrive
in batches — stacking a batch into one array and calling
`model.predict` / `model.decision_function` once would cut per-row sklearn
overhead substantially.

Care required on two points: the DLQ path is per-message, so unparseable
records have to be filtered out of the batch before stacking and dead-lettered
individually; and results must be zipped back to their `transaction_id` in
order. Keep the commit boundary where it is — after the whole batch is
published.

---

## 3. No observability

**Priority: medium.**

No metrics of any kind: consumer lag, scoring latency, DLQ rate, and publish
failures are all invisible. Consumer lag is the one that matters most — it is
the only signal that the consumer group has fallen behind ingestion, and the
scaling story depends on being able to see it.

*Model traceability, previously part of this item, is now done:* artifacts
carry a git-sha version, `model_version` is stamped on every score message,
and the consumer logs the version and training metrics at startup.

---

## Verification status

The full pipeline has been run against real Kafka and Postgres via
`docker compose`: end-to-end scoring, poison messages dead-lettered without
crashing the consumer, at-least-once delivery across a SIGKILL (302/302
transactions persisted, 8 redeliveries absorbed by the upsert), and clean
SIGTERM shutdown of both consumers. That run caught three bugs the mocks
could not: a YAML folding error that broke topic creation, and two SQL bugs
in the store. `tests/test_store.py` now exercises the store against real
Postgres when one is reachable.

Not yet tested: multiple scoring-consumer replicas rebalancing partitions,
and behaviour under sustained load.
