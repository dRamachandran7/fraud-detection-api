# Open Kafka Issues

Deferred items from the 2026-09-13 review of the producer and consumer. The
blockers, producer durability, and consumer at-least-once work are already
done; what follows is what was consciously left for later.

---

## 1. Nothing consumes `fraud-scores` — there is no read path

**Priority: high.** This is the largest remaining gap.

The pipeline terminates in a topic with no consumer. A client posts to
`/ingest`, receives a `transaction_id`, and has no way — ever — to learn
whether the transaction was flagged. There is no status endpoint, no result
store, and no push to a downstream service. Half the system is missing, and
neither `PLANNING.md` nor `README.md` acknowledges it.

This is a design decision before it is a code change. Two shapes:

- **Pull.** A results consumer reads `fraud-scores` and writes to Redis or
  Postgres keyed by `transaction_id`; the API grows a
  `GET /transactions/{id}` returning `pending` until a score lands. Simplest,
  and it gives the caller a story that matches the ID they already hold.
- **Push.** Scores are forwarded to a downstream service or webhook, and the
  API never serves results at all. Fewer moving parts, but callers need an
  out-of-band way to correlate.

Whichever is chosen, the store must dedupe on `transaction_id` — the consumer
is at-least-once, so duplicate scores are expected, not exceptional.

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

## 3. No observability, and scores are not traceable to a model

**Priority: medium.**

- No metrics of any kind: consumer lag, scoring latency, DLQ rate, and publish
  failures are all invisible. Consumer lag is the one that matters most — it
  is the only signal that the consumer group has fallen behind ingestion, and
  the scaling story depends on being able to see it.
- Output messages carry only `transaction_id`, `is_fraud`, and `score`. There
  is no model version or scoring timestamp, so a score cannot be traced back
  to the artifact that produced it. Once the model is retrained even once,
  historical scores become uninterpretable. Stamp a version — a hash or a
  build tag baked in at train time — onto both the artifact and every result
  message.

---

## Also unresolved (not code)

`data/creditcard.csv` and `src/model/artifacts/model.joblib` are both absent,
so `load_model()` raises at consumer startup and the stack cannot run
end-to-end. None of the Kafka work above has been exercised against a real
broker for that reason — it is covered by mock-based unit tests only. Worth a
`docker-compose up` against the real dataset before trusting any of it.
