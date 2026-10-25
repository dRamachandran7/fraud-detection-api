# Fraud Detection API

A fraud detection API backed by Apache Kafka for high-throughput transaction ingestion and an Isolation Forest model trained on a Kaggle credit card dataset.

## Architecture

Client → FastAPI `/ingest` → Kafka `transactions` → scoring consumer → Kafka `fraud-scores` → results consumer → Postgres ← `GET /transactions/{id}`

The API never blocks on inference. It accepts and enqueues. Kafka absorbs traffic spikes. Consumers scale independently.

## Setup

```bash
pip install -r requirements-dev.txt
python -m src.model.fetch_data     # public dataset; no Kaggle key needed
python -m src.model.train
docker compose up -d --build       # kafka, topics, postgres, api, both consumers
```

To run the API and consumer on the host instead of in containers, keep
`KAFKA_BOOTSTRAP_SERVERS=localhost:9092` in `.env` and start them directly:

```bash
uvicorn src.api.main:app --reload
python -m src.consumer.score_consumer  # in a separate terminal
```

Compose overrides that variable to `kafka:29092`, and `DATABASE_URL` to the
`postgres` service, since `localhost` inside a container is the container itself.

## API

| Method | Path | |
|---|---|---|
| `POST` | `/ingest` | Enqueue a transaction; returns `transaction_id` once the broker acks |
| `GET` | `/transactions/{id}` | The score, or 404 while still in flight |
| `GET` | `/transactions?since=&limit=` | Flagged transactions, newest first |

A 404 from `GET /transactions/{id}` means "still in flight *or* unknown" —
the two are deliberately indistinguishable. Writing a pending row at ingest
would put Postgres on the ingest hot path, so a database outage would stop
ingestion. Clients poll until the score appears; typical latency is well
under a second.

Scores are persisted by a separate results consumer, so a database stall
backs up `fraud-scores` rather than stopping scoring.

## Operations

**Scaling.** Scoring consumers scale horizontally up to the partition count
(6): `docker compose up -d --scale consumer=3`. Each replica owns a share of
the partitions; when one dies, the rest take over its partitions within
~10 seconds (`KAFKA_SESSION_TIMEOUT_MS`).

**Metrics.** Prometheus runs at `localhost:9090` and scrapes the API and every
consumer replica. The number to watch is consumer lag:

```
sum by (group) (fraud_consumer_lag_messages)
```

Lag that keeps rising means ingestion is outpacing scoring: add replicas.

**Load test.** `python -m scripts.load_test --total 5000 --concurrency 32`
drives `/ingest` with real transactions and checks Postgres that every
acknowledged one was persisted. Current results are in `KAFKA_ISSUES.md`.

## Model

A hybrid: a gradient-boosted classifier (`HistGradientBoostingClassifier`)
trained on the fraud labels, with an Isolation Forest anomaly score added as
one more input feature. `score` in the API is the classifier's probability of
fraud, 0 to 1.

```bash
python -m src.model.train --alert-rate 0.005
python -m src.model.train --no-anomaly-score   # supervised only, for comparison
```

`--alert-rate` is an **alert budget**, not an estimate of the fraud rate: it is
the share of traffic to flag, and should come from how many reviews the team
can absorb. The threshold is set at that percentile of held-out scores, and
`is_fraud` compares against it — not against 0.5.

Training prints **average precision** alongside precision and recall at the
chosen threshold. Watch average precision, not ROC-AUC: at a ~0.17% positive
rate, ROC-AUC looks flattering and says little about review workload.

Splits are stratified on label x 4-hour time bucket, not label alone: the
fraud rate swings ~8x across the dataset's 48 hours (overnight hours carry the
most), so a label-only split can over- or under-sample the night. Training
also runs 5-fold cross-validation, because a single holdout's average
precision moves by several points with the split alone, and a time-forward
check (fit on the first 24h, test on the last 24h).

Current numbers:

| | CV avg precision | CV recall @ 0.5% | Time-forward avg precision | Time-forward recall @ 0.5% |
|---|---|---|---|---|
| Hybrid (shipped) | 0.694 +/- 0.042 | 0.799 | 0.349 | 0.630 |
| Supervised only | 0.691 +/- 0.028 | 0.797 | 0.303 | 0.673 |
| Isolation Forest only (previous model) | 0.178 +/- 0.036 | 0.451 | 0.114 | 0.313 |

On this dataset the anomaly score adds nothing measurable over supervised
only. It is kept as a hedge for fraud patterns the labels do not contain,
which labelled data from the same two days cannot test.

Precision at a fixed alert rate is capped by the base rate: fraud is ~0.17%
of traffic, so flagging 0.5% can never exceed ~35% precision. The
time-forward numbers are the closest to production conditions; expect
real-world results nearer them than the CV means.

Why the Isolation Forest alone underperforms: fraud *is* unusual (84% of
frauds rank in the most anomalous 5% of transactions), but most unusual
transactions are not fraud. At the 0.5% budget, 370 of its 428 alerts were
legitimate, typically large purchases (median $118 vs $10 for fraud). The
fraud signal is concentrated in a few features (V17, V14, V12, V10), while the
forest picks split features at random and weighs every kind of unusual
equally. The labels tell the classifier which kind matters.

`scikit-learn` is pinned in `requirements.txt`: the artifact is a pickle and
must be loaded with the version that trained it.

### Feature contract

`src/model/preprocess.py` owns the column set and ordering, and both training
and inference go through it. `FEATURE_ORDER` is V1–V28 then `Amount` — the
positional contract for the `features` list sent to `/ingest`. `Time` is
excluded deliberately: it is an artifact of how the dataset was assembled.
`Amount` is log1p'd *inside* the serialized pipeline, so the transform cannot
drift between training and serving.

Artifacts are dicts, not bare estimators: they carry the pipeline, threshold,
feature order, a git-sha version, and the training metrics. The version is
stamped onto every score message, and the consumer logs it at startup. A
pre-artifact `model.joblib` is rejected with a retrain instruction rather than
silently scoring garbage.

## Delivery guarantees

`/ingest` blocks until the broker acknowledges the write with `acks=all`, so a
returned `transaction_id` means the transaction is durably queued; a broker
failure returns 503 rather than a false acknowledgement. The consumer commits
offsets only after a score is published, making the pipeline at-least-once —
downstream consumers should dedupe on `transaction_id`. Messages that can never
be scored (malformed JSON, wrong feature count) go to `transactions-dlq`.

## Tests

```bash
pytest
```
