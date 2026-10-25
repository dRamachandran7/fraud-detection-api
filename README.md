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

## Model

Isolation Forest, trained unsupervised. The labels are never passed to `fit`,
but they are used to evaluate and to choose the decision threshold — an
unvalidated anomaly detector is not worth deploying.

```bash
python -m src.model.train --alert-rate 0.005
```

`--alert-rate` is an **alert budget**, not an estimate of the fraud rate: it is
the share of traffic to flag, and should come from how many reviews the team
can absorb. The threshold is set at that percentile of held-out scores, and
`is_fraud` compares against it. Do not rely on sklearn's `.predict()` here —
its boundary comes from `contamination` and is not tuned to that budget.

Training prints **average precision** alongside precision and recall at the
chosen threshold. Watch average precision, not ROC-AUC: at a ~0.17% positive
rate, ROC-AUC looks flattering and says little about review workload.

The model is fitted on all training rows. Fitting on labelled-clean rows
only (`--fit-on-normals`) scored worse on the Kaggle data in every one of five
seeds — mean average precision 0.140 vs 0.188 — so it is kept only for
comparison.

Current numbers on the 30% held-out split (148 frauds in 85,443 rows):

| Alert rate | Alerts | Precision | Recall |
|---|---|---|---|
| 0.1% | 86 | 0.35 | 0.20 |
| 0.5% (default) | 428 | 0.17 | 0.49 |
| 1.0% | 855 | 0.11 | 0.64 |
| 2.0% | 1,709 | 0.07 | 0.75 |

Average precision is 0.186. At the default budget, about one alert in six is
real fraud and half of all fraud is caught.

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
