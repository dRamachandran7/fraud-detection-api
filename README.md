# Fraud Detection API

A fraud detection API backed by Apache Kafka for high-throughput transaction ingestion and an Isolation Forest model trained on a Kaggle credit card dataset.

## Architecture

Client → FastAPI `/ingest` → Kafka `transactions` topic → Consumer scores with model → Kafka `fraud-scores` topic

The API never blocks on inference. It accepts and enqueues. Kafka absorbs traffic spikes. Consumers scale independently.

## Setup

```bash
pip install -r requirements-dev.txt
python -m src.model.train          # requires data/creditcard.csv
docker-compose up -d               # kafka + topic creation + api + consumer
```

To run the API and consumer on the host instead of in containers, keep
`KAFKA_BOOTSTRAP_SERVERS=localhost:9092` in `.env` and start them directly:

```bash
uvicorn src.api.main:app --reload
python -m src.consumer.score_consumer  # in a separate terminal
```

Compose overrides that variable to `kafka:29092`, since `localhost` inside a
container is the container itself.

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

By default the model is fitted on labelled-clean rows only, making it a
novelty detector and keeping every fraud available for scoring. `--fit-on-all`
fits on everything instead; compare average precision between the two.

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
