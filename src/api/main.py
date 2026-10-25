"""FastAPI application — ingests transactions and serves their scores."""
from __future__ import annotations

from contextlib import asynccontextmanager
from datetime import datetime

from fastapi import FastAPI, HTTPException, Query

from src import config
from src.api.kafka_producer import PublishError, publish_transaction, shutdown_producer
from src.api.schemas import IngestRequest, IngestResponse, ScoreResponse
from src.store import scores as store


@asynccontextmanager
async def lifespan(app: FastAPI):
    yield
    # Flush anything still buffered so a redeploy does not drop transactions.
    shutdown_producer()
    store.close_pool()


app = FastAPI(title="Fraud Detection API", lifespan=lifespan)


# Deliberately sync: publish_transaction blocks on the broker ack, and FastAPI
# runs sync endpoints in a threadpool. As `async def` it would stall the event
# loop for every other request whenever Kafka is slow or unreachable.
@app.post("/ingest", response_model=IngestResponse)
def ingest(request: IngestRequest) -> IngestResponse:
    try:
        transaction_id = publish_transaction(request)
    except PublishError as exc:
        raise HTTPException(
            status_code=503, detail="Could not enqueue transaction"
        ) from exc
    return IngestResponse(transaction_id=transaction_id, status="queued")


@app.get("/transactions/{transaction_id}", response_model=ScoreResponse)
def get_transaction(transaction_id: str) -> ScoreResponse:
    """Fetch a transaction's score.

    404 covers both "still in flight" and "never seen". The two are not
    distinguishable here by design: writing a pending row at ingest would put
    Postgres on the ingest hot path, so a database outage would stop
    ingestion — the one thing the Kafka buffer exists to prevent.
    """
    record = store.get_score(transaction_id)
    if record is None:
        raise HTTPException(
            status_code=404,
            detail="No score for this transaction; it may still be in flight.",
        )
    return ScoreResponse(**record)


@app.get("/transactions", response_model=list[ScoreResponse])
def list_flagged_transactions(
    limit: int = Query(default=100, ge=1, le=config.MAX_LIST_LIMIT),
    since: datetime | None = Query(
        default=None, description="Only scores at or after this timestamp"
    ),
) -> list[ScoreResponse]:
    """List flagged transactions, newest first."""
    return [ScoreResponse(**row) for row in store.list_flagged(limit=limit, since=since)]
