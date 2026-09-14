"""FastAPI application — accepts transactions and enqueues to Kafka."""
from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException

from src.api.kafka_producer import PublishError, publish_transaction, shutdown_producer
from src.api.schemas import IngestRequest, IngestResponse


@asynccontextmanager
async def lifespan(app: FastAPI):
    yield
    # Flush anything still buffered so a redeploy does not drop transactions.
    shutdown_producer()


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
