"""FastAPI application — accepts transactions and enqueues to Kafka."""
from __future__ import annotations

from fastapi import FastAPI

from src.api.kafka_producer import publish_transaction
from src.api.schemas import IngestRequest, IngestResponse

app = FastAPI(title="Fraud Detection API")


@app.post("/ingest", response_model=IngestResponse)
async def ingest(request: IngestRequest) -> IngestResponse:
    transaction_id = publish_transaction(request)
    return IngestResponse(transaction_id=transaction_id, status="queued")
