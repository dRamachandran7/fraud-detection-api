"""Pydantic request/response models."""
from __future__ import annotations

from pydantic import BaseModel


class IngestRequest(BaseModel):
    """A single transaction to enqueue for scoring."""
    features: list[float]


class IngestResponse(BaseModel):
    """Acknowledgement that the transaction was enqueued."""
    transaction_id: str
    status: str
