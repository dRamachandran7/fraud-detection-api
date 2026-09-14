"""Pydantic request/response models."""
from __future__ import annotations

from pydantic import BaseModel, Field

from src import config


class IngestRequest(BaseModel):
    """A single transaction to enqueue for scoring."""

    # Length is enforced here so a malformed client gets a 422 immediately,
    # rather than the failure surfacing later and invisibly in the consumer.
    features: list[float] = Field(
        min_length=config.EXPECTED_FEATURE_COUNT,
        max_length=config.EXPECTED_FEATURE_COUNT,
    )


class IngestResponse(BaseModel):
    """Acknowledgement that the transaction was enqueued."""

    transaction_id: str
    status: str
