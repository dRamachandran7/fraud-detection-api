"""Postgres-backed store for fraud scores."""
from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path
from typing import Any

from psycopg_pool import ConnectionPool

from src import config

logger = logging.getLogger(__name__)

SCHEMA_PATH = Path(__file__).with_name("schema.sql")

_pool: ConnectionPool | None = None


def get_pool() -> ConnectionPool:
    """Lazily open the shared connection pool."""
    global _pool
    if _pool is None:
        _pool = ConnectionPool(
            config.DATABASE_URL,
            min_size=1,
            max_size=config.DB_POOL_MAX_SIZE,
            open=True,
            timeout=config.DB_CONNECT_TIMEOUT_SECONDS,
        )
    return _pool


def close_pool() -> None:
    """Close the pool. Safe to call when never opened."""
    global _pool
    if _pool is not None:
        _pool.close()
        _pool = None


def init_schema() -> None:
    """Apply schema.sql. Idempotent, so every process may call it at startup."""
    with get_pool().connection() as conn:
        conn.execute(SCHEMA_PATH.read_text())
    logger.info("Score store schema ready")


UPSERT_SQL = """
    INSERT INTO fraud_scores
        (transaction_id, is_fraud, score, model_version, scored_at)
    VALUES (%s, %s, %s, %s, COALESCE(%s, now()))
    ON CONFLICT (transaction_id) DO UPDATE SET
        is_fraud      = EXCLUDED.is_fraud,
        score         = EXCLUDED.score,
        model_version = EXCLUDED.model_version,
        scored_at     = EXCLUDED.scored_at
"""


def save_scores(scores: list[dict[str, Any]]) -> None:
    """Upsert many scores in one transaction: all land, or none do.

    Idempotent on transaction_id: redelivery from the at-least-once consumers
    overwrites rather than duplicating. A rescore under a newer model version
    legitimately replaces the old verdict.
    """
    if not scores:
        return
    params = [
        (
            s["transaction_id"],
            s["is_fraud"],
            s["score"],
            s["model_version"],
            s.get("scored_at"),
        )
        for s in scores
    ]
    with get_pool().connection() as conn:
        with conn.cursor() as cur:
            cur.executemany(UPSERT_SQL, params)


def save_score(
    transaction_id: str,
    is_fraud: bool,
    score: float,
    model_version: str,
    scored_at: datetime | None = None,
) -> None:
    """Upsert one score. See save_scores."""
    save_scores(
        [
            {
                "transaction_id": transaction_id,
                "is_fraud": is_fraud,
                "score": score,
                "model_version": model_version,
                "scored_at": scored_at,
            }
        ]
    )


def get_score(transaction_id: str) -> dict[str, Any] | None:
    """Return one score, or None if this transaction has not been scored."""
    with get_pool().connection() as conn:
        row = conn.execute(
            """
            SELECT transaction_id, is_fraud, score, model_version, scored_at
            FROM fraud_scores
            WHERE transaction_id = %s
            """,
            (transaction_id,),
        ).fetchone()

    if row is None:
        return None
    return _as_dict(row)


def list_flagged(limit: int = 100, since: datetime | None = None) -> list[dict[str, Any]]:
    """Return flagged transactions, newest first."""
    # The ::timestamptz casts are required: with server-side binding, Postgres
    # cannot infer the type of a NULL parameter compared with IS NULL. (Kept
    # out of the SQL string because psycopg counts placeholders in comments.)
    with get_pool().connection() as conn:
        rows = conn.execute(
            """
            SELECT transaction_id, is_fraud, score, model_version, scored_at
            FROM fraud_scores
            WHERE is_fraud
              AND (%s::timestamptz IS NULL OR scored_at >= %s::timestamptz)
            ORDER BY scored_at DESC
            LIMIT %s
            """,
            (since, since, limit),
        ).fetchall()
    return [_as_dict(row) for row in rows]


def _as_dict(row: tuple) -> dict[str, Any]:
    transaction_id, is_fraud, score, model_version, scored_at = row
    return {
        "transaction_id": transaction_id,
        "is_fraud": is_fraud,
        "score": score,
        "model_version": model_version,
        "scored_at": scored_at,
    }
