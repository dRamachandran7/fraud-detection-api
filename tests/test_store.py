"""Integration tests for the score store, against a real Postgres.

Mocks cannot catch SQL bugs — the store shipped with two (an untyped NULL
parameter, and a placeholder counted inside a SQL comment) that only a real
server revealed. Skipped when no database is reachable; run
`docker compose up -d postgres` to enable.
"""
from __future__ import annotations

import uuid
from datetime import datetime, timedelta, timezone

import pytest

psycopg = pytest.importorskip("psycopg")

from src import config  # noqa: E402
from src.store import scores as store  # noqa: E402


def _db_reachable() -> bool:
    try:
        with psycopg.connect(config.DATABASE_URL, connect_timeout=2):
            return True
    except psycopg.OperationalError:
        return False


pytestmark = pytest.mark.skipif(not _db_reachable(), reason="Postgres not reachable")


@pytest.fixture(autouse=True)
def schema():
    store.init_schema()
    yield
    store.close_pool()


def _tid() -> str:
    return f"test-{uuid.uuid4()}"


class TestScoreStore:
    def test_save_then_get(self):
        tid = _tid()
        store.save_score(tid, True, -0.2, "v1")

        row = store.get_score(tid)
        assert row["is_fraud"] is True
        assert row["score"] == pytest.approx(-0.2)
        assert row["model_version"] == "v1"

    def test_unknown_id_returns_none(self):
        assert store.get_score(_tid()) is None

    def test_redelivery_upserts_instead_of_duplicating(self):
        tid = _tid()
        store.save_score(tid, False, 0.1, "v1")
        store.save_score(tid, True, -0.3, "v2")

        row = store.get_score(tid)
        assert row["is_fraud"] is True
        assert row["model_version"] == "v2"

    def test_list_flagged_without_since(self):
        """Regression: an untyped NULL `since` raised IndeterminateDatatype."""
        tid = _tid()
        store.save_score(tid, True, -0.5, "v1")
        assert tid in {r["transaction_id"] for r in store.list_flagged(limit=1000)}

    def test_list_flagged_excludes_clean_and_respects_since(self):
        flagged, clean = _tid(), _tid()
        store.save_score(flagged, True, -0.5, "v1")
        store.save_score(clean, False, 0.5, "v1")

        ids = {r["transaction_id"] for r in store.list_flagged(limit=1000)}
        assert flagged in ids and clean not in ids

        future = datetime.now(timezone.utc) + timedelta(days=1)
        assert store.list_flagged(since=future) == []
