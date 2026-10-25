"""Tests for the FastAPI /ingest endpoint."""
from __future__ import annotations

from datetime import datetime, timezone
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

from src.api.kafka_producer import PublishError
from src.api.main import app


@pytest.fixture()
def client():
    yield TestClient(app)


class TestIngestEndpoint:
    @patch("src.api.main.publish_transaction", return_value="test-txn-id")
    def test_returns_200(self, mock_publish, client):
        resp = client.post("/ingest", json={"features": [0.1] * 29})
        assert resp.status_code == 200

    @patch("src.api.main.publish_transaction", return_value="test-txn-id")
    def test_response_has_transaction_id_and_status(self, mock_publish, client):
        resp = client.post("/ingest", json={"features": [0.1] * 29})
        data = resp.json()
        assert data["transaction_id"] == "test-txn-id"
        assert data["status"] == "queued"

    @patch("src.api.main.publish_transaction", return_value="test-txn-id")
    def test_publishes_to_kafka(self, mock_publish, client):
        client.post("/ingest", json={"features": [0.1] * 29})
        mock_publish.assert_called_once()

    def test_invalid_request_returns_422(self, client):
        resp = client.post("/ingest", json={})
        assert resp.status_code == 422

    def test_wrong_feature_count_returns_422(self, client):
        resp = client.post("/ingest", json={"features": [0.1] * 5})
        assert resp.status_code == 422

    @patch("src.api.main.publish_transaction", side_effect=PublishError("broker down"))
    def test_publish_failure_returns_503(self, mock_publish, client):
        """A caller must never get an ID for a transaction Kafka did not accept."""
        resp = client.post("/ingest", json={"features": [0.1] * 29})
        assert resp.status_code == 503


SCORED = {
    "transaction_id": "txn-1",
    "is_fraud": True,
    "score": -0.2,
    "model_version": "v1",
    "scored_at": datetime(2026, 10, 18, tzinfo=timezone.utc),
}


class TestGetTransaction:
    @patch("src.api.main.store")
    def test_scored_transaction_returned(self, store, client):
        store.get_score.return_value = SCORED
        resp = client.get("/transactions/txn-1")
        assert resp.status_code == 200
        body = resp.json()
        assert body["status"] == "scored"
        assert body["is_fraud"] is True
        assert body["model_version"] == "v1"

    @patch("src.api.main.store")
    def test_unscored_transaction_is_404(self, store, client):
        store.get_score.return_value = None
        assert client.get("/transactions/txn-1").status_code == 404


class TestListFlagged:
    @patch("src.api.main.store")
    def test_returns_flagged(self, store, client):
        store.list_flagged.return_value = [SCORED]
        resp = client.get("/transactions")
        assert resp.status_code == 200
        assert [r["transaction_id"] for r in resp.json()] == ["txn-1"]

    @patch("src.api.main.store")
    def test_limit_above_cap_is_rejected(self, store, client):
        assert client.get("/transactions?limit=1000000").status_code == 422
        store.list_flagged.assert_not_called()
