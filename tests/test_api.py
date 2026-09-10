"""Tests for the FastAPI /ingest endpoint."""
from __future__ import annotations

from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

from src.api.main import app


@pytest.fixture()
def client():
    yield TestClient(app)


class TestIngestEndpoint:
    @patch("src.api.kafka_producer.publish_transaction", return_value="test-txn-id")
    def test_returns_200(self, mock_publish, client):
        resp = client.post("/ingest", json={"features": [0.1] * 29})
        assert resp.status_code == 200

    @patch("src.api.kafka_producer.publish_transaction", return_value="test-txn-id")
    def test_response_has_transaction_id_and_status(self, mock_publish, client):
        resp = client.post("/ingest", json={"features": [0.1] * 29})
        data = resp.json()
        assert data["transaction_id"] == "test-txn-id"
        assert data["status"] == "queued"

    @patch("src.api.kafka_producer.publish_transaction", return_value="test-txn-id")
    def test_publishes_to_kafka(self, mock_publish, client):
        client.post("/ingest", json={"features": [0.1] * 29})
        mock_publish.assert_called_once()

    def test_invalid_request_returns_422(self, client):
        resp = client.post("/ingest", json={})
        assert resp.status_code == 422
