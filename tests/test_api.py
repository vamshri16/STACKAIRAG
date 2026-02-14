"""API endpoint tests using FastAPI TestClient.

Tests the thin HTTP layer (status codes, error handling, routing).
All core logic is mocked out.
"""

from unittest.mock import patch, MagicMock

import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.core.query_processor import QueryRefusalError
from app.models.schemas import QueryResponse, Source

client = TestClient(app)


# ---------------------------------------------------------------------------
# Health endpoint
# ---------------------------------------------------------------------------


class TestHealthEndpoint:
    def test_health_returns_200(self):
        response = client.get("/api/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert "mistral_configured" in data
        assert "documents_count" in data
        assert "chunks_count" in data


# ---------------------------------------------------------------------------
# Query endpoint
# ---------------------------------------------------------------------------


class TestQueryEndpoint:
    @patch("app.api.query.process_query")
    def test_valid_query_returns_200(self, mock_process):
        mock_process.return_value = QueryResponse(
            answer="Test answer.",
            sources=[],
            confidence=0.9,
            intent="KNOWLEDGE_SEARCH",
            processing_time_ms=100,
        )
        response = client.post(
            "/api/query",
            json={"query": "What is machine learning?"},
        )
        assert response.status_code == 200
        data = response.json()
        assert data["answer"] == "Test answer."
        assert data["confidence"] == 0.9

    @patch("app.api.query.process_query")
    def test_pii_returns_400(self, mock_process):
        mock_process.side_effect = QueryRefusalError("Contains PII")
        response = client.post(
            "/api/query",
            json={"query": "My SSN is 123-45-6789"},
        )
        assert response.status_code == 400
        assert "PII" in response.json()["detail"]

    @patch("app.api.query.process_query")
    def test_internal_error_returns_500(self, mock_process):
        mock_process.side_effect = RuntimeError("Something broke")
        response = client.post(
            "/api/query",
            json={"query": "What is AI?"},
        )
        assert response.status_code == 500

    def test_missing_query_field_returns_422(self):
        response = client.post("/api/query", json={})
        assert response.status_code == 422

    @patch("app.api.query.process_query")
    def test_query_with_sources(self, mock_process):
        mock_process.return_value = QueryResponse(
            answer="Answer here.",
            sources=[
                Source(
                    chunk_id="a.pdf_p1_c0",
                    source="a.pdf",
                    page=1,
                    text="Some chunk text.",
                    score=0.92,
                ),
            ],
            confidence=0.85,
            intent="KNOWLEDGE_SEARCH",
            processing_time_ms=200,
        )
        response = client.post(
            "/api/query",
            json={"query": "What is AI?", "include_sources": True},
        )
        assert response.status_code == 200
        data = response.json()
        assert len(data["sources"]) == 1
        assert data["sources"][0]["source"] == "a.pdf"


# ---------------------------------------------------------------------------
# Ingestion endpoint
# ---------------------------------------------------------------------------


class TestIngestionEndpoint:
    def test_non_pdf_returns_failed(self):
        response = client.post(
            "/api/ingest",
            files=[("files", ("test.txt", b"not a pdf", "text/plain"))],
        )
        assert response.status_code == 200
        data = response.json()
        assert data["failed"] == 1
        assert "PDF" in data["results"][0]["message"]

    def test_empty_file_returns_failed(self):
        response = client.post(
            "/api/ingest",
            files=[("files", ("test.pdf", b"", "application/pdf"))],
        )
        assert response.status_code == 200
        data = response.json()
        assert data["failed"] == 1


# ---------------------------------------------------------------------------
# Documents endpoint
# ---------------------------------------------------------------------------


class TestDocumentsEndpoint:
    def test_list_documents_returns_200(self):
        response = client.get("/api/documents")
        assert response.status_code == 200
        assert isinstance(response.json(), list)
