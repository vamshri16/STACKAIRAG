"""Query API endpoint.

Thin HTTP layer — no business logic, no search, no LLM calls.
Just: receive request → call core → return response.
"""

import logging

from fastapi import APIRouter, HTTPException

from app.core.query_processor import QueryRefusalError, process_query
from app.models.schemas import QueryRequest, QueryResponse

logger = logging.getLogger(__name__)

router = APIRouter()


@router.post("/api/query", response_model=QueryResponse)
def query(request: QueryRequest) -> QueryResponse:
    """Full RAG pipeline: intent → search → generate → verify."""
    try:
        return process_query(request)
    except QueryRefusalError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except Exception as exc:
        logger.exception("Query failed")
        raise HTTPException(
            status_code=500,
            detail=f"Query processing failed: {exc}",
        )
