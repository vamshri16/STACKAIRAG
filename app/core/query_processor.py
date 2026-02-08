"""Query pipeline orchestrator.

Pure logic — no FastAPI imports.  Coordinates:
  pii_detector → intent → search → rerank → llm → hallucination filter

Fail fast, fail loud on every step.
"""

import logging
import re
import time

from app.config import settings
from app.core.hallucination_filter import compute_confidence
from app.core.llm_client import format_context, build_qa_prompt, generate, generate_chitchat_response
from app.core.reranker import rerank
from app.core.search import hybrid_search
from app.models.schemas import QueryRequest, QueryResponse, Source
from app.utils.pii_detector import contains_pii, get_pii_types

logger = logging.getLogger(__name__)


class QueryRefusalError(Exception):
    """Raised when a query is refused (e.g., contains PII)."""


# ------------------------------------------------------------------
# Intent detection
# ------------------------------------------------------------------

_CHITCHAT_PATTERNS = [
    re.compile(r"^(hi|hello|hey|howdy|greetings)\b", re.IGNORECASE),
    re.compile(r"^how are you", re.IGNORECASE),
    re.compile(r"^good (morning|afternoon|evening)", re.IGNORECASE),
    re.compile(r"^(thanks|thank you|thx)\b", re.IGNORECASE),
    re.compile(r"^(bye|goodbye|see you)\b", re.IGNORECASE),
    re.compile(r"^what'?s up", re.IGNORECASE),
]


def detect_intent(query: str) -> str:
    """Classify the query into an intent category.

    Returns one of: ``"CHITCHAT"``, ``"KNOWLEDGE_SEARCH"``.
    """
    stripped = query.strip()
    for pattern in _CHITCHAT_PATTERNS:
        if pattern.search(stripped):
            return "CHITCHAT"
    return "KNOWLEDGE_SEARCH"


# ------------------------------------------------------------------
# Pipeline orchestrator
# ------------------------------------------------------------------


def process_query(request: QueryRequest) -> QueryResponse:
    """Execute the full query pipeline.

    Steps:
    1. PII check — refuse if detected.
    2. Intent detection.
    3. Chitchat handling (no retrieval).
    4. Hybrid search.
    5. Rerank.
    6. Handle no results.
    7. Generate answer via LLM.
    8. Hallucination check.
    9. Build response.

    Raises ``QueryRefusalError`` on PII detection.
    """
    start_time = time.time()

    # Step 1 — PII check.
    if contains_pii(request.query):
        pii_types = get_pii_types(request.query)
        raise QueryRefusalError(
            f"Query contains PII ({', '.join(pii_types)}). "
            "Please remove personal information and try again."
        )

    # Step 2 — Intent detection.
    intent = detect_intent(request.query)

    # Step 3 — Chitchat (no retrieval needed).
    if intent == "CHITCHAT":
        answer = generate_chitchat_response(request.query)
        elapsed = int((time.time() - start_time) * 1000)
        return QueryResponse(
            answer=answer,
            sources=[],
            confidence=1.0,
            intent=intent,
            processing_time_ms=elapsed,
        )

    # Step 4 — Hybrid search.
    candidates = hybrid_search(request.query, top_k=settings.top_k_retrieval)

    # Step 5 — Rerank.
    top_k = request.top_k or settings.top_k_final
    ranked = rerank(
        candidates,
        top_k=top_k,
        threshold=settings.similarity_threshold,
    )

    # Step 6 — No results.
    if not ranked:
        elapsed = int((time.time() - start_time) * 1000)
        return QueryResponse(
            answer=(
                "I don't have enough information in the provided "
                "documents to answer this question."
            ),
            sources=[],
            confidence=0.0,
            intent=intent,
            processing_time_ms=elapsed,
        )

    # Step 7 — Generate answer.
    context = format_context(ranked)
    messages = build_qa_prompt(request.query, context)
    answer = generate(messages)

    # Step 8 — Hallucination check.
    confidence = compute_confidence(answer, [chunk for chunk, _ in ranked])

    # Step 9 — Build sources list.
    sources: list[Source] = []
    if request.include_sources:
        for chunk, score in ranked:
            sources.append(
                Source(
                    chunk_id=chunk.chunk_id,
                    source=chunk.source,
                    page=chunk.page,
                    text=chunk.text[:500],
                    score=round(score, 4),
                )
            )

    elapsed = int((time.time() - start_time) * 1000)

    logger.info(
        "Query processed: intent=%s, sources=%d, confidence=%.2f, time=%dms",
        intent, len(sources), confidence, elapsed,
    )

    return QueryResponse(
        answer=answer,
        sources=sources,
        confidence=round(confidence, 4),
        intent=intent,
        processing_time_ms=elapsed,
    )
