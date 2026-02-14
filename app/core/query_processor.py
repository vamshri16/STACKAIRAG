"""Query pipeline orchestrator.

Pure logic — no FastAPI imports.  Coordinates:
  pii_detector → intent → search → rerank → llm → hallucination filter

Fail fast, fail loud on every step.
"""

import logging
import re
import time

from app.config import settings
from app.core.context_expander import expand_context
from app.core.hallucination_filter import compute_confidence
from app.core.llm_client import (
    build_qa_prompt,
    format_context,
    generate,
    generate_chitchat_response,
    rewrite_query,
)
from app.core.reranker import rerank
from app.core.search import hybrid_search
from app.models.schemas import Chunk, QueryRequest, QueryResponse, Source
from app.utils.pii_detector import contains_pii, get_pii_types, redact_pii

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
# Sub-intent detection (answer shaping)
# ------------------------------------------------------------------

_SUB_INTENT_PATTERNS: list[tuple[str, re.Pattern]] = [
    ("LIST", re.compile(
        r"(list|enumerate|what are the|name the|give me .* examples|steps to|ways to)",
        re.IGNORECASE,
    )),
    ("COMPARISON", re.compile(
        r"(compare|comparison|versus|vs\.?|differ|difference|pros and cons|advantages .* disadvantages)",
        re.IGNORECASE,
    )),
    ("SUMMARY", re.compile(
        r"(summarize|summarise|summary|overview|brief|in short|tldr|tl;dr)",
        re.IGNORECASE,
    )),
]


def detect_sub_intent(query: str) -> str:
    """Detect the answer format sub-intent for knowledge queries.

    Returns one of: ``"LIST"``, ``"COMPARISON"``, ``"SUMMARY"``, ``"FACTUAL"``.
    """
    for label, pattern in _SUB_INTENT_PATTERNS:
        if pattern.search(query):
            return label
    return "FACTUAL"


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------


def _merge_search_results(
    *result_lists: list[tuple[Chunk, float]],
) -> list[tuple[Chunk, float]]:
    """Merge multiple search result lists, keeping the highest score per chunk."""
    best: dict[str, tuple[Chunk, float]] = {}
    for results in result_lists:
        for chunk, score in results:
            if chunk.chunk_id not in best or score > best[chunk.chunk_id][1]:
                best[chunk.chunk_id] = (chunk, score)
    return sorted(best.values(), key=lambda x: x[1], reverse=True)


# ------------------------------------------------------------------
# Pipeline orchestrator
# ------------------------------------------------------------------


def process_query(request: QueryRequest) -> QueryResponse:
    """Execute the full query pipeline.

    Steps:
    1. PII check — refuse if detected.
    2. Intent detection.
    3. Chitchat handling (no retrieval).
    4. Query rewrite (LLM rewrites for better retrieval).
    5. Hybrid search (using rewritten query).
    6. Rerank.
    7. Conditional context expansion (+ re-rerank if expanded).
    8. Handle no results.
    9. Detect sub-intent and generate answer via LLM (using original query).
    10. Hallucination check.
    11. Build response.

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

    # Step 4 — Rewrite query for better retrieval.
    rewritten = rewrite_query(request.query)

    # Step 5 — Hybrid search with both original and rewritten queries.
    original_results = hybrid_search(request.query, top_k=settings.top_k_retrieval)
    rewritten_results = (
        hybrid_search(rewritten, top_k=settings.top_k_retrieval)
        if rewritten != request.query
        else []
    )
    candidates = _merge_search_results(original_results, rewritten_results)

    # Step 6 — Rerank.
    top_k = request.top_k or settings.top_k_final
    ranked = rerank(
        candidates,
        top_k=top_k,
        threshold=settings.similarity_threshold,
    )

    # Step 7 — Conditional context expansion.
    expanded = expand_context(ranked, top_k=top_k)
    if len(expanded) > len(ranked):
        ranked = rerank(
            expanded,
            top_k=top_k,
            threshold=settings.similarity_threshold,
        )

    # Step 8 — No results.
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

    # Step 9 — Detect sub-intent and generate answer (using original query).
    sub_intent = detect_sub_intent(request.query)
    context = format_context(ranked)
    messages = build_qa_prompt(request.query, context, sub_intent=sub_intent)
    answer = generate(messages)

    # Step 10 — Redact PII from LLM response.
    answer = redact_pii(answer)

    # Step 11 — Hallucination check.
    confidence = compute_confidence(answer, [chunk for chunk, _ in ranked])

    # Step 12 — Build sources list.
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
