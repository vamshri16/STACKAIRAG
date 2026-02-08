"""Reranker — score-based sorting, threshold filtering, deduplication.

No cross-encoder.  No external ranking library.
Pure Python sorting on numeric scores produced by hybrid search.
"""

from app.models.schemas import Chunk


def rerank(
    scored_chunks: list[tuple[Chunk, float]],
    top_k: int,
    threshold: float,
) -> list[tuple[Chunk, float]]:
    """Post-process search results into the final ranked list.

    Steps:
    1. Filter out chunks with score below *threshold*.
    2. Sort descending by score.
    3. Deduplicate — keep only the highest-scoring chunk per (source, page).
    4. Truncate to *top_k* results.

    Returns a list of ``(Chunk, score)`` pairs.
    """
    if not scored_chunks:
        return []

    # Step 1 — Filter.
    filtered = [(chunk, score) for chunk, score in scored_chunks if score >= threshold]

    if not filtered:
        return []

    # Step 2 — Sort descending by score.
    filtered.sort(key=lambda x: x[1], reverse=True)

    # Step 3 — Deduplicate on (source, page).
    seen: set[tuple[str, int]] = set()
    deduped: list[tuple[Chunk, float]] = []
    for chunk, score in filtered:
        key = (chunk.source, chunk.page)
        if key not in seen:
            seen.add(key)
            deduped.append((chunk, score))

    # Step 4 — Truncate.
    return deduped[:top_k]
