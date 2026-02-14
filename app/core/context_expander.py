"""Conditional context expansion — pull adjacent pages when retrieval is weak.

Pure logic — no FastAPI imports.
Operates on reranked results, expands only when scores or coverage are poor.
"""

import logging

from app.config import settings
from app.models.schemas import Chunk
from app.storage.vector_store import vector_store

logger = logging.getLogger(__name__)


def _needs_expansion(
    ranked: list[tuple[Chunk, float]],
    top_k: int,
) -> bool:
    """Decide whether the ranked results warrant context expansion.

    Expansion triggers when *either* condition is true:
    - Best score is below ``expansion_score_threshold``.
    - Coverage (results found / results requested) is below
      ``expansion_coverage_ratio``.
    """
    if not ranked:
        return False

    best_score = ranked[0][1]
    coverage = len(ranked) / top_k if top_k > 0 else 1.0

    score_weak = best_score < settings.expansion_score_threshold
    coverage_low = coverage < settings.expansion_coverage_ratio

    if score_weak or coverage_low:
        logger.debug(
            "Expansion triggered: best_score=%.3f (threshold=%.3f), "
            "coverage=%.2f (threshold=%.2f).",
            best_score, settings.expansion_score_threshold,
            coverage, settings.expansion_coverage_ratio,
        )
        return True

    return False


def _collect_neighbor_chunks(
    ranked: list[tuple[Chunk, float]],
) -> list[tuple[Chunk, float]]:
    """Look up ±1 page neighbors for each ranked chunk.

    Neighbors are assigned a discounted score so they rank below the
    chunk that triggered their inclusion.  Chunks already in ``ranked``
    are skipped.
    """
    existing_ids = {chunk.chunk_id for chunk, _ in ranked}
    neighbors: list[tuple[Chunk, float]] = []

    for chunk, score in ranked:
        adjacent_pages: set[int] = set()
        if chunk.page > 1:
            adjacent_pages.add(chunk.page - 1)
        adjacent_pages.add(chunk.page + 1)

        candidates = vector_store.get_chunks_by_source_and_pages(
            chunk.source, adjacent_pages,
        )

        neighbor_score = score * settings.expansion_neighbor_discount
        for candidate in candidates:
            if candidate.chunk_id not in existing_ids:
                existing_ids.add(candidate.chunk_id)
                neighbors.append((candidate, neighbor_score))

    return neighbors


def expand_context(
    ranked: list[tuple[Chunk, float]],
    top_k: int,
) -> list[tuple[Chunk, float]]:
    """Conditionally expand ranked results with adjacent-page chunks.

    If retrieval quality is strong (high scores, good coverage), the
    input is returned unchanged.  Otherwise, neighboring chunks are
    merged in and the combined list is sorted by score descending.
    """
    if not _needs_expansion(ranked, top_k):
        return ranked

    neighbors = _collect_neighbor_chunks(ranked)

    if not neighbors:
        return ranked

    merged = list(ranked) + neighbors
    merged.sort(key=lambda x: x[1], reverse=True)

    logger.info(
        "Context expanded: %d → %d candidates (+%d neighbors).",
        len(ranked), len(merged), len(neighbors),
    )

    return merged
