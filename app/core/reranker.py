"""Reranker — threshold filtering, deduplication, MMR diversification.

No cross-encoder.  No external ranking library.
Pure Python + NumPy on scores produced by hybrid search.
"""

import numpy as np

from app.models.schemas import Chunk
from app.storage.vector_store import vector_store


def rerank(
    scored_chunks: list[tuple[Chunk, float]],
    top_k: int,
    threshold: float,
    mmr_lambda: float = 0.7,
) -> list[tuple[Chunk, float]]:
    """Post-process search results into the final ranked list.

    Steps:
    1. Filter out chunks with score below *threshold*.
    2. Deduplicate — keep only the highest-scoring chunk per (source, page).
    3. Apply MMR to balance relevance with diversity.

    Returns a list of ``(Chunk, score)`` pairs.
    """
    if not scored_chunks:
        return []

    # Step 1 — Filter.
    filtered = [(chunk, score) for chunk, score in scored_chunks if score >= threshold]
    if not filtered:
        return []

    # Step 2 — Deduplicate on (source, page), keeping the highest score.
    filtered.sort(key=lambda x: x[1], reverse=True)
    seen: set[tuple[str, int]] = set()
    deduped: list[tuple[Chunk, float]] = []
    for chunk, score in filtered:
        key = (chunk.source, chunk.page)
        if key not in seen:
            seen.add(key)
            deduped.append((chunk, score))

    # Step 3 — MMR diversification.
    return _mmr_select(deduped, top_k, mmr_lambda)


def _mmr_select(
    candidates: list[tuple[Chunk, float]],
    top_k: int,
    lam: float,
) -> list[tuple[Chunk, float]]:
    """Select top-k chunks using Maximal Marginal Relevance.

    At each step, pick the candidate that maximizes:
        λ * relevance - (1-λ) * max_similarity_to_already_selected

    Falls back to simple truncation if embeddings are unavailable.
    """
    if len(candidates) <= top_k:
        return candidates

    # Build an embedding matrix for the candidates.
    embeddings = _get_candidate_embeddings(candidates)
    if embeddings is None:
        return candidates[:top_k]

    # Precompute pairwise cosine similarities.
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-10)
    normed = embeddings / norms
    sim_matrix = normed @ normed.T  # (n, n)

    # Normalize relevance scores to [0, 1] for fair weighting against similarity.
    scores = np.array([score for _, score in candidates])
    score_min, score_max = scores.min(), scores.max()
    if score_max > score_min:
        relevance = (scores - score_min) / (score_max - score_min)
    else:
        relevance = np.ones_like(scores)

    # Greedy MMR selection.
    selected_indices: list[int] = []
    remaining = set(range(len(candidates)))

    for _ in range(top_k):
        best_idx = -1
        best_mmr = -float("inf")

        for idx in remaining:
            rel = relevance[idx]

            if selected_indices:
                max_sim = max(sim_matrix[idx, s] for s in selected_indices)
            else:
                max_sim = 0.0

            mmr_score = lam * rel - (1 - lam) * max_sim

            if mmr_score > best_mmr:
                best_mmr = mmr_score
                best_idx = idx

        if best_idx == -1:
            break

        selected_indices.append(best_idx)
        remaining.discard(best_idx)

    return [candidates[i] for i in selected_indices]


def _get_candidate_embeddings(
    candidates: list[tuple[Chunk, float]],
) -> np.ndarray | None:
    """Look up embeddings for candidate chunks from the vector store."""
    all_chunks, all_embeddings = vector_store.get_all()
    if all_embeddings is None or not all_chunks:
        return None

    # Build a chunk_id → index lookup.
    id_to_idx = {c.chunk_id: i for i, c in enumerate(all_chunks)}

    rows = []
    for chunk, _ in candidates:
        idx = id_to_idx.get(chunk.chunk_id)
        if idx is None:
            return None  # Can't find embedding — fall back to simple truncation.
        rows.append(all_embeddings[idx])

    return np.vstack(rows)
