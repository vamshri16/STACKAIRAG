"""Hybrid search — cosine similarity + keyword overlap.

Pure logic — no FastAPI imports.
All retrieval from first principles: NumPy for cosine, set intersection for keywords.
No FAISS, no scikit-learn, no rank-bm25.
"""

import logging

import numpy as np

from app.config import settings
from app.core.embeddings import get_embedding
from app.models.schemas import Chunk
from app.storage.vector_store import vector_store
from app.utils.text_utils import tokenize

logger = logging.getLogger(__name__)


def cosine_similarity(
    query_embedding: np.ndarray,
    embeddings_matrix: np.ndarray,
) -> np.ndarray:
    """Compute cosine similarity between a query vector and all stored vectors.

    Returns a 1-D array of shape ``(n_chunks,)`` with scores in [-1, 1].
    """
    dot_products = embeddings_matrix @ query_embedding  # (n,)
    query_norm = np.linalg.norm(query_embedding)
    chunk_norms = np.linalg.norm(embeddings_matrix, axis=1)  # (n,)
    similarities = dot_products / (chunk_norms * query_norm + 1e-10)
    return similarities


def keyword_score(query_tokens: list[str], chunk_tokens: list[str]) -> float:
    """Compute term-overlap score between tokenized query and chunk.

    Returns the fraction of query tokens found in the chunk (0.0–1.0).
    """
    if not query_tokens:
        return 0.0
    query_set = set(query_tokens)
    chunk_set = set(chunk_tokens)
    overlap = query_set & chunk_set
    return len(overlap) / len(query_set)


def hybrid_search(
    query: str,
    top_k: int | None = None,
) -> list[tuple[Chunk, float]]:
    """Full hybrid search: semantic + keyword scoring.

    1. Embed the query.
    2. Cosine similarity against all stored embeddings.
    3. Keyword overlap score for each chunk.
    4. Combine: final = α * semantic + (1-α) * keyword.
    5. Return top-k ``(Chunk, score)`` pairs sorted descending.

    Returns an empty list if the store is empty.
    """
    if top_k is None:
        top_k = settings.top_k_retrieval

    chunks, embeddings = vector_store.get_all()

    if not chunks or embeddings is None:
        return []

    # Step 1 — Embed the query.
    query_embedding = get_embedding(query)

    # Step 2 — Semantic scores.
    semantic_scores = cosine_similarity(query_embedding, embeddings)

    # Step 3 — Keyword scores.
    query_tokens = tokenize(query)
    keyword_scores = np.array(
        [keyword_score(query_tokens, tokenize(c.text)) for c in chunks],
        dtype=np.float64,
    )

    # Step 4 — Combine.
    alpha = settings.semantic_weight
    combined_scores = alpha * semantic_scores + (1 - alpha) * keyword_scores

    # Step 5 — Top-k by argsort.
    top_indices = np.argsort(combined_scores)[::-1][:top_k]

    results = [(chunks[i], float(combined_scores[i])) for i in top_indices]
    return results
