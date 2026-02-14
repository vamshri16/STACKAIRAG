"""Hallucination filter — verify LLM answers against source chunks.

Two-tier confidence scoring:
1. Semantic: embed each answer sentence and check cosine similarity against
   source chunk embeddings (accurate, but requires an API call).
2. Token-overlap: keyword overlap fallback if embedding fails (cheap, shallow).
"""

import logging
import re

import numpy as np

from app.core.embeddings import EmbeddingError, get_embeddings_batch
from app.models.schemas import Chunk
from app.utils.text_utils import tokenize

logger = logging.getLogger(__name__)

_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")

# Sentences that are meta-statements or citations, not factual claims.
_SKIP_PREFIXES = (
    "i don't have",
    "i do not have",
    "based on the",
    "according to the",
    "[source:",
)

# Token-overlap: minimum keyword overlap ratio to consider a sentence supported.
_TOKEN_SUPPORT_THRESHOLD = 0.5

# Semantic: minimum cosine similarity to consider a sentence supported.
_SEMANTIC_SUPPORT_THRESHOLD = 0.6


def compute_confidence(answer: str, chunks: list[Chunk]) -> float:
    """Score how well *answer* is grounded in the source *chunks*.

    Tries semantic similarity first (embedding-based). Falls back to
    token-overlap if the embedding call fails.

    Returns a float in [0.0, 1.0].
    """
    if not answer or not answer.strip() or not chunks:
        return 0.0

    sentences = _split_scorable_sentences(answer)
    if not sentences:
        return 1.0

    # Try semantic check first.
    semantic_score = _semantic_confidence(sentences, chunks)
    if semantic_score is not None:
        return semantic_score

    # Fallback to token-overlap.
    return _token_overlap_confidence(sentences, chunks)


# ------------------------------------------------------------------
# Sentence splitting
# ------------------------------------------------------------------


def _split_scorable_sentences(answer: str) -> list[str]:
    """Split answer into sentences, filtering out meta-statements and fragments."""
    raw = _SENTENCE_SPLIT_RE.split(answer.strip())
    scorable: list[str] = []
    for s in raw:
        s = s.strip()
        if not s:
            continue
        lowered = s.lower()
        if any(lowered.startswith(p) for p in _SKIP_PREFIXES):
            continue
        if len(tokenize(s)) < 3:
            continue
        scorable.append(s)
    return scorable


# ------------------------------------------------------------------
# Semantic confidence (primary)
# ------------------------------------------------------------------


def _semantic_confidence(sentences: list[str], chunks: list[Chunk]) -> float | None:
    """Compute confidence using embedding similarity.

    Returns None if embeddings can't be obtained (caller should fall back).
    """
    try:
        sentence_embeddings = get_embeddings_batch(sentences)
        chunk_embeddings = get_embeddings_batch([c.text for c in chunks])
    except EmbeddingError as exc:
        logger.warning("Semantic hallucination check failed, falling back: %s", exc)
        return None

    # Cosine similarity: each sentence against each chunk.
    # sentence_embeddings: (S, D), chunk_embeddings: (C, D)
    s_norms = np.linalg.norm(sentence_embeddings, axis=1, keepdims=True)
    c_norms = np.linalg.norm(chunk_embeddings, axis=1, keepdims=True)
    s_normed = sentence_embeddings / np.maximum(s_norms, 1e-10)
    c_normed = chunk_embeddings / np.maximum(c_norms, 1e-10)

    sim_matrix = s_normed @ c_normed.T  # (S, C)
    best_per_sentence = sim_matrix.max(axis=1)  # (S,)

    supported = int((best_per_sentence >= _SEMANTIC_SUPPORT_THRESHOLD).sum())
    return supported / len(sentences)


# ------------------------------------------------------------------
# Token-overlap confidence (fallback)
# ------------------------------------------------------------------


def _token_overlap_confidence(sentences: list[str], chunks: list[Chunk]) -> float:
    """Compute confidence using keyword overlap between sentences and chunks."""
    chunk_token_sets = [set(tokenize(c.text)) for c in chunks]
    supported = 0

    for sentence in sentences:
        sentence_set = set(tokenize(sentence))
        if not sentence_set:
            continue

        best_overlap = max(
            len(sentence_set & ct) / len(sentence_set)
            for ct in chunk_token_sets
        )

        if best_overlap >= _TOKEN_SUPPORT_THRESHOLD:
            supported += 1

    return supported / len(sentences)
