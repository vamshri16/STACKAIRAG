"""Hallucination filter — verify LLM answers against source chunks.

Computes a confidence score based on keyword overlap between
answer sentences and source chunks.  No extra API call needed.
"""

import re

from app.models.schemas import Chunk
from app.utils.text_utils import tokenize

# Reuse the sentence-splitting regex from text_utils.
_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")

# Sentences that are meta-statements or citations, not factual claims.
_SKIP_PREFIXES = (
    "i don't have",
    "i do not have",
    "based on the",
    "according to the",
    "[source:",
)

# Minimum keyword overlap ratio to consider a sentence "supported".
_SUPPORT_THRESHOLD = 0.5


def compute_confidence(answer: str, chunks: list[Chunk]) -> float:
    """Score how well *answer* is grounded in the source *chunks*.

    Algorithm:
    1. Split the answer into sentences.
    2. For each sentence, tokenize it and check keyword overlap against
       each source chunk's tokens.
    3. If the best overlap >= 0.5 for any chunk, the sentence is "supported".
    4. confidence = supported_count / total_sentences.

    Returns a float in [0.0, 1.0].
    """
    if not answer or not answer.strip():
        return 0.0

    if not chunks:
        return 0.0

    sentences = _SENTENCE_SPLIT_RE.split(answer.strip())
    sentences = [s.strip() for s in sentences if s.strip()]

    if not sentences:
        return 0.0

    # Pre-tokenize all chunks once.
    chunk_token_sets = [set(tokenize(c.text)) for c in chunks]

    scorable: list[str] = []
    supported = 0

    for sentence in sentences:
        # Skip citation-only or meta-statement sentences.
        lowered = sentence.lower().strip()
        if any(lowered.startswith(prefix) for prefix in _SKIP_PREFIXES):
            continue

        # Skip very short fragments (e.g., bare citations like "[Source: ...]").
        sentence_tokens = tokenize(sentence)
        if len(sentence_tokens) < 3:
            continue

        scorable.append(sentence)

        # Check overlap against each chunk.
        sentence_set = set(sentence_tokens)
        best_overlap = 0.0
        for chunk_tokens in chunk_token_sets:
            if not sentence_set:
                break
            overlap = len(sentence_set & chunk_tokens) / len(sentence_set)
            if overlap > best_overlap:
                best_overlap = overlap

        if best_overlap >= _SUPPORT_THRESHOLD:
            supported += 1

    if not scorable:
        # All sentences were skipped (meta-statements / citations only).
        return 1.0

    return supported / len(scorable)
