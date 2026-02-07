"""Mistral AI embeddings client.

Synchronous — no async magic yet.  Pure logic — no FastAPI imports.
Testable in isolation.
"""

import logging
import time

import httpx
import numpy as np

from app.config import settings

logger = logging.getLogger(__name__)

# Maximum texts per Mistral embeddings API call.
_BATCH_SIZE = 16

# Retry configuration.
_MAX_RETRIES = 3
_INITIAL_BACKOFF_SECONDS = 1.0


class EmbeddingError(Exception):
    """Raised when the embeddings API call fails."""


def _validate_api_key() -> None:
    """Fail fast if the Mistral API key is missing or empty."""
    if not settings.mistral_api_key:
        raise EmbeddingError(
            "Mistral API key is not configured. "
            "Set MISTRAL_API_KEY in your .env file."
        )


def get_embeddings_batch(texts: list[str]) -> np.ndarray:
    """Embed a list of texts via the Mistral API.

    Returns a NumPy matrix of shape ``(len(texts), embedding_dim)``.

    Handles:
    - Batch splitting (respects API limits).
    - Retry with exponential backoff on rate limits (HTTP 429) and
      transient network errors.
    - Fail-fast on missing API key or auth errors.

    Raises ``EmbeddingError`` on unrecoverable failures.
    """
    if not texts:
        raise EmbeddingError("Cannot embed an empty list of texts.")

    _validate_api_key()

    all_embeddings: list[np.ndarray] = []

    # Process in sub-batches to respect API limits.
    for batch_start in range(0, len(texts), _BATCH_SIZE):
        batch = texts[batch_start : batch_start + _BATCH_SIZE]
        batch_embeddings = _embed_batch_with_retry(batch)
        all_embeddings.append(batch_embeddings)

    return np.vstack(all_embeddings)


def get_embedding(text: str) -> np.ndarray:
    """Embed a single text string.

    Convenience wrapper around ``get_embeddings_batch``.
    Returns a 1-D NumPy array of shape ``(embedding_dim,)``.
    """
    matrix = get_embeddings_batch([text])
    return matrix[0]


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _embed_batch_with_retry(texts: list[str]) -> np.ndarray:
    """Call the Mistral embeddings endpoint with retry logic.

    Returns a NumPy matrix of shape ``(len(texts), embedding_dim)``.
    """
    url = f"{settings.mistral_base_url}/embeddings"
    headers = {
        "Authorization": f"Bearer {settings.mistral_api_key}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": settings.mistral_embed_model,
        "input": texts,
    }

    last_error: Exception | None = None
    backoff = _INITIAL_BACKOFF_SECONDS

    for attempt in range(1, _MAX_RETRIES + 1):
        try:
            with httpx.Client(timeout=60.0) as client:
                response = client.post(url, json=payload, headers=headers)

            # --- Auth error: fail immediately, retrying won't help ----------
            if response.status_code == 401:
                raise EmbeddingError(
                    "Mistral API authentication failed (HTTP 401). "
                    "Check your MISTRAL_API_KEY."
                )

            # --- Rate limit: back off and retry ----------------------------
            if response.status_code == 429:
                if attempt < _MAX_RETRIES:
                    logger.warning(
                        "Rate limited (429). Retrying in %.1fs (attempt %d/%d).",
                        backoff, attempt, _MAX_RETRIES,
                    )
                    time.sleep(backoff)
                    backoff *= 2
                    continue
                raise EmbeddingError(
                    "Mistral API rate limit exceeded after retries."
                )

            # --- Other HTTP errors ------------------------------------------
            if response.status_code != 200:
                error_body = response.text[:500]
                raise EmbeddingError(
                    f"Mistral API error (HTTP {response.status_code}): {error_body}"
                )

            # --- Parse successful response ----------------------------------
            return _parse_embedding_response(response.json(), expected=len(texts))

        except EmbeddingError:
            raise  # Don't retry on auth or parse errors.

        except httpx.HTTPError as exc:
            last_error = exc
            if attempt < _MAX_RETRIES:
                logger.warning(
                    "Network error: %s. Retrying in %.1fs (attempt %d/%d).",
                    exc, backoff, attempt, _MAX_RETRIES,
                )
                time.sleep(backoff)
                backoff *= 2
                continue

    raise EmbeddingError(
        f"Mistral API request failed after {_MAX_RETRIES} attempts: {last_error}"
    )


def _parse_embedding_response(
    body: dict,
    expected: int,
) -> np.ndarray:
    """Extract embedding vectors from the Mistral API response body.

    Returns a NumPy matrix of shape ``(expected, embedding_dim)``.
    """
    try:
        data = body["data"]
    except (KeyError, TypeError) as exc:
        raise EmbeddingError(
            f"Unexpected Mistral API response format: {exc}"
        ) from exc

    if len(data) != expected:
        raise EmbeddingError(
            f"Expected {expected} embeddings, got {len(data)}."
        )

    # Sort by index to guarantee ordering matches input ordering.
    data_sorted = sorted(data, key=lambda d: d["index"])

    vectors = [item["embedding"] for item in data_sorted]
    return np.array(vectors, dtype=np.float64)
