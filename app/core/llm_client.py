"""Mistral AI chat completions client.

Synchronous — same pattern as embeddings.py.
Pure logic — no FastAPI imports.  Testable in isolation.
"""

import logging
import time

import httpx

from app.config import settings
from app.models.schemas import Chunk

logger = logging.getLogger(__name__)

_MAX_RETRIES = 3
_INITIAL_BACKOFF_SECONDS = 1.0

_QA_SYSTEM_PROMPT = """\
You are a precise, helpful assistant that answers questions based ONLY on the \
provided context from PDF documents.

Rules:
1. Answer ONLY from the provided context. Do not use prior knowledge.
2. Cite your sources inline using [Source: filename, Page N] format.
3. If the context does not contain enough information to answer, say: \
"I don't have enough information in the provided documents to answer this question."
4. Be concise and direct. Answer ONLY what the question asks — do not include \
additional information from the context that was not requested.
5. If multiple sources support the answer, cite all of them.
6. NEVER include personally identifiable information (PII) in your answer — \
this includes phone numbers, email addresses, SSNs, credit card numbers, and \
physical addresses. If asked for PII, respond: "I cannot share personal \
information from the documents.\""""

_QUERY_REWRITE_SYSTEM_PROMPT = """\
You are a search query optimizer. Rewrite the user's question to be more \
specific and search-friendly for retrieving relevant passages from PDF documents. \
Return ONLY the rewritten query, nothing else. Do not answer the question."""

# Sub-intent formatting instructions appended to the QA system prompt.
_SUB_INTENT_INSTRUCTIONS: dict[str, str] = {
    "LIST": "\n7. Format your answer as a numbered list.",
    "COMPARISON": (
        "\n7. Structure your answer as a clear comparison "
        "with arguments for and against each option."
    ),
    "SUMMARY": "\n7. Provide a concise summary in 2-4 sentences.",
    "TABLE": (
        "\n7. Format your answer as a markdown table with clear column headers. "
        "Use | column | column | format. Include a header row and separator row."
    ),
    "FACTUAL": "",
}

_CHITCHAT_SYSTEM_PROMPT = """\
You are a friendly assistant for a PDF knowledge base. \
Respond briefly to the greeting or small talk, then mention \
that you can help answer questions about the uploaded documents."""


class LLMError(Exception):
    """Raised when the chat completions API call fails."""


def _validate_api_key() -> None:
    if not settings.mistral_api_key:
        raise LLMError(
            "Mistral API key is not configured. "
            "Set MISTRAL_API_KEY in your .env file."
        )


# ------------------------------------------------------------------
# Context formatting
# ------------------------------------------------------------------


def format_context(chunks_with_scores: list[tuple[Chunk, float]]) -> str:
    """Convert retrieved chunks into a formatted context string for the prompt."""
    parts: list[str] = []
    for chunk, _score in chunks_with_scores:
        header = f"[Source: {chunk.source}, Page {chunk.page}]"
        parts.append(f"{header}\n{chunk.text}")
    return "\n\n".join(parts)


def build_qa_prompt(
    query: str, context: str, sub_intent: str = "FACTUAL"
) -> list[dict]:
    """Construct the messages array for the Mistral chat API.

    *sub_intent* selects an answer-shaping instruction appended to the
    system prompt (LIST, COMPARISON, SUMMARY, or FACTUAL).
    """
    extra = _SUB_INTENT_INSTRUCTIONS.get(sub_intent, "")
    system_prompt = _QA_SYSTEM_PROMPT + extra
    return [
        {"role": "system", "content": system_prompt},
        {
            "role": "user",
            "content": (
                f"Context:\n{context}\n\n"
                f"Question: {query}\n\n"
                "Answer based only on the context above, with inline citations:"
            ),
        },
    ]


# ------------------------------------------------------------------
# Generation
# ------------------------------------------------------------------


def generate(messages: list[dict]) -> str:
    """Call the Mistral chat completions API and return the generated text.

    Retries on rate limits (429) and network errors.
    Fails fast on auth errors (401).
    """
    _validate_api_key()

    url = f"{settings.mistral_base_url}/chat/completions"
    headers = {
        "Authorization": f"Bearer {settings.mistral_api_key}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": settings.mistral_chat_model,
        "messages": messages,
    }

    last_error: Exception | None = None
    backoff = _INITIAL_BACKOFF_SECONDS

    for attempt in range(1, _MAX_RETRIES + 1):
        try:
            with httpx.Client(timeout=120.0) as client:
                response = client.post(url, json=payload, headers=headers)

            if response.status_code == 401:
                raise LLMError(
                    "Mistral API authentication failed (HTTP 401). "
                    "Check your MISTRAL_API_KEY."
                )

            if response.status_code == 429:
                if attempt < _MAX_RETRIES:
                    logger.warning(
                        "Rate limited (429). Retrying in %.1fs (attempt %d/%d).",
                        backoff, attempt, _MAX_RETRIES,
                    )
                    time.sleep(backoff)
                    backoff *= 2
                    continue
                raise LLMError("Mistral API rate limit exceeded after retries.")

            if response.status_code != 200:
                error_body = response.text[:500]
                raise LLMError(
                    f"Mistral API error (HTTP {response.status_code}): {error_body}"
                )

            return _parse_chat_response(response.json())

        except LLMError:
            raise

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

    raise LLMError(
        f"Mistral API request failed after {_MAX_RETRIES} attempts: {last_error}"
    )


def rewrite_query(query: str) -> str:
    """Rewrite a user query to improve retrieval.

    Returns the rewritten query, or the original if the LLM call fails.
    """
    messages = [
        {"role": "system", "content": _QUERY_REWRITE_SYSTEM_PROMPT},
        {"role": "user", "content": query},
    ]
    try:
        rewritten = generate(messages)
        rewritten = rewritten.strip().strip('"')
        if rewritten:
            logger.debug("Query rewritten: '%s' → '%s'", query, rewritten)
            return rewritten
    except LLMError as exc:
        logger.warning("Query rewrite failed, using original: %s", exc)
    return query


def generate_chitchat_response(query: str) -> str:
    """Handle chitchat queries without retrieval."""
    messages = [
        {"role": "system", "content": _CHITCHAT_SYSTEM_PROMPT},
        {"role": "user", "content": query},
    ]
    return generate(messages)


# ------------------------------------------------------------------
# Internal helpers
# ------------------------------------------------------------------


def _parse_chat_response(body: dict) -> str:
    """Extract the assistant message from the Mistral chat API response."""
    try:
        return body["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError) as exc:
        raise LLMError(
            f"Unexpected Mistral chat API response format: {exc}"
        ) from exc
