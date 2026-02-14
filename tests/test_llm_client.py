"""Tests for app/core/llm_client.py.

Unit tests for pure logic (format_context, build_qa_prompt, _parse_chat_response).
Live API tests for generate() and generate_chitchat_response().
"""

from unittest.mock import patch

import pytest

from app.core.llm_client import (
    LLMError,
    format_context,
    build_qa_prompt,
    generate,
    generate_chitchat_response,
    _parse_chat_response,
)
from app.models.schemas import Chunk
from tests.conftest import requires_api_key


# ---------------------------------------------------------------------------
# Unit tests — pure logic, no API calls
# ---------------------------------------------------------------------------


def _chunk(text: str, source: str = "test.pdf", page: int = 1) -> Chunk:
    return Chunk(chunk_id="x", text=text, source=source, page=page)


class TestFormatContext:
    def test_single_chunk(self):
        chunks = [(_chunk("Some text here.", "doc.pdf", 1), 0.92)]
        result = format_context(chunks)
        assert "[Source: doc.pdf, Page 1]" in result
        assert "Some text here." in result
        # Scores must NOT leak into the LLM context.
        assert "Score" not in result
        assert "0.92" not in result

    def test_multiple_chunks(self):
        chunks = [
            (_chunk("First chunk.", "a.pdf", 1), 0.95),
            (_chunk("Second chunk.", "b.pdf", 2), 0.80),
        ]
        result = format_context(chunks)
        assert "a.pdf" in result
        assert "b.pdf" in result
        assert "First chunk." in result
        assert "Second chunk." in result

    def test_empty_list(self):
        assert format_context([]) == ""


class TestBuildQaPrompt:
    def test_returns_two_messages(self):
        messages = build_qa_prompt("What is AI?", "Some context here.")
        assert len(messages) == 2
        assert messages[0]["role"] == "system"
        assert messages[1]["role"] == "user"

    def test_user_message_contains_query_and_context(self):
        messages = build_qa_prompt("my question", "my context")
        user_content = messages[1]["content"]
        assert "my question" in user_content
        assert "my context" in user_content

    def test_system_prompt_has_rules(self):
        messages = build_qa_prompt("q", "c")
        assert "ONLY" in messages[0]["content"]
        assert "Source:" in messages[0]["content"]

    def test_factual_no_extra_instruction(self):
        messages = build_qa_prompt("q", "c", sub_intent="FACTUAL")
        system = messages[0]["content"]
        assert "numbered list" not in system
        assert "comparison" not in system
        assert "summary" not in system.lower().split("concise")[-1:]

    def test_list_sub_intent(self):
        messages = build_qa_prompt("q", "c", sub_intent="LIST")
        assert "numbered list" in messages[0]["content"]

    def test_comparison_sub_intent(self):
        messages = build_qa_prompt("q", "c", sub_intent="COMPARISON")
        assert "comparison" in messages[0]["content"]

    def test_summary_sub_intent(self):
        messages = build_qa_prompt("q", "c", sub_intent="SUMMARY")
        assert "concise summary" in messages[0]["content"]

    def test_default_sub_intent_is_factual(self):
        default = build_qa_prompt("q", "c")
        factual = build_qa_prompt("q", "c", sub_intent="FACTUAL")
        assert default[0]["content"] == factual[0]["content"]


class TestParseChatResponse:
    def test_valid_response(self):
        body = {
            "choices": [
                {"message": {"content": "The answer is 42."}}
            ]
        }
        assert _parse_chat_response(body) == "The answer is 42."

    def test_missing_choices_raises(self):
        with pytest.raises(LLMError, match="Unexpected"):
            _parse_chat_response({"result": "something"})

    def test_empty_choices_raises(self):
        with pytest.raises(LLMError, match="Unexpected"):
            _parse_chat_response({"choices": []})

    def test_none_body_raises(self):
        with pytest.raises(LLMError, match="Unexpected"):
            _parse_chat_response(None)


class TestGenerateValidation:
    @patch("app.core.llm_client.settings")
    def test_missing_api_key_raises(self, mock_settings):
        mock_settings.mistral_api_key = ""
        with pytest.raises(LLMError, match="not configured"):
            generate([{"role": "user", "content": "hello"}])


# ---------------------------------------------------------------------------
# Live API tests — require MISTRAL_API_KEY
# ---------------------------------------------------------------------------


@requires_api_key
class TestGenerateLive:
    def test_simple_qa(self):
        chunk = _chunk(
            "Machine learning enables computers to learn from data "
            "without being explicitly programmed.",
            source="ml_intro.pdf",
            page=1,
        )
        context = format_context([(chunk, 0.95)])
        messages = build_qa_prompt("What is machine learning?", context)
        answer = generate(messages)
        assert isinstance(answer, str)
        assert len(answer) > 10

    def test_response_references_context(self):
        chunk = _chunk(
            "Revenue grew by 25% to $10 million in fiscal year 2023.",
            source="financials.pdf",
            page=3,
        )
        context = format_context([(chunk, 0.90)])
        messages = build_qa_prompt("What was the revenue?", context)
        answer = generate(messages)
        # The answer should mention something about revenue.
        assert "revenue" in answer.lower() or "10" in answer or "25" in answer


@requires_api_key
class TestGenerateChitchatLive:
    def test_chitchat_response(self):
        answer = generate_chitchat_response("Hello!")
        assert isinstance(answer, str)
        assert len(answer) > 0

    def test_chitchat_mentions_documents(self):
        answer = generate_chitchat_response("Hi there, how are you?")
        # The system prompt tells it to mention documents.
        answer_lower = answer.lower()
        assert any(
            word in answer_lower
            for word in ["document", "help", "question", "assist", "pdf"]
        )
