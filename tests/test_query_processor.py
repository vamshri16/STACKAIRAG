"""Unit tests for app/core/query_processor.py.

Tests detect_intent() (pure logic) and process_query() (mocked dependencies).
"""

from unittest.mock import patch, MagicMock

import pytest

from app.core.query_processor import (
    detect_intent,
    detect_sub_intent,
    process_query,
    QueryRefusalError,
)
from app.models.schemas import Chunk, QueryRequest


# ---------------------------------------------------------------------------
# detect_intent — pure logic, no mocking needed
# ---------------------------------------------------------------------------


class TestDetectIntent:
    def test_hello(self):
        assert detect_intent("hello") == "CHITCHAT"

    def test_hi_there(self):
        assert detect_intent("Hi there!") == "CHITCHAT"

    def test_hey(self):
        assert detect_intent("hey") == "CHITCHAT"

    def test_how_are_you(self):
        assert detect_intent("How are you doing?") == "CHITCHAT"

    def test_good_morning(self):
        assert detect_intent("Good morning") == "CHITCHAT"

    def test_thanks(self):
        assert detect_intent("Thanks!") == "CHITCHAT"

    def test_bye(self):
        assert detect_intent("Bye") == "CHITCHAT"

    def test_whats_up(self):
        assert detect_intent("What's up") == "CHITCHAT"

    def test_knowledge_query(self):
        assert detect_intent("What is machine learning?") == "KNOWLEDGE_SEARCH"

    def test_knowledge_query_complex(self):
        assert detect_intent("Explain the revenue trends in Q3.") == "KNOWLEDGE_SEARCH"

    def test_empty_string(self):
        assert detect_intent("") == "KNOWLEDGE_SEARCH"

    def test_whitespace(self):
        assert detect_intent("   ") == "KNOWLEDGE_SEARCH"

    def test_hello_in_middle_not_chitchat(self):
        """Chitchat patterns are anchored to the start."""
        assert detect_intent("Can you say hello?") == "KNOWLEDGE_SEARCH"

    def test_case_insensitive(self):
        assert detect_intent("HELLO") == "CHITCHAT"
        assert detect_intent("Hello") == "CHITCHAT"


# ---------------------------------------------------------------------------
# detect_sub_intent — pure logic, no mocking needed
# ---------------------------------------------------------------------------


class TestDetectSubIntent:
    def test_list_query(self):
        assert detect_sub_intent("List the key findings") == "LIST"

    def test_list_what_are(self):
        assert detect_sub_intent("What are the main benefits?") == "LIST"

    def test_list_steps(self):
        assert detect_sub_intent("Steps to configure the system") == "LIST"

    def test_comparison_compare(self):
        assert detect_sub_intent("Compare approach A and approach B") == "COMPARISON"

    def test_comparison_vs(self):
        assert detect_sub_intent("Redis vs Memcached") == "COMPARISON"

    def test_comparison_difference(self):
        assert detect_sub_intent("What is the difference between X and Y?") == "COMPARISON"

    def test_summary_summarize(self):
        assert detect_sub_intent("Summarize chapter 3") == "SUMMARY"

    def test_summary_overview(self):
        assert detect_sub_intent("Give me an overview of the report") == "SUMMARY"

    def test_summary_tldr(self):
        assert detect_sub_intent("tldr of the document") == "SUMMARY"

    def test_table_tabulate(self):
        assert detect_sub_intent("Tabulate the results") == "TABLE"

    def test_table_breakdown(self):
        assert detect_sub_intent("Give me a breakdown of the features") == "TABLE"

    def test_table_side_by_side(self):
        assert detect_sub_intent("Show a side by side of the options") == "TABLE"

    def test_factual_default(self):
        assert detect_sub_intent("What is the default port?") == "FACTUAL"

    def test_factual_plain_question(self):
        assert detect_sub_intent("How does authentication work?") == "FACTUAL"

    def test_case_insensitive(self):
        assert detect_sub_intent("LIST the main topics") == "LIST"
        assert detect_sub_intent("COMPARE X and Y") == "COMPARISON"
        assert detect_sub_intent("SUMMARIZE the findings") == "SUMMARY"
        assert detect_sub_intent("TABULATE the data") == "TABLE"


# ---------------------------------------------------------------------------
# process_query — needs mocking for external dependencies
# ---------------------------------------------------------------------------


class TestProcessQueryPII:
    def test_pii_ssn_refused(self):
        request = QueryRequest(query="My SSN is 123-45-6789")
        with pytest.raises(QueryRefusalError, match="PII"):
            process_query(request)

    def test_pii_email_refused(self):
        request = QueryRequest(query="Contact me at user@example.com")
        with pytest.raises(QueryRefusalError, match="PII"):
            process_query(request)


class TestProcessQueryChitchat:
    @patch("app.core.query_processor.generate_chitchat_response")
    def test_chitchat_returns_response(self, mock_chitchat):
        mock_chitchat.return_value = "Hello! How can I help you?"
        request = QueryRequest(query="Hello")
        response = process_query(request)

        assert response.intent == "CHITCHAT"
        assert response.answer == "Hello! How can I help you?"
        assert response.sources == []
        assert response.confidence == 1.0
        mock_chitchat.assert_called_once_with("Hello")


class TestProcessQueryKnowledge:
    @patch("app.core.query_processor.compute_confidence")
    @patch("app.core.query_processor.generate")
    @patch("app.core.query_processor.build_qa_prompt")
    @patch("app.core.query_processor.format_context")
    @patch("app.core.query_processor.rerank")
    @patch("app.core.query_processor.hybrid_search")
    def test_full_knowledge_pipeline(
        self,
        mock_search,
        mock_rerank,
        mock_format,
        mock_build,
        mock_generate,
        mock_confidence,
    ):
        chunk = Chunk(
            chunk_id="a.pdf_p1_c0",
            text="Machine learning is great.",
            source="a.pdf",
            page=1,
        )
        mock_search.return_value = [(chunk, 0.9)]
        mock_rerank.return_value = [(chunk, 0.9)]
        mock_format.return_value = "formatted context"
        mock_build.return_value = [{"role": "user", "content": "test"}]
        mock_generate.return_value = "Machine learning is great."
        mock_confidence.return_value = 0.85

        request = QueryRequest(query="What is machine learning?")
        response = process_query(request)

        assert response.intent == "KNOWLEDGE_SEARCH"
        assert response.answer == "Machine learning is great."
        assert response.confidence == 0.85
        assert len(response.sources) == 1
        assert response.sources[0].source == "a.pdf"
        assert response.processing_time_ms >= 0

    @patch("app.core.query_processor.rerank")
    @patch("app.core.query_processor.hybrid_search")
    def test_no_results_returns_fallback(self, mock_search, mock_rerank):
        mock_search.return_value = []
        mock_rerank.return_value = []

        request = QueryRequest(query="What is quantum computing?")
        response = process_query(request)

        assert response.intent == "KNOWLEDGE_SEARCH"
        assert "don't have enough information" in response.answer
        assert response.confidence == 0.0
        assert response.sources == []

    @patch("app.core.query_processor.compute_confidence")
    @patch("app.core.query_processor.generate")
    @patch("app.core.query_processor.build_qa_prompt")
    @patch("app.core.query_processor.format_context")
    @patch("app.core.query_processor.rerank")
    @patch("app.core.query_processor.hybrid_search")
    def test_include_sources_false(
        self,
        mock_search,
        mock_rerank,
        mock_format,
        mock_build,
        mock_generate,
        mock_confidence,
    ):
        chunk = Chunk(
            chunk_id="a.pdf_p1_c0",
            text="Some text.",
            source="a.pdf",
            page=1,
        )
        mock_search.return_value = [(chunk, 0.9)]
        mock_rerank.return_value = [(chunk, 0.9)]
        mock_format.return_value = "context"
        mock_build.return_value = [{"role": "user", "content": "test"}]
        mock_generate.return_value = "Answer."
        mock_confidence.return_value = 0.9

        request = QueryRequest(
            query="What is this?", include_sources=False
        )
        response = process_query(request)

        assert response.sources == []
