"""Tests for app/core/embeddings.py.

Unit tests mock the HTTP calls. Integration tests hit the real Mistral API.
"""

from unittest.mock import patch, MagicMock

import numpy as np
import pytest

from app.core.embeddings import (
    EmbeddingError,
    get_embedding,
    get_embeddings_batch,
    _parse_embedding_response,
)
from tests.conftest import requires_api_key


# ---------------------------------------------------------------------------
# Unit tests — pure logic, no API calls
# ---------------------------------------------------------------------------


class TestParseEmbeddingResponse:
    def test_valid_response(self):
        body = {
            "data": [
                {"index": 0, "embedding": [0.1, 0.2, 0.3]},
                {"index": 1, "embedding": [0.4, 0.5, 0.6]},
            ]
        }
        result = _parse_embedding_response(body, expected=2)
        assert isinstance(result, np.ndarray)
        assert result.shape == (2, 3)
        np.testing.assert_array_almost_equal(result[0], [0.1, 0.2, 0.3])

    def test_out_of_order_indices_sorted(self):
        body = {
            "data": [
                {"index": 1, "embedding": [0.4, 0.5]},
                {"index": 0, "embedding": [0.1, 0.2]},
            ]
        }
        result = _parse_embedding_response(body, expected=2)
        np.testing.assert_array_almost_equal(result[0], [0.1, 0.2])
        np.testing.assert_array_almost_equal(result[1], [0.4, 0.5])

    def test_count_mismatch_raises(self):
        body = {
            "data": [
                {"index": 0, "embedding": [0.1, 0.2]},
            ]
        }
        with pytest.raises(EmbeddingError, match="Expected 2"):
            _parse_embedding_response(body, expected=2)

    def test_missing_data_key_raises(self):
        with pytest.raises(EmbeddingError, match="Unexpected"):
            _parse_embedding_response({"result": []}, expected=1)

    def test_none_body_raises(self):
        with pytest.raises(EmbeddingError, match="Unexpected"):
            _parse_embedding_response(None, expected=1)


class TestGetEmbeddingsBatchValidation:
    def test_empty_list_raises(self):
        with pytest.raises(EmbeddingError, match="empty"):
            get_embeddings_batch([])

    @patch("app.core.embeddings.settings")
    def test_missing_api_key_raises(self, mock_settings):
        mock_settings.mistral_api_key = ""
        with pytest.raises(EmbeddingError, match="not configured"):
            get_embeddings_batch(["hello"])


# ---------------------------------------------------------------------------
# Live API tests — require MISTRAL_API_KEY
# ---------------------------------------------------------------------------


@requires_api_key
class TestGetEmbeddingLive:
    def test_single_embedding(self):
        result = get_embedding("What is machine learning?")
        assert isinstance(result, np.ndarray)
        assert result.ndim == 1
        assert result.shape[0] > 0
        assert np.isfinite(result).all()

    def test_embedding_is_nonzero(self):
        result = get_embedding("Hello world")
        assert np.linalg.norm(result) > 0


@requires_api_key
class TestGetEmbeddingsBatchLive:
    def test_batch_of_two(self):
        texts = [
            "Machine learning is a subset of AI.",
            "Revenue increased by 20% in 2023.",
        ]
        result = get_embeddings_batch(texts)
        assert isinstance(result, np.ndarray)
        assert result.shape[0] == 2
        assert result.ndim == 2
        assert result.shape[1] > 0

    def test_single_item_batch(self):
        result = get_embeddings_batch(["Just one text."])
        assert result.shape[0] == 1

    def test_different_texts_different_embeddings(self):
        texts = [
            "Machine learning algorithms.",
            "Revenue and financial earnings.",
        ]
        result = get_embeddings_batch(texts)
        # The two embeddings should not be identical.
        assert not np.allclose(result[0], result[1])
