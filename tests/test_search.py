"""Unit tests for app/core/search.py.

Tests the math functions (cosine_similarity, keyword_score) in isolation.
hybrid_search() calls the Mistral API, so it's tested in integration tests.
"""

import numpy as np

from app.core.search import cosine_similarity, keyword_score


class TestCosineSimilarity:
    def test_identical_vectors(self):
        v = np.array([1.0, 2.0, 3.0])
        matrix = np.array([v])
        scores = cosine_similarity(v, matrix)
        assert scores.shape == (1,)
        assert abs(scores[0] - 1.0) < 1e-6

    def test_orthogonal_vectors(self):
        query = np.array([1.0, 0.0])
        matrix = np.array([[0.0, 1.0]])
        scores = cosine_similarity(query, matrix)
        assert abs(scores[0]) < 1e-6

    def test_opposite_vectors(self):
        query = np.array([1.0, 0.0])
        matrix = np.array([[-1.0, 0.0]])
        scores = cosine_similarity(query, matrix)
        assert abs(scores[0] - (-1.0)) < 1e-6

    def test_multiple_vectors(self):
        query = np.array([1.0, 0.0, 0.0])
        matrix = np.array([
            [1.0, 0.0, 0.0],  # identical → ~1.0
            [0.0, 1.0, 0.0],  # orthogonal → ~0.0
            [-1.0, 0.0, 0.0], # opposite → ~-1.0
        ])
        scores = cosine_similarity(query, matrix)
        assert scores.shape == (3,)
        assert scores[0] > 0.99
        assert abs(scores[1]) < 1e-6
        assert scores[2] < -0.99

    def test_zero_vector_no_crash(self):
        query = np.array([0.0, 0.0, 0.0])
        matrix = np.array([[1.0, 2.0, 3.0]])
        scores = cosine_similarity(query, matrix)
        assert scores.shape == (1,)
        # Should not crash due to 1e-10 guard.
        assert np.isfinite(scores[0])


class TestKeywordScore:
    def test_full_overlap(self):
        query = ["machine", "learning"]
        chunk = ["machine", "learning", "algorithms", "data"]
        assert keyword_score(query, chunk) == 1.0

    def test_no_overlap(self):
        query = ["machine", "learning"]
        chunk = ["revenue", "earnings", "profit"]
        assert keyword_score(query, chunk) == 0.0

    def test_partial_overlap(self):
        query = ["machine", "learning", "deep", "neural"]
        chunk = ["machine", "learning", "algorithms"]
        assert keyword_score(query, chunk) == 0.5  # 2 of 4

    def test_empty_query(self):
        assert keyword_score([], ["some", "tokens"]) == 0.0

    def test_empty_chunk(self):
        assert keyword_score(["machine", "learning"], []) == 0.0
