"""Tests for domain value objects."""

import pytest
import math

from rag_core.domain.value_objects.embedding import Embedding
from rag_core.domain.value_objects.score import (
    SimilarityScore,
    ScoreType,
    RelevanceScore,
)


class TestEmbedding:
    """Tests for Embedding value object."""

    def test_create_embedding(self):
        """Test basic embedding creation."""
        emb = Embedding([0.1, 0.2, 0.3])

        assert emb.dimension == 3
        assert emb.values == (0.1, 0.2, 0.3)

    def test_embedding_requires_values(self):
        """Test that embedding requires non-empty values."""
        with pytest.raises(ValueError, match="cannot be empty"):
            Embedding([])

    def test_embedding_is_immutable(self):
        """Test that embedding is immutable."""
        emb = Embedding([0.1, 0.2, 0.3])
        with pytest.raises(AttributeError):
            emb.values = (0.4, 0.5, 0.6)

    def test_embedding_norm(self):
        """Test L2 norm calculation."""
        emb = Embedding([3.0, 4.0])
        assert emb.norm == 5.0

    def test_embedding_normalize(self):
        """Test normalization."""
        emb = Embedding([3.0, 4.0])
        normalized = emb.normalize()

        assert abs(normalized.norm - 1.0) < 1e-6
        assert abs(normalized.values[0] - 0.6) < 1e-6
        assert abs(normalized.values[1] - 0.8) < 1e-6

    def test_normalize_zero_vector(self):
        """Test normalizing zero vector raises error."""
        emb = Embedding([0.0, 0.0, 0.0])
        with pytest.raises(ValueError, match="zero vector"):
            emb.normalize()

    def test_dot_product(self):
        """Test dot product calculation."""
        emb1 = Embedding([1.0, 2.0, 3.0])
        emb2 = Embedding([4.0, 5.0, 6.0])

        result = emb1.dot(emb2)
        assert result == 32.0  # 1*4 + 2*5 + 3*6

    def test_dot_product_dimension_mismatch(self):
        """Test dot product with mismatched dimensions."""
        emb1 = Embedding([1.0, 2.0])
        emb2 = Embedding([1.0, 2.0, 3.0])

        with pytest.raises(ValueError, match="dimensions must match"):
            emb1.dot(emb2)

    def test_cosine_similarity(self):
        """Test cosine similarity calculation."""
        emb1 = Embedding([1.0, 0.0])
        emb2 = Embedding([1.0, 0.0])

        # Same direction = 1.0
        assert emb1.cosine_similarity(emb2) == 1.0

        emb3 = Embedding([0.0, 1.0])
        # Orthogonal = 0.0
        assert emb1.cosine_similarity(emb3) == 0.0

    def test_euclidean_distance(self):
        """Test Euclidean distance calculation."""
        emb1 = Embedding([0.0, 0.0])
        emb2 = Embedding([3.0, 4.0])

        assert emb1.euclidean_distance(emb2) == 5.0

    def test_to_list(self):
        """Test conversion to list."""
        emb = Embedding([0.1, 0.2, 0.3])
        lst = emb.to_list()

        assert lst == [0.1, 0.2, 0.3]
        assert isinstance(lst, list)

    def test_from_list(self):
        """Test creation from list."""
        emb = Embedding.from_list([0.1, 0.2, 0.3])
        assert emb.values == (0.1, 0.2, 0.3)

    def test_len_and_getitem(self):
        """Test length and indexing."""
        emb = Embedding([0.1, 0.2, 0.3])

        assert len(emb) == 3
        assert emb[0] == 0.1
        assert emb[2] == 0.3


class TestSimilarityScore:
    """Tests for SimilarityScore value object."""

    def test_create_score(self):
        """Test basic score creation."""
        score = SimilarityScore(value=0.85)

        assert score.value == 0.85
        assert score.score_type == ScoreType.COSINE

    def test_score_with_type(self):
        """Test score with specific type."""
        score = SimilarityScore(value=0.5, score_type=ScoreType.EUCLIDEAN)
        assert score.score_type == ScoreType.EUCLIDEAN

    def test_cosine_score_bounds(self):
        """Test cosine score bounds validation."""
        # Valid range
        SimilarityScore(value=-1.0)
        SimilarityScore(value=1.0)

        # Invalid range
        with pytest.raises(ValueError, match="between -1 and 1"):
            SimilarityScore(value=1.5)

    def test_euclidean_score_bounds(self):
        """Test Euclidean score bounds validation."""
        # Valid (non-negative)
        SimilarityScore(value=0.0, score_type=ScoreType.EUCLIDEAN)
        SimilarityScore(value=10.0, score_type=ScoreType.EUCLIDEAN)

        # Invalid (negative)
        with pytest.raises(ValueError, match="cannot be negative"):
            SimilarityScore(value=-0.5, score_type=ScoreType.EUCLIDEAN)

    def test_is_high_relevance(self):
        """Test high relevance detection."""
        high = SimilarityScore(value=0.9)
        low = SimilarityScore(value=0.5)

        assert high.is_high_relevance
        assert not low.is_high_relevance

    def test_is_low_relevance(self):
        """Test low relevance detection."""
        high = SimilarityScore(value=0.9)
        low = SimilarityScore(value=0.3)

        assert not high.is_low_relevance
        assert low.is_low_relevance

    def test_to_percentage(self):
        """Test conversion to percentage."""
        score = SimilarityScore(value=0.8)
        # Maps [−1, 1] to [0, 100], so 0.8 -> 90%
        assert score.to_percentage() == 90.0

    def test_comparison(self):
        """Test score comparison."""
        low = SimilarityScore(value=0.5)
        high = SimilarityScore(value=0.9)

        assert low < high
        assert high > low
        assert low <= high
        assert high >= low

    def test_comparison_different_types(self):
        """Test comparison of different score types raises error."""
        cosine = SimilarityScore(value=0.5, score_type=ScoreType.COSINE)
        euclidean = SimilarityScore(value=0.5, score_type=ScoreType.EUCLIDEAN)

        with pytest.raises(ValueError, match="different types"):
            _ = cosine < euclidean


class TestRelevanceScore:
    """Tests for RelevanceScore value object."""

    def test_create_relevance_score(self):
        """Test basic relevance score creation."""
        score = RelevanceScore(value=0.75)
        assert score.value == 0.75

    def test_relevance_score_bounds(self):
        """Test relevance score bounds validation."""
        # Valid range
        RelevanceScore(value=0.0)
        RelevanceScore(value=1.0)

        # Invalid range
        with pytest.raises(ValueError, match="between 0 and 1"):
            RelevanceScore(value=-0.1)
        with pytest.raises(ValueError, match="between 0 and 1"):
            RelevanceScore(value=1.1)

    def test_from_similarity(self):
        """Test conversion from similarity score."""
        sim = SimilarityScore(value=0.5, score_type=ScoreType.COSINE)
        rel = RelevanceScore.from_similarity(sim)

        # 0.5 cosine -> (0.5 + 1) / 2 = 0.75
        assert rel.value == 0.75

    def test_is_relevant(self):
        """Test relevance detection."""
        high = RelevanceScore(value=0.8)
        low = RelevanceScore(value=0.3)

        assert high.is_relevant
        assert not low.is_relevant
