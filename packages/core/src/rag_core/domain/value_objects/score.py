"""Score Value Objects for RAG pipeline."""

from dataclasses import dataclass
from enum import Enum


class ScoreType(Enum):
    """Type of similarity score."""

    COSINE = "cosine"
    DOT_PRODUCT = "dot_product"
    EUCLIDEAN = "euclidean"
    BM25 = "bm25"
    HYBRID = "hybrid"


@dataclass(frozen=True)
class SimilarityScore:
    """An immutable similarity score value object."""

    value: float
    score_type: ScoreType = ScoreType.COSINE

    def __post_init__(self) -> None:
        if self.score_type in (ScoreType.COSINE, ScoreType.DOT_PRODUCT):
            if not -1.0 <= self.value <= 1.0:
                # Allow slightly out of range due to floating point
                if not -1.01 <= self.value <= 1.01:
                    raise ValueError(
                        f"Cosine/dot product score must be between -1 and 1, got {self.value}"
                    )
        elif self.score_type == ScoreType.EUCLIDEAN:
            if self.value < 0:
                raise ValueError(f"Euclidean distance cannot be negative, got {self.value}")

    @property
    def is_high_relevance(self) -> bool:
        """Check if this is a high relevance score (> 0.8 for cosine)."""
        if self.score_type == ScoreType.COSINE:
            return self.value > 0.8
        elif self.score_type == ScoreType.EUCLIDEAN:
            return self.value < 0.5  # Lower is better for distance
        return self.value > 0.8

    @property
    def is_low_relevance(self) -> bool:
        """Check if this is a low relevance score (< 0.5 for cosine)."""
        if self.score_type == ScoreType.COSINE:
            return self.value < 0.5
        elif self.score_type == ScoreType.EUCLIDEAN:
            return self.value > 1.0  # Higher is worse for distance
        return self.value < 0.5

    def to_percentage(self) -> float:
        """Convert score to a 0-100 percentage."""
        if self.score_type == ScoreType.COSINE:
            # Map [-1, 1] to [0, 100]
            return (self.value + 1) * 50
        elif self.score_type == ScoreType.EUCLIDEAN:
            # Map distance to percentage (closer = higher)
            return max(0, (1 - self.value) * 100)
        return self.value * 100

    def __lt__(self, other: "SimilarityScore") -> bool:
        if self.score_type != other.score_type:
            raise ValueError("Cannot compare scores of different types")
        if self.score_type == ScoreType.EUCLIDEAN:
            # For distance, lower is better, so reverse comparison
            return self.value > other.value
        return self.value < other.value

    def __le__(self, other: "SimilarityScore") -> bool:
        return self == other or self < other

    def __gt__(self, other: "SimilarityScore") -> bool:
        if self.score_type != other.score_type:
            raise ValueError("Cannot compare scores of different types")
        if self.score_type == ScoreType.EUCLIDEAN:
            return self.value < other.value
        return self.value > other.value

    def __ge__(self, other: "SimilarityScore") -> bool:
        return self == other or self > other


@dataclass(frozen=True)
class RelevanceScore:
    """A normalized relevance score between 0 and 1."""

    value: float

    def __post_init__(self) -> None:
        if not 0.0 <= self.value <= 1.0:
            raise ValueError(f"Relevance score must be between 0 and 1, got {self.value}")

    @classmethod
    def from_similarity(cls, score: SimilarityScore) -> "RelevanceScore":
        """Convert a similarity score to a relevance score."""
        if score.score_type == ScoreType.COSINE:
            # Map [-1, 1] to [0, 1]
            return cls((score.value + 1) / 2)
        elif score.score_type == ScoreType.EUCLIDEAN:
            # Map distance to relevance (closer = higher relevance)
            return cls(1 / (1 + score.value))
        return cls(max(0, min(1, score.value)))

    @property
    def is_relevant(self) -> bool:
        """Check if the score indicates relevance (> 0.5)."""
        return self.value > 0.5

    def __lt__(self, other: "RelevanceScore") -> bool:
        return self.value < other.value

    def __le__(self, other: "RelevanceScore") -> bool:
        return self.value <= other.value

    def __gt__(self, other: "RelevanceScore") -> bool:
        return self.value > other.value

    def __ge__(self, other: "RelevanceScore") -> bool:
        return self.value >= other.value
