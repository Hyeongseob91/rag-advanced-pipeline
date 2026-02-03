"""Embedding Value Object for RAG pipeline."""

from dataclasses import dataclass
from typing import Sequence
import math


@dataclass(frozen=True)
class Embedding:
    """An immutable embedding vector."""

    values: tuple[float, ...]

    def __init__(self, values: Sequence[float]) -> None:
        if not values:
            raise ValueError("Embedding values cannot be empty")
        # Use object.__setattr__ since this is a frozen dataclass
        object.__setattr__(self, "values", tuple(values))

    @property
    def dimension(self) -> int:
        """Return the dimension of the embedding."""
        return len(self.values)

    @property
    def norm(self) -> float:
        """Return the L2 norm of the embedding."""
        return math.sqrt(sum(v * v for v in self.values))

    def normalize(self) -> "Embedding":
        """Return a normalized (unit) embedding."""
        n = self.norm
        if n == 0:
            raise ValueError("Cannot normalize zero vector")
        return Embedding([v / n for v in self.values])

    def dot(self, other: "Embedding") -> float:
        """Compute dot product with another embedding."""
        if self.dimension != other.dimension:
            raise ValueError(
                f"Embedding dimensions must match: {self.dimension} vs {other.dimension}"
            )
        return sum(a * b for a, b in zip(self.values, other.values, strict=True))

    def cosine_similarity(self, other: "Embedding") -> float:
        """Compute cosine similarity with another embedding."""
        if self.dimension != other.dimension:
            raise ValueError(
                f"Embedding dimensions must match: {self.dimension} vs {other.dimension}"
            )
        dot_product = self.dot(other)
        norm_product = self.norm * other.norm
        if norm_product == 0:
            return 0.0
        return dot_product / norm_product

    def euclidean_distance(self, other: "Embedding") -> float:
        """Compute Euclidean distance to another embedding."""
        if self.dimension != other.dimension:
            raise ValueError(
                f"Embedding dimensions must match: {self.dimension} vs {other.dimension}"
            )
        return math.sqrt(
            sum((a - b) ** 2 for a, b in zip(self.values, other.values, strict=True))
        )

    def to_list(self) -> list[float]:
        """Convert to a mutable list."""
        return list(self.values)

    @classmethod
    def from_list(cls, values: list[float]) -> "Embedding":
        """Create an Embedding from a list."""
        return cls(values)

    def __len__(self) -> int:
        return len(self.values)

    def __getitem__(self, index: int) -> float:
        return self.values[index]
