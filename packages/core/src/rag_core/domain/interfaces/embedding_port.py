"""Embedding Port - Abstract interface for embedding adapters."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any


@dataclass
class EmbeddingResult:
    """Result from an embedding call."""

    embeddings: list[list[float]]
    model: str = ""
    total_tokens: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def dimension(self) -> int:
        """Return the embedding dimension."""
        if not self.embeddings:
            return 0
        return len(self.embeddings[0])

    @property
    def count(self) -> int:
        """Return the number of embeddings."""
        return len(self.embeddings)


class EmbeddingPort(ABC):
    """Abstract interface for embedding adapters.

    This port defines the contract that all embedding implementations must follow.
    Implementations include OpenAI, Infinity, SentenceTransformers, etc.
    """

    @abstractmethod
    async def embed(self, text: str) -> list[float]:
        """Generate an embedding for a single text.

        Args:
            text: The text to embed.

        Returns:
            A list of floats representing the embedding vector.
        """
        pass

    @abstractmethod
    async def embed_batch(self, texts: list[str]) -> EmbeddingResult:
        """Generate embeddings for multiple texts.

        Args:
            texts: List of texts to embed.

        Returns:
            EmbeddingResult containing all embeddings and metadata.
        """
        pass

    @property
    @abstractmethod
    def model_name(self) -> str:
        """Return the model name/identifier."""
        pass

    @property
    @abstractmethod
    def dimension(self) -> int:
        """Return the embedding dimension for this model."""
        pass

    async def health_check(self) -> bool:
        """Check if the embedding service is healthy.

        Returns:
            True if the service is available, False otherwise.
        """
        try:
            embedding = await self.embed("test")
            return len(embedding) == self.dimension
        except Exception:
            return False
