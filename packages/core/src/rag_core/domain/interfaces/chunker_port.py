"""Chunker Port - Abstract interface for text chunking adapters."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

from rag_core.domain.entities.document import Chunk, Document


@dataclass
class ChunkingConfig:
    """Configuration for chunking behavior."""

    chunk_size: int = 512
    chunk_overlap: int = 50
    min_chunk_size: int = 100
    max_chunk_size: int = 2000
    separator: str = "\n\n"
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.chunk_size <= 0:
            raise ValueError("chunk_size must be positive")
        if self.chunk_overlap < 0:
            raise ValueError("chunk_overlap cannot be negative")
        if self.chunk_overlap >= self.chunk_size:
            raise ValueError("chunk_overlap must be less than chunk_size")
        if self.min_chunk_size <= 0:
            raise ValueError("min_chunk_size must be positive")
        if self.max_chunk_size < self.chunk_size:
            raise ValueError("max_chunk_size must be >= chunk_size")


@dataclass
class ChunkingResult:
    """Result from a chunking operation."""

    chunks: list[Chunk]
    total_characters: int = 0
    chunking_strategy: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def chunk_count(self) -> int:
        """Number of chunks created."""
        return len(self.chunks)

    @property
    def average_chunk_size(self) -> float:
        """Average character count per chunk."""
        if not self.chunks:
            return 0.0
        return sum(len(c.content) for c in self.chunks) / len(self.chunks)


class ChunkerPort(ABC):
    """Abstract interface for text chunking adapters.

    This port defines the contract that all chunking implementations must follow.
    Implementations include SemanticChunker, FixedSizeChunker, etc.
    """

    @abstractmethod
    async def chunk(
        self,
        document: Document,
        config: ChunkingConfig | None = None,
    ) -> ChunkingResult:
        """Chunk a document into smaller pieces.

        Args:
            document: The document to chunk.
            config: Optional chunking configuration.

        Returns:
            ChunkingResult containing the chunks and metadata.
        """
        pass

    @abstractmethod
    async def chunk_text(
        self,
        text: str,
        document_id: str | None = None,
        config: ChunkingConfig | None = None,
    ) -> list[Chunk]:
        """Chunk raw text into smaller pieces.

        Args:
            text: The text to chunk.
            document_id: Optional document ID to associate with chunks.
            config: Optional chunking configuration.

        Returns:
            List of Chunk objects.
        """
        pass

    @property
    @abstractmethod
    def strategy_name(self) -> str:
        """Return the name of the chunking strategy."""
        pass

    @property
    def default_config(self) -> ChunkingConfig:
        """Return the default chunking configuration."""
        return ChunkingConfig()
