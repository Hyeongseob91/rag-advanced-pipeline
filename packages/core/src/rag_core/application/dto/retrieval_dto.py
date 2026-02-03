"""Retrieval DTOs for the application layer."""

from dataclasses import dataclass, field
from typing import Any
from uuid import UUID


@dataclass
class RetrievalRequest:
    """Request DTO for document retrieval."""

    query: str
    query_embedding: list[float] | None = None
    top_k: int = 10
    score_threshold: float | None = None
    collection_name: str | None = None
    filter_metadata: dict[str, Any] | None = None
    include_embeddings: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.query or not self.query.strip():
            raise ValueError("Query cannot be empty")
        if self.top_k <= 0:
            raise ValueError("top_k must be positive")


@dataclass
class RetrievedChunkDTO:
    """DTO for a retrieved chunk."""

    chunk_id: UUID
    document_id: UUID
    content: str
    score: float
    rank: int
    embedding: list[float] | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class RetrievalResponse:
    """Response DTO for document retrieval."""

    query: str
    chunks: list[RetrievedChunkDTO]
    total_candidates: int = 0
    retrieval_time_ms: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def chunk_count(self) -> int:
        """Number of chunks retrieved."""
        return len(self.chunks)

    def get_context(self, separator: str = "\n\n") -> str:
        """Combine all retrieved chunks into a context string."""
        return separator.join(chunk.content for chunk in self.chunks)

    def get_top_chunks(self, n: int) -> list[RetrievedChunkDTO]:
        """Get the top N chunks by rank."""
        return sorted(self.chunks, key=lambda c: c.rank)[:n]
