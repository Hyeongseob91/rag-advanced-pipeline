"""Query DTOs for the application layer."""

from dataclasses import dataclass, field
from typing import Any
from uuid import UUID


@dataclass
class QueryRequest:
    """Request DTO for a RAG query."""

    query: str
    top_k: int = 5
    score_threshold: float | None = None
    rewrite_query: bool = True
    collection_name: str | None = None
    filter_metadata: dict[str, Any] | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.query or not self.query.strip():
            raise ValueError("Query cannot be empty")
        if self.top_k <= 0:
            raise ValueError("top_k must be positive")
        if self.score_threshold is not None and not 0 <= self.score_threshold <= 1:
            raise ValueError("score_threshold must be between 0 and 1")


@dataclass
class QueryResponse:
    """Response DTO for a RAG query."""

    query_id: UUID
    original_query: str
    rewritten_query: str | None
    answer: str
    sources: list["SourceDTO"]
    model: str
    total_tokens: int = 0
    retrieval_time_ms: float = 0.0
    generation_time_ms: float = 0.0
    total_time_ms: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def source_count(self) -> int:
        """Number of sources used in the response."""
        return len(self.sources)


@dataclass
class SourceDTO:
    """DTO for a source/citation in the response."""

    chunk_id: UUID
    document_id: UUID
    content: str
    score: float
    rank: int
    metadata: dict[str, Any] = field(default_factory=dict)
