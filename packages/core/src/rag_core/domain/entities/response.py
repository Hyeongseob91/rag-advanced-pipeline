"""Response entities for RAG pipeline."""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any
from uuid import UUID, uuid4

from rag_core.domain.entities.document import Chunk
from rag_core.domain.entities.query import Query
from rag_core.domain.value_objects.score import SimilarityScore


@dataclass
class RetrievalResult:
    """A single retrieval result with chunk and relevance score."""

    chunk: Chunk
    score: SimilarityScore
    rank: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.rank < 0:
            raise ValueError("Rank must be non-negative")

    @property
    def content(self) -> str:
        """Return the chunk content."""
        return self.chunk.content

    @property
    def document_id(self) -> UUID:
        """Return the source document ID."""
        return self.chunk.document_id


@dataclass
class RetrievalResultSet:
    """A set of retrieval results for a query."""

    query: Query
    results: list[RetrievalResult]
    total_candidates: int = 0
    retrieval_time_ms: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def top_k(self) -> int:
        """Number of results returned."""
        return len(self.results)

    def get_context(self, separator: str = "\n\n") -> str:
        """Combine all retrieved chunks into a single context string."""
        return separator.join(result.content for result in self.results)

    def filter_by_threshold(self, threshold: float) -> "RetrievalResultSet":
        """Return a new result set with only results above the threshold."""
        filtered = [r for r in self.results if r.score.value >= threshold]
        return RetrievalResultSet(
            query=self.query,
            results=filtered,
            total_candidates=self.total_candidates,
            retrieval_time_ms=self.retrieval_time_ms,
            metadata=self.metadata.copy(),
        )


@dataclass
class GeneratedResponse:
    """A generated response from the LLM."""

    query: Query
    answer: str
    sources: list[RetrievalResult] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
    id: UUID = field(default_factory=uuid4)
    created_at: datetime = field(default_factory=datetime.now)
    model: str = ""
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    generation_time_ms: float = 0.0

    def __post_init__(self) -> None:
        if not self.answer:
            raise ValueError("Generated answer cannot be empty")
        if self.total_tokens == 0 and (self.prompt_tokens > 0 or self.completion_tokens > 0):
            self.total_tokens = self.prompt_tokens + self.completion_tokens

    @property
    def source_count(self) -> int:
        """Number of source chunks used for generation."""
        return len(self.sources)

    @property
    def has_sources(self) -> bool:
        """Check if the response has source citations."""
        return len(self.sources) > 0
