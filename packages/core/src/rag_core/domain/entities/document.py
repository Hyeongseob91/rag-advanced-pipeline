"""Document and Chunk entities for RAG pipeline."""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any
from uuid import UUID, uuid4


@dataclass
class Chunk:
    """A chunk of text from a document with optional embedding."""

    content: str
    document_id: UUID
    chunk_index: int
    metadata: dict[str, Any] = field(default_factory=dict)
    id: UUID = field(default_factory=uuid4)
    embedding: list[float] | None = None
    start_char: int | None = None
    end_char: int | None = None

    def __post_init__(self) -> None:
        if not self.content:
            raise ValueError("Chunk content cannot be empty")
        if self.chunk_index < 0:
            raise ValueError("Chunk index must be non-negative")

    @property
    def token_count(self) -> int:
        """Approximate token count (rough estimate: 4 chars per token)."""
        return len(self.content) // 4

    def with_embedding(self, embedding: list[float]) -> "Chunk":
        """Return a new chunk with the given embedding."""
        return Chunk(
            content=self.content,
            document_id=self.document_id,
            chunk_index=self.chunk_index,
            metadata=self.metadata.copy(),
            id=self.id,
            embedding=embedding,
            start_char=self.start_char,
            end_char=self.end_char,
        )


@dataclass
class Document:
    """A document that can be chunked and indexed for retrieval."""

    content: str
    source: str
    metadata: dict[str, Any] = field(default_factory=dict)
    id: UUID = field(default_factory=uuid4)
    created_at: datetime = field(default_factory=datetime.now)
    chunks: list[Chunk] = field(default_factory=list)

    def __post_init__(self) -> None:
        if not self.content:
            raise ValueError("Document content cannot be empty")
        if not self.source:
            raise ValueError("Document source cannot be empty")

    @property
    def chunk_count(self) -> int:
        """Number of chunks in this document."""
        return len(self.chunks)

    @property
    def char_count(self) -> int:
        """Total character count of the document."""
        return len(self.content)

    def add_chunks(self, chunks: list[Chunk]) -> "Document":
        """Return a new document with the given chunks added."""
        return Document(
            content=self.content,
            source=self.source,
            metadata=self.metadata.copy(),
            id=self.id,
            created_at=self.created_at,
            chunks=self.chunks + chunks,
        )
