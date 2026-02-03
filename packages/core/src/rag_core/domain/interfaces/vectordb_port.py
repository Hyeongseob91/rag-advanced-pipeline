"""VectorDB Port - Abstract interface for vector database adapters."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any
from uuid import UUID

from rag_core.domain.entities.document import Chunk
from rag_core.domain.value_objects.score import SimilarityScore


@dataclass
class SearchResult:
    """A single search result from the vector database."""

    chunk_id: UUID
    content: str
    score: SimilarityScore
    metadata: dict[str, Any] = field(default_factory=dict)
    document_id: UUID | None = None
    embedding: list[float] | None = None


@dataclass
class SearchResults:
    """Results from a vector search query."""

    results: list[SearchResult]
    total_count: int = 0
    query_time_ms: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def count(self) -> int:
        """Number of results returned."""
        return len(self.results)

    def to_chunks(self) -> list[tuple[Chunk, SimilarityScore]]:
        """Convert search results to Chunk objects with scores."""
        return [
            (
                Chunk(
                    content=r.content,
                    document_id=r.document_id or r.chunk_id,
                    chunk_index=0,
                    metadata=r.metadata,
                    id=r.chunk_id,
                    embedding=r.embedding,
                ),
                r.score,
            )
            for r in self.results
        ]


class VectorDBPort(ABC):
    """Abstract interface for vector database adapters.

    This port defines the contract that all vector database implementations must follow.
    Implementations include Weaviate, ChromaDB, etc.
    """

    @abstractmethod
    async def upsert(
        self,
        chunks: list[Chunk],
        collection_name: str | None = None,
    ) -> list[UUID]:
        """Insert or update chunks in the vector database.

        Args:
            chunks: List of Chunk objects to upsert (must have embeddings).
            collection_name: Optional collection name (uses default if not specified).

        Returns:
            List of UUIDs for the upserted chunks.
        """
        pass

    @abstractmethod
    async def search(
        self,
        query_embedding: list[float],
        top_k: int = 10,
        collection_name: str | None = None,
        filter_metadata: dict[str, Any] | None = None,
        score_threshold: float | None = None,
    ) -> SearchResults:
        """Search for similar chunks by embedding.

        Args:
            query_embedding: The query embedding vector.
            top_k: Number of results to return.
            collection_name: Optional collection name (uses default if not specified).
            filter_metadata: Optional metadata filters.
            score_threshold: Optional minimum score threshold.

        Returns:
            SearchResults containing matching chunks and scores.
        """
        pass

    @abstractmethod
    async def delete(
        self,
        chunk_ids: list[UUID],
        collection_name: str | None = None,
    ) -> int:
        """Delete chunks by their IDs.

        Args:
            chunk_ids: List of chunk UUIDs to delete.
            collection_name: Optional collection name (uses default if not specified).

        Returns:
            Number of chunks deleted.
        """
        pass

    @abstractmethod
    async def delete_by_document(
        self,
        document_id: UUID,
        collection_name: str | None = None,
    ) -> int:
        """Delete all chunks belonging to a document.

        Args:
            document_id: The document UUID whose chunks should be deleted.
            collection_name: Optional collection name (uses default if not specified).

        Returns:
            Number of chunks deleted.
        """
        pass

    @abstractmethod
    async def create_collection(
        self,
        collection_name: str,
        dimension: int,
        metadata: dict[str, Any] | None = None,
    ) -> bool:
        """Create a new collection in the vector database.

        Args:
            collection_name: Name of the collection to create.
            dimension: Embedding dimension for the collection.
            metadata: Optional collection metadata.

        Returns:
            True if collection was created, False if it already exists.
        """
        pass

    @abstractmethod
    async def delete_collection(self, collection_name: str) -> bool:
        """Delete a collection from the vector database.

        Args:
            collection_name: Name of the collection to delete.

        Returns:
            True if collection was deleted, False if it didn't exist.
        """
        pass

    @abstractmethod
    async def collection_exists(self, collection_name: str) -> bool:
        """Check if a collection exists.

        Args:
            collection_name: Name of the collection to check.

        Returns:
            True if the collection exists, False otherwise.
        """
        pass

    @abstractmethod
    async def get_collection_count(self, collection_name: str | None = None) -> int:
        """Get the number of items in a collection.

        Args:
            collection_name: Optional collection name (uses default if not specified).

        Returns:
            Number of items in the collection.
        """
        pass

    async def health_check(self) -> bool:
        """Check if the vector database is healthy.

        Returns:
            True if the service is available, False otherwise.
        """
        try:
            # Try to check if default collection exists
            await self.collection_exists("health_check")
            return True
        except Exception:
            return False
