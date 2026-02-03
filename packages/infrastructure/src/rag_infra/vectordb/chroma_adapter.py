"""ChromaDB VectorDB Adapter."""

from typing import Any
from uuid import UUID, uuid4

import chromadb
from chromadb.config import Settings as ChromaSettings

from rag_core.domain.entities.document import Chunk
from rag_core.domain.interfaces.vectordb_port import (
    SearchResult,
    SearchResults,
    VectorDBPort,
)
from rag_core.domain.value_objects.score import ScoreType, SimilarityScore


class ChromaAdapter(VectorDBPort):
    """ChromaDB adapter implementing VectorDBPort.

    ChromaDB is an open-source embedding database that can run
    locally or as a server.
    """

    def __init__(
        self,
        host: str | None = None,
        port: int = 8000,
        persist_directory: str | None = None,
        default_collection: str = "documents",
    ):
        """Initialize the ChromaDB adapter.

        Args:
            host: Chroma server host (None for local/persistent mode).
            port: Chroma server port.
            persist_directory: Directory for persistent storage (None for in-memory).
            default_collection: Default collection name.
        """
        self._host = host
        self._port = port
        self._persist_directory = persist_directory
        self._default_collection = default_collection
        self._client: chromadb.ClientAPI | None = None

    def _get_client(self) -> chromadb.ClientAPI:
        """Get or create the ChromaDB client."""
        if self._client is None:
            if self._host:
                # Connect to Chroma server
                self._client = chromadb.HttpClient(
                    host=self._host,
                    port=self._port,
                )
            elif self._persist_directory:
                # Persistent local storage
                self._client = chromadb.PersistentClient(
                    path=self._persist_directory,
                    settings=ChromaSettings(anonymized_telemetry=False),
                )
            else:
                # In-memory client
                self._client = chromadb.Client(
                    settings=ChromaSettings(anonymized_telemetry=False),
                )
        return self._client

    def _get_collection_name(self, name: str | None) -> str:
        """Get the effective collection name."""
        return name or self._default_collection

    async def upsert(
        self,
        chunks: list[Chunk],
        collection_name: str | None = None,
    ) -> list[UUID]:
        """Insert or update chunks in ChromaDB.

        Args:
            chunks: List of chunks with embeddings.
            collection_name: Optional collection name.

        Returns:
            List of chunk UUIDs.
        """
        client = self._get_client()
        coll_name = self._get_collection_name(collection_name)
        collection = client.get_or_create_collection(name=coll_name)

        ids: list[str] = []
        embeddings: list[list[float]] = []
        documents: list[str] = []
        metadatas: list[dict[str, Any]] = []

        for chunk in chunks:
            if chunk.embedding is None:
                raise ValueError(f"Chunk {chunk.id} has no embedding")

            ids.append(str(chunk.id))
            embeddings.append(chunk.embedding)
            documents.append(chunk.content)
            metadatas.append({
                "document_id": str(chunk.document_id),
                "chunk_index": chunk.chunk_index,
                **{k: str(v) for k, v in chunk.metadata.items()},
            })

        collection.upsert(
            ids=ids,
            embeddings=embeddings,
            documents=documents,
            metadatas=metadatas,
        )

        return [chunk.id for chunk in chunks]

    async def search(
        self,
        query_embedding: list[float],
        top_k: int = 10,
        collection_name: str | None = None,
        filter_metadata: dict[str, Any] | None = None,
        score_threshold: float | None = None,
    ) -> SearchResults:
        """Search for similar chunks in ChromaDB.

        Args:
            query_embedding: Query embedding vector.
            top_k: Number of results.
            collection_name: Optional collection name.
            filter_metadata: Optional filters.
            score_threshold: Optional minimum score.

        Returns:
            SearchResults with matching chunks.
        """
        import time

        start_time = time.perf_counter()

        client = self._get_client()
        coll_name = self._get_collection_name(collection_name)
        collection = client.get_collection(name=coll_name)

        # Build query
        where_filter = None
        if filter_metadata:
            where_filter = {
                "$and": [{k: {"$eq": str(v)}} for k, v in filter_metadata.items()]
            }

        query_result = collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            where=where_filter,
            include=["documents", "metadatas", "distances"],
        )

        results: list[SearchResult] = []

        if query_result["ids"] and query_result["ids"][0]:
            ids = query_result["ids"][0]
            documents = query_result["documents"][0] if query_result["documents"] else []
            metadatas = query_result["metadatas"][0] if query_result["metadatas"] else []
            distances = query_result["distances"][0] if query_result["distances"] else []

            for i, chunk_id in enumerate(ids):
                # ChromaDB returns L2 distance; convert to similarity
                distance = distances[i] if i < len(distances) else 0.0
                # Convert L2 distance to cosine-like similarity (approximate)
                similarity = 1.0 / (1.0 + distance)

                # Apply threshold if specified
                if score_threshold is not None and similarity < score_threshold:
                    continue

                metadata = metadatas[i] if i < len(metadatas) else {}
                document_id_str = metadata.pop("document_id", chunk_id)

                results.append(
                    SearchResult(
                        chunk_id=UUID(chunk_id),
                        content=documents[i] if i < len(documents) else "",
                        score=SimilarityScore(value=similarity, score_type=ScoreType.COSINE),
                        metadata=metadata,
                        document_id=UUID(document_id_str),
                    )
                )

        elapsed_ms = (time.perf_counter() - start_time) * 1000

        return SearchResults(
            results=results,
            total_count=len(results),
            query_time_ms=elapsed_ms,
        )

    async def delete(
        self,
        chunk_ids: list[UUID],
        collection_name: str | None = None,
    ) -> int:
        """Delete chunks by ID.

        Args:
            chunk_ids: List of chunk UUIDs.
            collection_name: Optional collection name.

        Returns:
            Number of deleted chunks.
        """
        client = self._get_client()
        coll_name = self._get_collection_name(collection_name)
        collection = client.get_collection(name=coll_name)

        ids_to_delete = [str(cid) for cid in chunk_ids]
        collection.delete(ids=ids_to_delete)

        return len(ids_to_delete)

    async def delete_by_document(
        self,
        document_id: UUID,
        collection_name: str | None = None,
    ) -> int:
        """Delete all chunks for a document.

        Args:
            document_id: Document UUID.
            collection_name: Optional collection name.

        Returns:
            Number of deleted chunks.
        """
        client = self._get_client()
        coll_name = self._get_collection_name(collection_name)
        collection = client.get_collection(name=coll_name)

        # Delete using filter
        collection.delete(where={"document_id": {"$eq": str(document_id)}})

        # ChromaDB doesn't return count, return 0 as placeholder
        return 0

    async def create_collection(
        self,
        collection_name: str,
        dimension: int,
        metadata: dict[str, Any] | None = None,
    ) -> bool:
        """Create a new collection in ChromaDB.

        Args:
            collection_name: Collection name.
            dimension: Embedding dimension (not used by ChromaDB).
            metadata: Optional metadata.

        Returns:
            True if created, False if exists.
        """
        client = self._get_client()

        # Check if exists
        existing = client.list_collections()
        if any(c.name == collection_name for c in existing):
            return False

        client.create_collection(
            name=collection_name,
            metadata=metadata,
        )
        return True

    async def delete_collection(self, collection_name: str) -> bool:
        """Delete a collection.

        Args:
            collection_name: Collection name.

        Returns:
            True if deleted, False if didn't exist.
        """
        client = self._get_client()

        try:
            client.delete_collection(name=collection_name)
            return True
        except Exception:
            return False

    async def collection_exists(self, collection_name: str) -> bool:
        """Check if a collection exists.

        Args:
            collection_name: Collection name.

        Returns:
            True if exists.
        """
        client = self._get_client()
        existing = client.list_collections()
        return any(c.name == collection_name for c in existing)

    async def get_collection_count(self, collection_name: str | None = None) -> int:
        """Get the number of items in a collection.

        Args:
            collection_name: Optional collection name.

        Returns:
            Number of items.
        """
        client = self._get_client()
        coll_name = self._get_collection_name(collection_name)
        collection = client.get_collection(name=coll_name)
        return collection.count()
