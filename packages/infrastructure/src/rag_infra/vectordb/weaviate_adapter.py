"""Weaviate VectorDB Adapter."""

from typing import Any
from uuid import UUID

import weaviate
from weaviate.classes.config import Configure, DataType, Property
from weaviate.classes.query import MetadataQuery

from rag_core.domain.entities.document import Chunk
from rag_core.domain.interfaces.vectordb_port import (
    SearchResult,
    SearchResults,
    VectorDBPort,
)
from rag_core.domain.value_objects.score import ScoreType, SimilarityScore


class WeaviateAdapter(VectorDBPort):
    """Weaviate adapter implementing VectorDBPort.

    Weaviate is a vector database with semantic search capabilities
    and support for hybrid search (vector + keyword).
    """

    def __init__(
        self,
        host: str = "localhost",
        port: int = 8080,
        grpc_port: int = 50051,
        api_key: str | None = None,
        default_collection: str = "documents",
    ):
        """Initialize the Weaviate adapter.

        Args:
            host: Weaviate host.
            port: Weaviate HTTP port.
            grpc_port: Weaviate gRPC port.
            api_key: Optional API key.
            default_collection: Default collection name.
        """
        self._host = host
        self._port = port
        self._grpc_port = grpc_port
        self._api_key = api_key
        self._default_collection = default_collection
        self._client: weaviate.WeaviateClient | None = None

    def _get_client(self) -> weaviate.WeaviateClient:
        """Get or create the Weaviate client."""
        if self._client is None:
            if self._api_key:
                self._client = weaviate.connect_to_custom(
                    http_host=self._host,
                    http_port=self._port,
                    http_secure=False,
                    grpc_host=self._host,
                    grpc_port=self._grpc_port,
                    grpc_secure=False,
                    auth_credentials=weaviate.auth.AuthApiKey(self._api_key),
                )
            else:
                self._client = weaviate.connect_to_local(
                    host=self._host,
                    port=self._port,
                    grpc_port=self._grpc_port,
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
        """Insert or update chunks in Weaviate.

        Args:
            chunks: List of chunks with embeddings.
            collection_name: Optional collection name.

        Returns:
            List of chunk UUIDs.
        """
        client = self._get_client()
        coll_name = self._get_collection_name(collection_name)
        collection = client.collections.get(coll_name)

        uuids: list[UUID] = []
        with collection.batch.dynamic() as batch:
            for chunk in chunks:
                if chunk.embedding is None:
                    raise ValueError(f"Chunk {chunk.id} has no embedding")

                properties = {
                    "content": chunk.content,
                    "document_id": str(chunk.document_id),
                    "chunk_index": chunk.chunk_index,
                    **chunk.metadata,
                }

                batch.add_object(
                    uuid=chunk.id,
                    properties=properties,
                    vector=chunk.embedding,
                )
                uuids.append(chunk.id)

        return uuids

    async def search(
        self,
        query_embedding: list[float],
        top_k: int = 10,
        collection_name: str | None = None,
        filter_metadata: dict[str, Any] | None = None,
        score_threshold: float | None = None,
    ) -> SearchResults:
        """Search for similar chunks in Weaviate.

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
        collection = client.collections.get(coll_name)

        # Build query
        query = collection.query.near_vector(
            near_vector=query_embedding,
            limit=top_k,
            return_metadata=MetadataQuery(certainty=True, distance=True),
        )

        results: list[SearchResult] = []
        for obj in query.objects:
            # Weaviate returns certainty (0-1) or distance
            certainty = obj.metadata.certainty or 0.0

            # Apply threshold if specified
            if score_threshold is not None and certainty < score_threshold:
                continue

            results.append(
                SearchResult(
                    chunk_id=UUID(str(obj.uuid)),
                    content=obj.properties.get("content", ""),
                    score=SimilarityScore(value=certainty, score_type=ScoreType.COSINE),
                    metadata={
                        k: v
                        for k, v in obj.properties.items()
                        if k not in ("content", "document_id")
                    },
                    document_id=UUID(obj.properties.get("document_id", str(obj.uuid))),
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
        collection = client.collections.get(coll_name)

        deleted = 0
        for chunk_id in chunk_ids:
            try:
                collection.data.delete_by_id(chunk_id)
                deleted += 1
            except Exception:
                pass

        return deleted

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
        from weaviate.classes.query import Filter

        client = self._get_client()
        coll_name = self._get_collection_name(collection_name)
        collection = client.collections.get(coll_name)

        # Delete using filter
        result = collection.data.delete_many(
            where=Filter.by_property("document_id").equal(str(document_id))
        )

        return result.successful if result else 0

    async def create_collection(
        self,
        collection_name: str,
        dimension: int,
        metadata: dict[str, Any] | None = None,
    ) -> bool:
        """Create a new collection in Weaviate.

        Args:
            collection_name: Collection name.
            dimension: Embedding dimension.
            metadata: Optional metadata.

        Returns:
            True if created, False if exists.
        """
        client = self._get_client()

        if client.collections.exists(collection_name):
            return False

        client.collections.create(
            name=collection_name,
            properties=[
                Property(name="content", data_type=DataType.TEXT),
                Property(name="document_id", data_type=DataType.TEXT),
                Property(name="chunk_index", data_type=DataType.INT),
            ],
            vectorizer_config=Configure.Vectorizer.none(),
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

        if not client.collections.exists(collection_name):
            return False

        client.collections.delete(collection_name)
        return True

    async def collection_exists(self, collection_name: str) -> bool:
        """Check if a collection exists.

        Args:
            collection_name: Collection name.

        Returns:
            True if exists.
        """
        client = self._get_client()
        return client.collections.exists(collection_name)

    async def get_collection_count(self, collection_name: str | None = None) -> int:
        """Get the number of items in a collection.

        Args:
            collection_name: Optional collection name.

        Returns:
            Number of items.
        """
        client = self._get_client()
        coll_name = self._get_collection_name(collection_name)
        collection = client.collections.get(coll_name)
        result = collection.aggregate.over_all(total_count=True)
        return result.total_count or 0

    def close(self) -> None:
        """Close the Weaviate client."""
        if self._client:
            self._client.close()
            self._client = None
