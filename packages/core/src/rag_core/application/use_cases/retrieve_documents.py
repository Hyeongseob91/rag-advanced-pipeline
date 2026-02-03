"""Retrieve Documents Use Case."""

from dataclasses import dataclass
from time import perf_counter
from typing import Any
from uuid import UUID

from rag_core.domain.entities.document import Chunk
from rag_core.domain.entities.query import Query
from rag_core.domain.entities.response import RetrievalResult, RetrievalResultSet
from rag_core.domain.interfaces.embedding_port import EmbeddingPort
from rag_core.domain.interfaces.vectordb_port import VectorDBPort
from rag_core.domain.value_objects.score import ScoreType, SimilarityScore


@dataclass
class RetrieveDocumentsUseCase:
    """Use case for retrieving relevant documents for a query.

    This use case embeds the query and searches the vector database
    for similar document chunks.
    """

    embedding_port: EmbeddingPort
    vectordb_port: VectorDBPort
    default_top_k: int = 5
    default_collection: str = "documents"

    async def execute(
        self,
        query: Query,
        top_k: int | None = None,
        collection_name: str | None = None,
        filter_metadata: dict[str, Any] | None = None,
        score_threshold: float | None = None,
    ) -> RetrievalResultSet:
        """Execute the document retrieval use case.

        Args:
            query: The query to search for.
            top_k: Number of results to return.
            collection_name: Collection to search in.
            filter_metadata: Optional metadata filters.
            score_threshold: Minimum score threshold.

        Returns:
            RetrievalResultSet with matching chunks and scores.
        """
        start_time = perf_counter()

        # Embed the query
        query_embedding = await self.embedding_port.embed(query.cleaned_text)

        # Search the vector database
        search_results = await self.vectordb_port.search(
            query_embedding=query_embedding,
            top_k=top_k or self.default_top_k,
            collection_name=collection_name or self.default_collection,
            filter_metadata=filter_metadata,
            score_threshold=score_threshold,
        )

        # Convert to RetrievalResults
        results: list[RetrievalResult] = []
        for rank, search_result in enumerate(search_results.results):
            chunk = Chunk(
                content=search_result.content,
                document_id=search_result.document_id or search_result.chunk_id,
                chunk_index=0,
                metadata=search_result.metadata,
                id=search_result.chunk_id,
                embedding=search_result.embedding,
            )

            results.append(
                RetrievalResult(
                    chunk=chunk,
                    score=search_result.score,
                    rank=rank + 1,
                    metadata=search_result.metadata,
                )
            )

        elapsed_ms = (perf_counter() - start_time) * 1000

        return RetrievalResultSet(
            query=query,
            results=results,
            total_candidates=search_results.total_count,
            retrieval_time_ms=elapsed_ms,
            metadata={
                "collection": collection_name or self.default_collection,
                "embedding_model": self.embedding_port.model_name,
            },
        )

    async def execute_with_embedding(
        self,
        query_embedding: list[float],
        query: Query,
        top_k: int | None = None,
        collection_name: str | None = None,
        filter_metadata: dict[str, Any] | None = None,
        score_threshold: float | None = None,
    ) -> RetrievalResultSet:
        """Execute retrieval with a pre-computed embedding.

        Useful for HyDE or when the embedding is already available.

        Args:
            query_embedding: Pre-computed query embedding.
            query: The original query (for metadata).
            top_k: Number of results to return.
            collection_name: Collection to search in.
            filter_metadata: Optional metadata filters.
            score_threshold: Minimum score threshold.

        Returns:
            RetrievalResultSet with matching chunks and scores.
        """
        start_time = perf_counter()

        # Search the vector database
        search_results = await self.vectordb_port.search(
            query_embedding=query_embedding,
            top_k=top_k or self.default_top_k,
            collection_name=collection_name or self.default_collection,
            filter_metadata=filter_metadata,
            score_threshold=score_threshold,
        )

        # Convert to RetrievalResults
        results: list[RetrievalResult] = []
        for rank, search_result in enumerate(search_results.results):
            chunk = Chunk(
                content=search_result.content,
                document_id=search_result.document_id or search_result.chunk_id,
                chunk_index=0,
                metadata=search_result.metadata,
                id=search_result.chunk_id,
                embedding=search_result.embedding,
            )

            results.append(
                RetrievalResult(
                    chunk=chunk,
                    score=search_result.score,
                    rank=rank + 1,
                    metadata=search_result.metadata,
                )
            )

        elapsed_ms = (perf_counter() - start_time) * 1000

        return RetrievalResultSet(
            query=query,
            results=results,
            total_candidates=search_results.total_count,
            retrieval_time_ms=elapsed_ms,
            metadata={
                "collection": collection_name or self.default_collection,
                "embedding_model": self.embedding_port.model_name,
                "precomputed_embedding": True,
            },
        )

    async def execute_hybrid(
        self,
        query: Query,
        top_k: int | None = None,
        collection_name: str | None = None,
        filter_metadata: dict[str, Any] | None = None,
        alpha: float = 0.5,
    ) -> RetrievalResultSet:
        """Execute hybrid search combining dense and sparse retrieval.

        Args:
            query: The query to search for.
            top_k: Number of results to return.
            collection_name: Collection to search in.
            filter_metadata: Optional metadata filters.
            alpha: Weight for dense vs sparse (1.0 = all dense, 0.0 = all sparse).

        Returns:
            RetrievalResultSet with merged results.
        """
        # For now, fall back to dense retrieval
        # Hybrid search would require BM25 or similar sparse retrieval
        return await self.execute(
            query=query,
            top_k=top_k,
            collection_name=collection_name,
            filter_metadata=filter_metadata,
        )
