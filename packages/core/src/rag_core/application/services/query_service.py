"""Query Service - Simplified query interface."""

from dataclasses import dataclass
from typing import Any

from rag_core.application.dto.retrieval_dto import (
    RetrievalRequest,
    RetrievalResponse,
    RetrievedChunkDTO,
)
from rag_core.application.use_cases.retrieve_documents import RetrieveDocumentsUseCase
from rag_core.domain.entities.query import Query


@dataclass
class QueryService:
    """Service that provides simplified query operations.

    This service focuses on retrieval operations without generation,
    useful for search-only use cases.
    """

    retrieve_use_case: RetrieveDocumentsUseCase

    async def search(self, request: RetrievalRequest) -> RetrievalResponse:
        """Search for relevant chunks.

        Args:
            request: RetrievalRequest with query and options.

        Returns:
            RetrievalResponse with retrieved chunks.
        """
        # Create query entity
        query = Query(text=request.query, metadata=request.metadata)

        # Execute retrieval with pre-computed embedding if provided
        if request.query_embedding:
            result = await self.retrieve_use_case.execute_with_embedding(
                query_embedding=request.query_embedding,
                query=query,
                top_k=request.top_k,
                collection_name=request.collection_name,
                filter_metadata=request.filter_metadata,
                score_threshold=request.score_threshold,
            )
        else:
            result = await self.retrieve_use_case.execute(
                query=query,
                top_k=request.top_k,
                collection_name=request.collection_name,
                filter_metadata=request.filter_metadata,
                score_threshold=request.score_threshold,
            )

        # Convert to response DTO
        chunks = [
            RetrievedChunkDTO(
                chunk_id=r.chunk.id,
                document_id=r.document_id,
                content=r.content,
                score=r.score.value,
                rank=r.rank,
                embedding=r.chunk.embedding if request.include_embeddings else None,
                metadata=r.metadata,
            )
            for r in result.results
        ]

        return RetrievalResponse(
            query=request.query,
            chunks=chunks,
            total_candidates=result.total_candidates,
            retrieval_time_ms=result.retrieval_time_ms,
            metadata=result.metadata,
        )

    async def search_simple(
        self,
        query: str,
        top_k: int = 5,
        collection_name: str | None = None,
        score_threshold: float | None = None,
    ) -> list[RetrievedChunkDTO]:
        """Simplified search method.

        Args:
            query: The search query text.
            top_k: Number of results to return.
            collection_name: Optional collection name.
            score_threshold: Optional minimum score.

        Returns:
            List of RetrievedChunkDTO with results.
        """
        request = RetrievalRequest(
            query=query,
            top_k=top_k,
            collection_name=collection_name,
            score_threshold=score_threshold,
        )
        response = await self.search(request)
        return response.chunks

    async def get_context(
        self,
        query: str,
        top_k: int = 5,
        collection_name: str | None = None,
        separator: str = "\n\n",
    ) -> str:
        """Get combined context for a query.

        Args:
            query: The search query text.
            top_k: Number of results to return.
            collection_name: Optional collection name.
            separator: String to separate chunks.

        Returns:
            Combined context string from top chunks.
        """
        request = RetrievalRequest(
            query=query,
            top_k=top_k,
            collection_name=collection_name,
        )
        response = await self.search(request)
        return response.get_context(separator)
