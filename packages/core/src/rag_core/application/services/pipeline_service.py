"""Pipeline Service - Orchestrates the RAG pipeline."""

from dataclasses import dataclass
from time import perf_counter
from typing import Any
from uuid import uuid4

from rag_core.application.dto.query_dto import QueryRequest, QueryResponse, SourceDTO
from rag_core.application.use_cases.generate_answer import GenerateAnswerUseCase
from rag_core.application.use_cases.query_rewrite import QueryRewriteUseCase
from rag_core.application.use_cases.retrieve_documents import RetrieveDocumentsUseCase
from rag_core.domain.entities.query import Query


@dataclass
class PipelineService:
    """Service that orchestrates the full RAG pipeline.

    This service coordinates the query rewriting, retrieval, and generation
    use cases to provide a complete RAG experience.
    """

    query_rewrite_use_case: QueryRewriteUseCase
    retrieve_documents_use_case: RetrieveDocumentsUseCase
    generate_answer_use_case: GenerateAnswerUseCase

    async def execute(self, request: QueryRequest) -> QueryResponse:
        """Execute the full RAG pipeline.

        Args:
            request: QueryRequest with the user query and options.

        Returns:
            QueryResponse with the generated answer and sources.
        """
        start_time = perf_counter()

        # Create the query entity
        query = Query(text=request.query, metadata=request.metadata)

        # Step 1: Optionally rewrite the query
        rewritten_text: str | None = None
        search_query = query

        if request.rewrite_query:
            rewritten = await self.query_rewrite_use_case.execute(query)
            rewritten_text = rewritten.rewritten_text
            # Use rewritten query for retrieval
            search_query = Query(text=rewritten.rewritten_text)

        # Step 2: Retrieve relevant documents
        retrieval_result = await self.retrieve_documents_use_case.execute(
            query=search_query,
            top_k=request.top_k,
            collection_name=request.collection_name,
            filter_metadata=request.filter_metadata,
            score_threshold=request.score_threshold,
        )

        # Step 3: Build context from retrieved chunks
        context = retrieval_result.get_context()

        # Step 4: Generate answer
        generation_result = await self.generate_answer_use_case.execute(
            query=query,  # Use original query for generation
            context=context,
            sources=retrieval_result.results,
        )

        # Build response
        sources = [
            SourceDTO(
                chunk_id=result.chunk.id,
                document_id=result.document_id,
                content=result.content,
                score=result.score.value,
                rank=result.rank,
                metadata=result.metadata,
            )
            for result in retrieval_result.results
        ]

        total_time = (perf_counter() - start_time) * 1000

        return QueryResponse(
            query_id=query.id,
            original_query=request.query,
            rewritten_query=rewritten_text,
            answer=generation_result.answer,
            sources=sources,
            model=generation_result.model,
            total_tokens=generation_result.total_tokens,
            retrieval_time_ms=retrieval_result.retrieval_time_ms,
            generation_time_ms=generation_result.generation_time_ms,
            total_time_ms=total_time,
            metadata={
                "rewrite_enabled": request.rewrite_query,
                "chunks_retrieved": len(retrieval_result.results),
            },
        )

    async def execute_streaming(self, request: QueryRequest):
        """Execute the RAG pipeline with streaming answer generation.

        Args:
            request: QueryRequest with the user query and options.

        Yields:
            String chunks of the generated answer.
        """
        # Create the query entity
        query = Query(text=request.query, metadata=request.metadata)

        # Step 1: Optionally rewrite the query
        search_query = query

        if request.rewrite_query:
            rewritten = await self.query_rewrite_use_case.execute(query)
            search_query = Query(text=rewritten.rewritten_text)

        # Step 2: Retrieve relevant documents
        retrieval_result = await self.retrieve_documents_use_case.execute(
            query=search_query,
            top_k=request.top_k,
            collection_name=request.collection_name,
            filter_metadata=request.filter_metadata,
            score_threshold=request.score_threshold,
        )

        # Step 3: Build context from retrieved chunks
        context = retrieval_result.get_context()

        # Step 4: Stream the answer
        async for chunk in self.generate_answer_use_case.execute_with_streaming(
            query=query,
            context=context,
        ):
            yield chunk

    async def retrieve_only(self, request: QueryRequest) -> list[SourceDTO]:
        """Execute only the retrieval step without generation.

        Useful for debugging or when you only need relevant chunks.

        Args:
            request: QueryRequest with the user query and options.

        Returns:
            List of SourceDTOs with retrieved chunks.
        """
        query = Query(text=request.query, metadata=request.metadata)

        # Optionally rewrite
        search_query = query
        if request.rewrite_query:
            rewritten = await self.query_rewrite_use_case.execute(query)
            search_query = Query(text=rewritten.rewritten_text)

        # Retrieve
        retrieval_result = await self.retrieve_documents_use_case.execute(
            query=search_query,
            top_k=request.top_k,
            collection_name=request.collection_name,
            filter_metadata=request.filter_metadata,
            score_threshold=request.score_threshold,
        )

        return [
            SourceDTO(
                chunk_id=result.chunk.id,
                document_id=result.document_id,
                content=result.content,
                score=result.score.value,
                rank=result.rank,
                metadata=result.metadata,
            )
            for result in retrieval_result.results
        ]
