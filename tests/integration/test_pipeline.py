"""Integration tests for the RAG pipeline.

These tests require external services (LLM, VectorDB) to be running.
Mark them to skip if services are not available.
"""

import pytest
from unittest.mock import AsyncMock, MagicMock
from uuid import uuid4

from rag_core.application.dto.query_dto import QueryRequest
from rag_core.application.services.pipeline_service import PipelineService
from rag_core.application.use_cases.generate_answer import GenerateAnswerUseCase
from rag_core.application.use_cases.query_rewrite import QueryRewriteUseCase
from rag_core.application.use_cases.retrieve_documents import RetrieveDocumentsUseCase
from rag_core.domain.entities.document import Chunk
from rag_core.domain.interfaces.llm_port import LLMPort, LLMResponse
from rag_core.domain.interfaces.embedding_port import EmbeddingPort, EmbeddingResult
from rag_core.domain.interfaces.vectordb_port import (
    VectorDBPort,
    SearchResult,
    SearchResults,
)
from rag_core.domain.value_objects.score import SimilarityScore


class TestPipelineServiceIntegration:
    """Integration tests for PipelineService with mocked adapters."""

    @pytest.fixture
    def mock_llm(self):
        """Create a mock LLM adapter."""
        llm = AsyncMock(spec=LLMPort)
        llm.model_name = "mock-model"
        llm.generate.return_value = LLMResponse(
            content="This is a generated answer.",
            model="mock-model",
            prompt_tokens=100,
            completion_tokens=50,
        )
        return llm

    @pytest.fixture
    def mock_embedding(self):
        """Create a mock embedding adapter."""
        embedding = AsyncMock(spec=EmbeddingPort)
        embedding.model_name = "mock-embedding"
        embedding.dimension = 384
        embedding.embed.return_value = [0.1] * 384
        embedding.embed_batch.return_value = EmbeddingResult(
            embeddings=[[0.1] * 384],
            model="mock-embedding",
        )
        return embedding

    @pytest.fixture
    def mock_vectordb(self):
        """Create a mock vector database adapter."""
        vectordb = AsyncMock(spec=VectorDBPort)
        vectordb.search.return_value = SearchResults(
            results=[
                SearchResult(
                    chunk_id=uuid4(),
                    content="Relevant content from the database.",
                    score=SimilarityScore(value=0.9),
                    document_id=uuid4(),
                ),
            ],
            total_count=1,
            query_time_ms=10.0,
        )
        return vectordb

    @pytest.fixture
    def pipeline_service(self, mock_llm, mock_embedding, mock_vectordb):
        """Create a PipelineService with mocked dependencies."""
        return PipelineService(
            query_rewrite_use_case=QueryRewriteUseCase(llm=mock_llm),
            retrieve_documents_use_case=RetrieveDocumentsUseCase(
                embedding_port=mock_embedding,
                vectordb_port=mock_vectordb,
            ),
            generate_answer_use_case=GenerateAnswerUseCase(llm=mock_llm),
        )

    @pytest.mark.asyncio
    async def test_execute_full_pipeline(self, pipeline_service, mock_llm):
        """Test executing the full RAG pipeline."""
        request = QueryRequest(
            query="What is machine learning?",
            top_k=3,
            rewrite_query=True,
        )

        response = await pipeline_service.execute(request)

        assert response.answer == "This is a generated answer."
        assert response.model == "mock-model"
        assert len(response.sources) == 1
        assert response.total_time_ms > 0

    @pytest.mark.asyncio
    async def test_execute_without_query_rewrite(
        self, pipeline_service, mock_llm, mock_embedding
    ):
        """Test pipeline without query rewriting."""
        request = QueryRequest(
            query="What is machine learning?",
            top_k=3,
            rewrite_query=False,
        )

        response = await pipeline_service.execute(request)

        assert response.rewritten_query is None
        # Should only call generate once (for answer, not rewrite)
        assert mock_llm.generate.call_count == 1

    @pytest.mark.asyncio
    async def test_execute_with_score_threshold(
        self, pipeline_service, mock_vectordb
    ):
        """Test pipeline with score threshold."""
        request = QueryRequest(
            query="What is machine learning?",
            top_k=3,
            score_threshold=0.8,
        )

        await pipeline_service.execute(request)

        # Verify threshold was passed to search
        call_args = mock_vectordb.search.call_args
        assert call_args.kwargs.get("score_threshold") == 0.8

    @pytest.mark.asyncio
    async def test_retrieve_only(self, pipeline_service):
        """Test retrieval without generation."""
        request = QueryRequest(
            query="What is machine learning?",
            top_k=5,
        )

        sources = await pipeline_service.retrieve_only(request)

        assert len(sources) == 1
        assert sources[0].score == 0.9


class TestPipelineWithMockedServices:
    """Test pipeline behavior with various service responses."""

    @pytest.mark.asyncio
    async def test_handles_empty_retrieval(self):
        """Test pipeline handles empty retrieval results."""
        mock_llm = AsyncMock(spec=LLMPort)
        mock_llm.model_name = "test"
        mock_llm.generate.return_value = LLMResponse(
            content="I don't have enough information.",
            model="test",
        )

        mock_embedding = AsyncMock(spec=EmbeddingPort)
        mock_embedding.embed.return_value = [0.1] * 384

        mock_vectordb = AsyncMock(spec=VectorDBPort)
        mock_vectordb.search.return_value = SearchResults(
            results=[],  # Empty results
            total_count=0,
        )

        service = PipelineService(
            query_rewrite_use_case=QueryRewriteUseCase(llm=mock_llm),
            retrieve_documents_use_case=RetrieveDocumentsUseCase(
                embedding_port=mock_embedding,
                vectordb_port=mock_vectordb,
            ),
            generate_answer_use_case=GenerateAnswerUseCase(llm=mock_llm),
        )

        request = QueryRequest(query="Unknown topic", rewrite_query=False)
        response = await service.execute(request)

        assert response.answer is not None
        assert len(response.sources) == 0

    @pytest.mark.asyncio
    async def test_handles_multiple_chunks(self):
        """Test pipeline handles multiple retrieved chunks."""
        mock_llm = AsyncMock(spec=LLMPort)
        mock_llm.model_name = "test"
        mock_llm.generate.return_value = LLMResponse(
            content="Based on multiple sources...",
            model="test",
        )

        mock_embedding = AsyncMock(spec=EmbeddingPort)
        mock_embedding.embed.return_value = [0.1] * 384

        mock_vectordb = AsyncMock(spec=VectorDBPort)
        mock_vectordb.search.return_value = SearchResults(
            results=[
                SearchResult(
                    chunk_id=uuid4(),
                    content=f"Chunk {i} content",
                    score=SimilarityScore(value=0.9 - i * 0.1),
                    document_id=uuid4(),
                )
                for i in range(5)
            ],
            total_count=5,
        )

        service = PipelineService(
            query_rewrite_use_case=QueryRewriteUseCase(llm=mock_llm),
            retrieve_documents_use_case=RetrieveDocumentsUseCase(
                embedding_port=mock_embedding,
                vectordb_port=mock_vectordb,
            ),
            generate_answer_use_case=GenerateAnswerUseCase(llm=mock_llm),
        )

        request = QueryRequest(query="Test query", top_k=5, rewrite_query=False)
        response = await service.execute(request)

        assert len(response.sources) == 5
        # Sources should be ranked
        for i, source in enumerate(response.sources):
            assert source.rank == i + 1
