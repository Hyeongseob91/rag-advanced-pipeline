"""Tests for application layer DTOs."""

import pytest
from uuid import uuid4

from rag_core.application.dto.query_dto import QueryRequest, QueryResponse, SourceDTO
from rag_core.application.dto.retrieval_dto import (
    RetrievalRequest,
    RetrievalResponse,
    RetrievedChunkDTO,
)
from rag_core.application.dto.generation_dto import (
    GenerationRequest,
    GenerationResponse,
    IngestRequest,
    IngestResponse,
)


class TestQueryRequest:
    """Tests for QueryRequest DTO."""

    def test_create_request(self):
        """Test basic request creation."""
        request = QueryRequest(query="What is AI?")

        assert request.query == "What is AI?"
        assert request.top_k == 5
        assert request.rewrite_query is True
        assert request.collection_name is None

    def test_request_with_options(self):
        """Test request with all options."""
        request = QueryRequest(
            query="What is AI?",
            top_k=10,
            rewrite_query=False,
            collection_name="test",
            score_threshold=0.8,
            filter_metadata={"type": "article"},
        )

        assert request.top_k == 10
        assert request.rewrite_query is False
        assert request.collection_name == "test"
        assert request.score_threshold == 0.8
        assert request.filter_metadata == {"type": "article"}

    def test_request_requires_query(self):
        """Test that request requires non-empty query."""
        with pytest.raises(ValueError, match="cannot be empty"):
            QueryRequest(query="")

    def test_request_requires_positive_top_k(self):
        """Test that request requires positive top_k."""
        with pytest.raises(ValueError, match="positive"):
            QueryRequest(query="test", top_k=0)

    def test_request_validates_score_threshold(self):
        """Test score threshold validation."""
        with pytest.raises(ValueError, match="between 0 and 1"):
            QueryRequest(query="test", score_threshold=1.5)


class TestQueryResponse:
    """Tests for QueryResponse DTO."""

    def test_create_response(self):
        """Test basic response creation."""
        response = QueryResponse(
            query_id=uuid4(),
            original_query="What is AI?",
            rewritten_query="Explain artificial intelligence",
            answer="AI is...",
            sources=[],
            model="gpt-4",
        )

        assert response.original_query == "What is AI?"
        assert response.answer == "AI is..."
        assert response.source_count == 0

    def test_response_with_sources(self):
        """Test response with sources."""
        source = SourceDTO(
            chunk_id=uuid4(),
            document_id=uuid4(),
            content="AI info",
            score=0.9,
            rank=1,
        )

        response = QueryResponse(
            query_id=uuid4(),
            original_query="What is AI?",
            rewritten_query=None,
            answer="AI is...",
            sources=[source],
            model="gpt-4",
        )

        assert response.source_count == 1


class TestRetrievalRequest:
    """Tests for RetrievalRequest DTO."""

    def test_create_request(self):
        """Test basic request creation."""
        request = RetrievalRequest(query="test query")

        assert request.query == "test query"
        assert request.top_k == 10
        assert request.query_embedding is None

    def test_request_with_embedding(self):
        """Test request with pre-computed embedding."""
        embedding = [0.1, 0.2, 0.3]
        request = RetrievalRequest(
            query="test",
            query_embedding=embedding,
        )

        assert request.query_embedding == embedding


class TestRetrievalResponse:
    """Tests for RetrievalResponse DTO."""

    def test_create_response(self):
        """Test basic response creation."""
        chunk = RetrievedChunkDTO(
            chunk_id=uuid4(),
            document_id=uuid4(),
            content="Test content",
            score=0.9,
            rank=1,
        )

        response = RetrievalResponse(
            query="test",
            chunks=[chunk],
        )

        assert response.chunk_count == 1
        assert response.get_context() == "Test content"

    def test_get_context_with_separator(self):
        """Test context generation with custom separator."""
        chunks = [
            RetrievedChunkDTO(
                chunk_id=uuid4(),
                document_id=uuid4(),
                content="First",
                score=0.9,
                rank=1,
            ),
            RetrievedChunkDTO(
                chunk_id=uuid4(),
                document_id=uuid4(),
                content="Second",
                score=0.8,
                rank=2,
            ),
        ]

        response = RetrievalResponse(query="test", chunks=chunks)

        assert response.get_context(" | ") == "First | Second"

    def test_get_top_chunks(self):
        """Test getting top N chunks."""
        chunks = [
            RetrievedChunkDTO(
                chunk_id=uuid4(),
                document_id=uuid4(),
                content=f"Chunk {i}",
                score=0.9 - i * 0.1,
                rank=i + 1,
            )
            for i in range(5)
        ]

        response = RetrievalResponse(query="test", chunks=chunks)
        top_2 = response.get_top_chunks(2)

        assert len(top_2) == 2
        assert top_2[0].rank == 1


class TestGenerationRequest:
    """Tests for GenerationRequest DTO."""

    def test_create_request(self):
        """Test basic request creation."""
        request = GenerationRequest(
            query="What is AI?",
            context="AI is artificial intelligence.",
        )

        assert request.query == "What is AI?"
        assert request.context == "AI is artificial intelligence."
        assert request.temperature == 0.0
        assert request.max_tokens == 2000

    def test_request_requires_query(self):
        """Test that request requires query."""
        with pytest.raises(ValueError, match="cannot be empty"):
            GenerationRequest(query="", context="context")

    def test_request_requires_context(self):
        """Test that request requires context."""
        with pytest.raises(ValueError, match="cannot be empty"):
            GenerationRequest(query="test", context="")

    def test_request_validates_temperature(self):
        """Test temperature validation."""
        with pytest.raises(ValueError, match="between 0 and 2"):
            GenerationRequest(query="test", context="ctx", temperature=3.0)


class TestIngestRequest:
    """Tests for IngestRequest DTO."""

    def test_create_request(self):
        """Test basic request creation."""
        request = IngestRequest(
            content="Document content",
            source="test.txt",
        )

        assert request.content == "Document content"
        assert request.source == "test.txt"
        assert request.chunk_size == 512
        assert request.chunk_overlap == 50

    def test_request_requires_content(self):
        """Test that request requires content."""
        with pytest.raises(ValueError, match="cannot be empty"):
            IngestRequest(content="", source="test.txt")

    def test_request_requires_source(self):
        """Test that request requires source."""
        with pytest.raises(ValueError, match="cannot be empty"):
            IngestRequest(content="content", source="")

    def test_request_validates_chunk_size(self):
        """Test chunk size validation."""
        with pytest.raises(ValueError, match="positive"):
            IngestRequest(content="test", source="test.txt", chunk_size=0)


class TestIngestResponse:
    """Tests for IngestResponse DTO."""

    def test_create_response(self):
        """Test basic response creation."""
        response = IngestResponse(
            document_id="doc-123",
            chunk_count=5,
            total_characters=1000,
            collection_name="documents",
            processing_time_ms=100.5,
        )

        assert response.document_id == "doc-123"
        assert response.chunk_count == 5
        assert response.total_characters == 1000
