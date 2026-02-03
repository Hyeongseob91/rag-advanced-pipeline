"""Tests for domain entities."""

import pytest
from uuid import UUID, uuid4
from datetime import datetime

from rag_core.domain.entities.document import Document, Chunk
from rag_core.domain.entities.query import Query, RewrittenQuery
from rag_core.domain.entities.response import GeneratedResponse, RetrievalResult
from rag_core.domain.value_objects.score import SimilarityScore, ScoreType


class TestDocument:
    """Tests for Document entity."""

    def test_create_document(self):
        """Test basic document creation."""
        doc = Document(content="Hello world", source="test.txt")

        assert doc.content == "Hello world"
        assert doc.source == "test.txt"
        assert isinstance(doc.id, UUID)
        assert isinstance(doc.created_at, datetime)
        assert doc.chunks == []

    def test_document_requires_content(self):
        """Test that document requires non-empty content."""
        with pytest.raises(ValueError, match="content cannot be empty"):
            Document(content="", source="test.txt")

    def test_document_requires_source(self):
        """Test that document requires non-empty source."""
        with pytest.raises(ValueError, match="source cannot be empty"):
            Document(content="Hello", source="")

    def test_document_char_count(self):
        """Test character count property."""
        doc = Document(content="Hello world", source="test.txt")
        assert doc.char_count == 11

    def test_document_add_chunks(self):
        """Test adding chunks to a document."""
        doc = Document(content="Hello world", source="test.txt")
        chunk = Chunk(content="Hello", document_id=doc.id, chunk_index=0)

        new_doc = doc.add_chunks([chunk])

        assert new_doc.chunk_count == 1
        assert doc.chunk_count == 0  # Original unchanged


class TestChunk:
    """Tests for Chunk entity."""

    def test_create_chunk(self):
        """Test basic chunk creation."""
        doc_id = uuid4()
        chunk = Chunk(content="Hello", document_id=doc_id, chunk_index=0)

        assert chunk.content == "Hello"
        assert chunk.document_id == doc_id
        assert chunk.chunk_index == 0
        assert chunk.embedding is None

    def test_chunk_requires_content(self):
        """Test that chunk requires non-empty content."""
        with pytest.raises(ValueError, match="content cannot be empty"):
            Chunk(content="", document_id=uuid4(), chunk_index=0)

    def test_chunk_requires_nonnegative_index(self):
        """Test that chunk index must be non-negative."""
        with pytest.raises(ValueError, match="non-negative"):
            Chunk(content="Hello", document_id=uuid4(), chunk_index=-1)

    def test_chunk_with_embedding(self):
        """Test creating chunk with embedding."""
        chunk = Chunk(content="Hello", document_id=uuid4(), chunk_index=0)
        embedding = [0.1, 0.2, 0.3]

        new_chunk = chunk.with_embedding(embedding)

        assert new_chunk.embedding == embedding
        assert chunk.embedding is None  # Original unchanged

    def test_chunk_token_count(self):
        """Test approximate token count."""
        chunk = Chunk(content="Hello world test", document_id=uuid4(), chunk_index=0)
        # 16 chars / 4 = 4 tokens (approximate)
        assert chunk.token_count == 4


class TestQuery:
    """Tests for Query entity."""

    def test_create_query(self):
        """Test basic query creation."""
        query = Query(text="What is AI?")

        assert query.text == "What is AI?"
        assert isinstance(query.id, UUID)
        assert query.cleaned_text == "What is AI?"

    def test_query_requires_text(self):
        """Test that query requires non-empty text."""
        with pytest.raises(ValueError, match="cannot be empty"):
            Query(text="")

        with pytest.raises(ValueError, match="cannot be empty"):
            Query(text="   ")

    def test_query_cleaned_text(self):
        """Test cleaned text removes whitespace."""
        query = Query(text="  What is AI?  ")
        assert query.cleaned_text == "What is AI?"


class TestRewrittenQuery:
    """Tests for RewrittenQuery entity."""

    def test_create_rewritten_query(self):
        """Test basic rewritten query creation."""
        original = Query(text="What is AI?")
        rewritten = RewrittenQuery(
            original_query=original,
            rewritten_text="Explain artificial intelligence",
        )

        assert rewritten.original_text == "What is AI?"
        assert rewritten.rewritten_text == "Explain artificial intelligence"
        assert rewritten.rewrite_strategy == "default"

    def test_rewritten_query_requires_text(self):
        """Test that rewritten query requires non-empty text."""
        original = Query(text="What is AI?")
        with pytest.raises(ValueError, match="cannot be empty"):
            RewrittenQuery(original_query=original, rewritten_text="")

    def test_all_search_terms(self):
        """Test all search terms collection."""
        original = Query(text="AI")
        rewritten = RewrittenQuery(
            original_query=original,
            rewritten_text="artificial intelligence",
            expansion_terms=["machine learning", "deep learning"],
        )

        terms = rewritten.all_search_terms
        assert "AI" in terms
        assert "artificial intelligence" in terms
        assert "machine learning" in terms
        assert "deep learning" in terms


class TestGeneratedResponse:
    """Tests for GeneratedResponse entity."""

    def test_create_response(self):
        """Test basic response creation."""
        query = Query(text="What is AI?")
        response = GeneratedResponse(
            query=query,
            answer="AI is artificial intelligence.",
        )

        assert response.answer == "AI is artificial intelligence."
        assert response.source_count == 0
        assert not response.has_sources

    def test_response_requires_answer(self):
        """Test that response requires non-empty answer."""
        query = Query(text="What is AI?")
        with pytest.raises(ValueError, match="cannot be empty"):
            GeneratedResponse(query=query, answer="")

    def test_response_with_sources(self):
        """Test response with source citations."""
        query = Query(text="What is AI?")
        chunk = Chunk(content="AI info", document_id=uuid4(), chunk_index=0)
        source = RetrievalResult(
            chunk=chunk,
            score=SimilarityScore(value=0.9),
            rank=1,
        )

        response = GeneratedResponse(
            query=query,
            answer="AI is artificial intelligence.",
            sources=[source],
        )

        assert response.source_count == 1
        assert response.has_sources

    def test_response_token_calculation(self):
        """Test automatic token total calculation."""
        query = Query(text="Test")
        response = GeneratedResponse(
            query=query,
            answer="Answer",
            prompt_tokens=100,
            completion_tokens=50,
        )

        assert response.total_tokens == 150
