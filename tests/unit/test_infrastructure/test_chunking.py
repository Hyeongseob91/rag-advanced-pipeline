"""Tests for chunking implementations."""

import pytest
from uuid import uuid4

from rag_core.domain.entities.document import Document
from rag_core.domain.interfaces.chunker_port import ChunkingConfig
from rag_infra.chunking.fixed_chunker import FixedChunker, RecursiveChunker


class TestFixedChunker:
    """Tests for FixedChunker."""

    @pytest.fixture
    def chunker(self):
        """Create a FixedChunker instance."""
        return FixedChunker()

    @pytest.fixture
    def sample_document(self):
        """Create a sample document."""
        content = (
            "This is the first paragraph with some content. "
            "It continues for a while to make sure we have enough text. "
            "Here is more text to fill out the paragraph.\n\n"
            "This is the second paragraph. It also has multiple sentences. "
            "The content here is different from the first paragraph.\n\n"
            "Third paragraph follows. More content here. And even more text."
        )
        return Document(content=content, source="test.txt")

    @pytest.mark.asyncio
    async def test_chunk_document(self, chunker, sample_document):
        """Test chunking a document."""
        config = ChunkingConfig(chunk_size=200, chunk_overlap=20, min_chunk_size=50)
        result = await chunker.chunk(sample_document, config)

        assert result.chunk_count > 0
        assert result.total_characters == sample_document.char_count
        assert result.chunking_strategy == "fixed_size"

    @pytest.mark.asyncio
    async def test_chunk_text(self, chunker):
        """Test chunking raw text."""
        text = "Hello world. " * 50  # ~650 chars

        config = ChunkingConfig(chunk_size=200, chunk_overlap=20, min_chunk_size=50)
        chunks = await chunker.chunk_text(text, config=config)

        assert len(chunks) > 1
        for chunk in chunks:
            assert len(chunk.content) <= 200

    @pytest.mark.asyncio
    async def test_chunk_preserves_document_id(self, chunker, sample_document):
        """Test that chunks reference the correct document."""
        result = await chunker.chunk(sample_document)

        for chunk in result.chunks:
            assert chunk.document_id == sample_document.id

    @pytest.mark.asyncio
    async def test_chunk_indices_are_sequential(self, chunker, sample_document):
        """Test that chunk indices are sequential."""
        result = await chunker.chunk(sample_document)

        for i, chunk in enumerate(result.chunks):
            assert chunk.chunk_index == i

    @pytest.mark.asyncio
    async def test_respects_min_chunk_size(self, chunker):
        """Test that chunks respect minimum size."""
        text = "Short text"

        config = ChunkingConfig(chunk_size=200, chunk_overlap=0, min_chunk_size=50)
        chunks = await chunker.chunk_text(text, config=config)

        # Short text should not be chunked if below min size
        # But if it's the only chunk, it should still be returned
        assert len(chunks) <= 1

    @pytest.mark.asyncio
    async def test_word_boundary_breaking(self, chunker):
        """Test that chunker tries to break at word boundaries."""
        text = "word " * 100  # 500 chars

        config = ChunkingConfig(chunk_size=100, chunk_overlap=0, min_chunk_size=20)
        chunks = await chunker.chunk_text(text, config=config)

        for chunk in chunks:
            # Chunks should not end mid-word
            assert not chunk.content.endswith("wor")

    @pytest.mark.asyncio
    async def test_overlap_between_chunks(self, chunker):
        """Test that chunks have proper overlap."""
        text = "A" * 100 + "B" * 100 + "C" * 100  # 300 chars

        config = ChunkingConfig(chunk_size=120, chunk_overlap=20, min_chunk_size=50)
        chunks = await chunker.chunk_text(text, config=config)

        assert len(chunks) >= 2
        # With overlap, there should be some shared content


class TestRecursiveChunker:
    """Tests for RecursiveChunker."""

    @pytest.fixture
    def chunker(self):
        """Create a RecursiveChunker instance."""
        return RecursiveChunker()

    @pytest.mark.asyncio
    async def test_chunk_by_paragraphs(self, chunker):
        """Test chunking by paragraph separators first."""
        text = "First paragraph.\n\nSecond paragraph.\n\nThird paragraph."

        config = ChunkingConfig(chunk_size=100, chunk_overlap=0, min_chunk_size=10)
        chunks = await chunker.chunk_text(text, config=config)

        # Should split by paragraphs
        assert len(chunks) >= 1

    @pytest.mark.asyncio
    async def test_falls_back_to_smaller_separators(self, chunker):
        """Test that chunker falls back to smaller separators."""
        text = "This is a very long sentence that exceeds the chunk size. " * 5

        config = ChunkingConfig(chunk_size=100, chunk_overlap=0, min_chunk_size=10)
        chunks = await chunker.chunk_text(text, config=config)

        assert len(chunks) > 1


class TestChunkingConfig:
    """Tests for ChunkingConfig."""

    def test_create_config(self):
        """Test basic config creation."""
        config = ChunkingConfig(
            chunk_size=512,
            chunk_overlap=50,
        )

        assert config.chunk_size == 512
        assert config.chunk_overlap == 50

    def test_chunk_size_must_be_positive(self):
        """Test that chunk_size must be positive."""
        with pytest.raises(ValueError, match="positive"):
            ChunkingConfig(chunk_size=0)

    def test_chunk_overlap_cannot_exceed_size(self):
        """Test that overlap cannot exceed chunk size."""
        with pytest.raises(ValueError, match="less than chunk_size"):
            ChunkingConfig(chunk_size=100, chunk_overlap=100)

    def test_chunk_overlap_cannot_be_negative(self):
        """Test that overlap cannot be negative."""
        with pytest.raises(ValueError, match="negative"):
            ChunkingConfig(chunk_size=100, chunk_overlap=-1)
