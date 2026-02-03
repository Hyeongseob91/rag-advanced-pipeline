"""Semantic Chunker implementation."""

import asyncio
from concurrent.futures import ThreadPoolExecutor
from typing import Any
from uuid import UUID, uuid4

import numpy as np

from rag_core.domain.entities.document import Chunk, Document
from rag_core.domain.interfaces.chunker_port import (
    ChunkerPort,
    ChunkingConfig,
    ChunkingResult,
)
from rag_core.domain.interfaces.embedding_port import EmbeddingPort


class SemanticChunker(ChunkerPort):
    """Semantic chunker implementing ChunkerPort.

    This chunker uses embeddings to split text at semantic boundaries.
    It identifies breakpoints where the semantic similarity between
    adjacent sentences drops significantly.
    """

    def __init__(
        self,
        embedding_port: EmbeddingPort | None = None,
        breakpoint_threshold: float = 0.5,
        min_sentences_per_chunk: int = 2,
        max_sentences_per_chunk: int = 20,
    ):
        """Initialize the semantic chunker.

        Args:
            embedding_port: Optional embedding port for semantic analysis.
            breakpoint_threshold: Similarity threshold for chunk breaks.
            min_sentences_per_chunk: Minimum sentences per chunk.
            max_sentences_per_chunk: Maximum sentences per chunk.
        """
        self._embedding_port = embedding_port
        self._breakpoint_threshold = breakpoint_threshold
        self._min_sentences = min_sentences_per_chunk
        self._max_sentences = max_sentences_per_chunk
        self._executor = ThreadPoolExecutor(max_workers=2)

    @property
    def strategy_name(self) -> str:
        """Return the strategy name."""
        return "semantic"

    async def chunk(
        self,
        document: Document,
        config: ChunkingConfig | None = None,
    ) -> ChunkingResult:
        """Chunk a document semantically.

        Args:
            document: The document to chunk.
            config: Optional chunking configuration.

        Returns:
            ChunkingResult with semantically coherent chunks.
        """
        cfg = config or self.default_config
        chunks = await self.chunk_text(
            text=document.content,
            document_id=str(document.id),
            config=cfg,
        )

        return ChunkingResult(
            chunks=chunks,
            total_characters=document.char_count,
            chunking_strategy=self.strategy_name,
            metadata={
                "breakpoint_threshold": self._breakpoint_threshold,
                "min_sentences": self._min_sentences,
                "max_sentences": self._max_sentences,
            },
        )

    async def chunk_text(
        self,
        text: str,
        document_id: str | None = None,
        config: ChunkingConfig | None = None,
    ) -> list[Chunk]:
        """Chunk text semantically.

        Args:
            text: The text to chunk.
            document_id: Optional document ID.
            config: Optional chunking configuration.

        Returns:
            List of Chunk objects.
        """
        cfg = config or self.default_config
        doc_id = UUID(document_id) if document_id else uuid4()

        # Split into sentences
        sentences = self._split_sentences(text)

        if len(sentences) <= self._min_sentences:
            # Too few sentences, return as single chunk
            return [
                Chunk(
                    content=text.strip(),
                    document_id=doc_id,
                    chunk_index=0,
                )
            ]

        # If no embedding port, fall back to fixed chunking
        if self._embedding_port is None:
            return await self._fallback_chunk(text, doc_id, cfg)

        # Get embeddings for all sentences
        embeddings = await self._embed_sentences(sentences)

        # Find semantic breakpoints
        breakpoints = self._find_breakpoints(embeddings)

        # Create chunks based on breakpoints
        chunks = self._create_chunks_from_breakpoints(
            sentences, breakpoints, doc_id, cfg
        )

        return chunks

    def _split_sentences(self, text: str) -> list[str]:
        """Split text into sentences.

        Args:
            text: Text to split.

        Returns:
            List of sentences.
        """
        import re

        # Simple sentence splitting
        # Split on period, exclamation, question mark followed by space or end
        sentence_endings = re.compile(r'(?<=[.!?])\s+')
        sentences = sentence_endings.split(text)

        # Clean up sentences
        return [s.strip() for s in sentences if s.strip()]

    async def _embed_sentences(self, sentences: list[str]) -> list[list[float]]:
        """Embed sentences using the embedding port.

        Args:
            sentences: List of sentences.

        Returns:
            List of embeddings.
        """
        if self._embedding_port is None:
            raise ValueError("No embedding port configured")

        result = await self._embedding_port.embed_batch(sentences)
        return result.embeddings

    def _find_breakpoints(self, embeddings: list[list[float]]) -> list[int]:
        """Find semantic breakpoints between sentences.

        Args:
            embeddings: List of sentence embeddings.

        Returns:
            List of breakpoint indices.
        """
        if len(embeddings) < 2:
            return []

        # Calculate cosine similarities between adjacent sentences
        similarities = []
        for i in range(len(embeddings) - 1):
            sim = self._cosine_similarity(embeddings[i], embeddings[i + 1])
            similarities.append(sim)

        # Find breakpoints where similarity drops below threshold
        breakpoints = []
        for i, sim in enumerate(similarities):
            if sim < self._breakpoint_threshold:
                breakpoints.append(i + 1)  # Break after this sentence

        return breakpoints

    def _cosine_similarity(self, a: list[float], b: list[float]) -> float:
        """Calculate cosine similarity between two vectors.

        Args:
            a: First vector.
            b: Second vector.

        Returns:
            Cosine similarity score.
        """
        a_arr = np.array(a)
        b_arr = np.array(b)
        dot = np.dot(a_arr, b_arr)
        norm_a = np.linalg.norm(a_arr)
        norm_b = np.linalg.norm(b_arr)
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return float(dot / (norm_a * norm_b))

    def _create_chunks_from_breakpoints(
        self,
        sentences: list[str],
        breakpoints: list[int],
        doc_id: UUID,
        config: ChunkingConfig,
    ) -> list[Chunk]:
        """Create chunks from sentences using breakpoints.

        Args:
            sentences: List of sentences.
            breakpoints: List of breakpoint indices.
            doc_id: Document UUID.
            config: Chunking configuration.

        Returns:
            List of Chunk objects.
        """
        chunks: list[Chunk] = []
        chunk_index = 0
        start_idx = 0

        # Add end boundary
        all_breaks = breakpoints + [len(sentences)]

        for end_idx in all_breaks:
            # Ensure minimum sentences
            if end_idx - start_idx < self._min_sentences and end_idx < len(sentences):
                continue

            # Ensure maximum sentences
            while end_idx - start_idx > self._max_sentences:
                mid = start_idx + self._max_sentences
                chunk_text = " ".join(sentences[start_idx:mid])

                if len(chunk_text) >= config.min_chunk_size:
                    chunks.append(
                        Chunk(
                            content=chunk_text,
                            document_id=doc_id,
                            chunk_index=chunk_index,
                        )
                    )
                    chunk_index += 1

                start_idx = mid

            # Create chunk from remaining sentences
            chunk_text = " ".join(sentences[start_idx:end_idx])

            if chunk_text and len(chunk_text) >= config.min_chunk_size:
                chunks.append(
                    Chunk(
                        content=chunk_text,
                        document_id=doc_id,
                        chunk_index=chunk_index,
                    )
                )
                chunk_index += 1

            start_idx = end_idx

        return chunks

    async def _fallback_chunk(
        self,
        text: str,
        doc_id: UUID,
        config: ChunkingConfig,
    ) -> list[Chunk]:
        """Fallback to sentence-based chunking without embeddings.

        Args:
            text: Text to chunk.
            doc_id: Document UUID.
            config: Chunking configuration.

        Returns:
            List of Chunk objects.
        """
        sentences = self._split_sentences(text)
        chunks: list[Chunk] = []
        chunk_index = 0
        current_sentences: list[str] = []
        current_length = 0

        for sentence in sentences:
            sentence_length = len(sentence)

            # Check if adding this sentence would exceed chunk size
            if (
                current_length + sentence_length > config.chunk_size
                and len(current_sentences) >= self._min_sentences
            ):
                # Create chunk
                chunk_text = " ".join(current_sentences)
                chunks.append(
                    Chunk(
                        content=chunk_text,
                        document_id=doc_id,
                        chunk_index=chunk_index,
                    )
                )
                chunk_index += 1
                current_sentences = []
                current_length = 0

            current_sentences.append(sentence)
            current_length += sentence_length + 1  # +1 for space

            # Enforce max sentences
            if len(current_sentences) >= self._max_sentences:
                chunk_text = " ".join(current_sentences)
                chunks.append(
                    Chunk(
                        content=chunk_text,
                        document_id=doc_id,
                        chunk_index=chunk_index,
                    )
                )
                chunk_index += 1
                current_sentences = []
                current_length = 0

        # Handle remaining sentences
        if current_sentences:
            chunk_text = " ".join(current_sentences)
            if len(chunk_text) >= config.min_chunk_size:
                chunks.append(
                    Chunk(
                        content=chunk_text,
                        document_id=doc_id,
                        chunk_index=chunk_index,
                    )
                )

        return chunks
