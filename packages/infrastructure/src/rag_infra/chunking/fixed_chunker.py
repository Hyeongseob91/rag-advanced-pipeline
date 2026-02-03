"""Fixed Size Chunker implementation."""

from uuid import UUID, uuid4

from rag_core.domain.entities.document import Chunk, Document
from rag_core.domain.interfaces.chunker_port import (
    ChunkerPort,
    ChunkingConfig,
    ChunkingResult,
)


class FixedChunker(ChunkerPort):
    """Fixed size chunker implementing ChunkerPort.

    This chunker splits text into fixed-size chunks with optional overlap.
    It respects word boundaries when possible.
    """

    @property
    def strategy_name(self) -> str:
        """Return the strategy name."""
        return "fixed_size"

    async def chunk(
        self,
        document: Document,
        config: ChunkingConfig | None = None,
    ) -> ChunkingResult:
        """Chunk a document into fixed-size pieces.

        Args:
            document: The document to chunk.
            config: Optional chunking configuration.

        Returns:
            ChunkingResult with chunks.
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
                "chunk_size": cfg.chunk_size,
                "chunk_overlap": cfg.chunk_overlap,
            },
        )

    async def chunk_text(
        self,
        text: str,
        document_id: str | None = None,
        config: ChunkingConfig | None = None,
    ) -> list[Chunk]:
        """Chunk raw text into fixed-size pieces.

        Args:
            text: The text to chunk.
            document_id: Optional document ID.
            config: Optional chunking configuration.

        Returns:
            List of Chunk objects.
        """
        cfg = config or self.default_config
        doc_id = UUID(document_id) if document_id else uuid4()

        # Split text into chunks
        chunks: list[Chunk] = []
        start = 0
        chunk_index = 0

        while start < len(text):
            # Calculate end position
            end = min(start + cfg.chunk_size, len(text))

            # Try to break at a word boundary if not at the end
            if end < len(text):
                # Look for whitespace within the last 20% of the chunk
                search_start = max(start, end - cfg.chunk_size // 5)
                last_space = text.rfind(" ", search_start, end)
                if last_space > start:
                    end = last_space

            # Extract chunk text
            chunk_text = text[start:end].strip()

            # Skip empty chunks
            if chunk_text and len(chunk_text) >= cfg.min_chunk_size:
                chunks.append(
                    Chunk(
                        content=chunk_text,
                        document_id=doc_id,
                        chunk_index=chunk_index,
                        start_char=start,
                        end_char=end,
                    )
                )
                chunk_index += 1

            # Move start position with overlap
            start = end - cfg.chunk_overlap
            if start <= 0 or end >= len(text):
                start = end

        return chunks


class RecursiveChunker(ChunkerPort):
    """Recursive character text splitter.

    Splits text by trying different separators recursively,
    from paragraphs to sentences to words.
    """

    def __init__(self, separators: list[str] | None = None):
        """Initialize the recursive chunker.

        Args:
            separators: List of separators to try, in order.
        """
        self._separators = separators or ["\n\n", "\n", ". ", " ", ""]

    @property
    def strategy_name(self) -> str:
        """Return the strategy name."""
        return "recursive"

    async def chunk(
        self,
        document: Document,
        config: ChunkingConfig | None = None,
    ) -> ChunkingResult:
        """Chunk a document recursively.

        Args:
            document: The document to chunk.
            config: Optional chunking configuration.

        Returns:
            ChunkingResult with chunks.
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
                "chunk_size": cfg.chunk_size,
                "chunk_overlap": cfg.chunk_overlap,
                "separators": self._separators,
            },
        )

    async def chunk_text(
        self,
        text: str,
        document_id: str | None = None,
        config: ChunkingConfig | None = None,
    ) -> list[Chunk]:
        """Chunk text recursively.

        Args:
            text: The text to chunk.
            document_id: Optional document ID.
            config: Optional chunking configuration.

        Returns:
            List of Chunk objects.
        """
        cfg = config or self.default_config
        doc_id = UUID(document_id) if document_id else uuid4()

        # Split text recursively
        splits = self._split_text(text, cfg.chunk_size, 0)

        # Create chunks with overlap handling
        chunks: list[Chunk] = []
        for i, split_text in enumerate(splits):
            if split_text.strip() and len(split_text.strip()) >= cfg.min_chunk_size:
                chunks.append(
                    Chunk(
                        content=split_text.strip(),
                        document_id=doc_id,
                        chunk_index=i,
                    )
                )

        return chunks

    def _split_text(
        self,
        text: str,
        chunk_size: int,
        separator_index: int,
    ) -> list[str]:
        """Recursively split text.

        Args:
            text: Text to split.
            chunk_size: Target chunk size.
            separator_index: Current separator index.

        Returns:
            List of text splits.
        """
        if separator_index >= len(self._separators):
            # No more separators, just split by size
            return [text[i : i + chunk_size] for i in range(0, len(text), chunk_size)]

        separator = self._separators[separator_index]

        if separator == "":
            # Split by character
            return [text[i : i + chunk_size] for i in range(0, len(text), chunk_size)]

        # Split by current separator
        parts = text.split(separator)
        result: list[str] = []
        current = ""

        for part in parts:
            # Check if adding this part would exceed chunk size
            test_text = current + separator + part if current else part

            if len(test_text) <= chunk_size:
                current = test_text
            else:
                # Current chunk is full
                if current:
                    result.append(current)

                # Check if part itself needs splitting
                if len(part) > chunk_size:
                    # Recursively split with next separator
                    sub_splits = self._split_text(part, chunk_size, separator_index + 1)
                    result.extend(sub_splits)
                    current = ""
                else:
                    current = part

        # Add remaining text
        if current:
            result.append(current)

        return result
