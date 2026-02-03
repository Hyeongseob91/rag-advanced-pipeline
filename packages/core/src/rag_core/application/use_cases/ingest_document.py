"""Ingest Document Use Case."""

from dataclasses import dataclass
from time import perf_counter
from typing import Any
from uuid import UUID

from rag_core.domain.entities.document import Document
from rag_core.domain.interfaces.chunker_port import ChunkerPort, ChunkingConfig
from rag_core.domain.interfaces.embedding_port import EmbeddingPort
from rag_core.domain.interfaces.vectordb_port import VectorDBPort


@dataclass
class IngestResult:
    """Result from document ingestion."""

    document_id: UUID
    chunk_count: int
    total_characters: int
    collection_name: str
    processing_time_ms: float
    chunk_ids: list[UUID]
    metadata: dict[str, Any]


@dataclass
class IngestDocumentUseCase:
    """Use case for ingesting documents into the RAG pipeline.

    This use case handles the full ingestion flow:
    1. Chunk the document
    2. Generate embeddings for each chunk
    3. Store chunks in the vector database
    """

    chunker: ChunkerPort
    embedding_port: EmbeddingPort
    vectordb_port: VectorDBPort
    default_collection: str = "documents"

    async def execute(
        self,
        document: Document,
        collection_name: str | None = None,
        chunking_config: ChunkingConfig | None = None,
    ) -> IngestResult:
        """Execute the document ingestion use case.

        Args:
            document: The document to ingest.
            collection_name: Collection to store in.
            chunking_config: Optional chunking configuration.

        Returns:
            IngestResult with ingestion details.
        """
        start_time = perf_counter()
        collection = collection_name or self.default_collection

        # Step 1: Chunk the document
        chunking_result = await self.chunker.chunk(document, chunking_config)
        chunks = chunking_result.chunks

        if not chunks:
            return IngestResult(
                document_id=document.id,
                chunk_count=0,
                total_characters=document.char_count,
                collection_name=collection,
                processing_time_ms=(perf_counter() - start_time) * 1000,
                chunk_ids=[],
                metadata={"error": "No chunks generated"},
            )

        # Step 2: Generate embeddings for all chunks
        texts = [chunk.content for chunk in chunks]
        embedding_result = await self.embedding_port.embed_batch(texts)

        # Step 3: Attach embeddings to chunks
        chunks_with_embeddings = [
            chunk.with_embedding(embedding)
            for chunk, embedding in zip(chunks, embedding_result.embeddings, strict=True)
        ]

        # Step 4: Ensure collection exists
        if not await self.vectordb_port.collection_exists(collection):
            await self.vectordb_port.create_collection(
                collection_name=collection,
                dimension=self.embedding_port.dimension,
                metadata={"created_by": "ingest_document_use_case"},
            )

        # Step 5: Store chunks in vector database
        chunk_ids = await self.vectordb_port.upsert(
            chunks=chunks_with_embeddings,
            collection_name=collection,
        )

        elapsed_ms = (perf_counter() - start_time) * 1000

        return IngestResult(
            document_id=document.id,
            chunk_count=len(chunks),
            total_characters=document.char_count,
            collection_name=collection,
            processing_time_ms=elapsed_ms,
            chunk_ids=chunk_ids,
            metadata={
                "chunking_strategy": chunking_result.chunking_strategy,
                "embedding_model": self.embedding_port.model_name,
                "embedding_tokens": embedding_result.total_tokens,
                "average_chunk_size": chunking_result.average_chunk_size,
            },
        )

    async def execute_batch(
        self,
        documents: list[Document],
        collection_name: str | None = None,
        chunking_config: ChunkingConfig | None = None,
    ) -> list[IngestResult]:
        """Ingest multiple documents.

        Args:
            documents: List of documents to ingest.
            collection_name: Collection to store in.
            chunking_config: Optional chunking configuration.

        Returns:
            List of IngestResult for each document.
        """
        results = []
        for document in documents:
            result = await self.execute(
                document=document,
                collection_name=collection_name,
                chunking_config=chunking_config,
            )
            results.append(result)
        return results

    async def delete_document(
        self,
        document_id: UUID,
        collection_name: str | None = None,
    ) -> int:
        """Delete a document and all its chunks.

        Args:
            document_id: The document ID to delete.
            collection_name: Collection to delete from.

        Returns:
            Number of chunks deleted.
        """
        return await self.vectordb_port.delete_by_document(
            document_id=document_id,
            collection_name=collection_name or self.default_collection,
        )
