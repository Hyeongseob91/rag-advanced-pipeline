"""Ingest Service - Orchestrates document ingestion."""

from dataclasses import dataclass
from time import perf_counter
from typing import Any
from uuid import UUID

from rag_core.application.dto.generation_dto import IngestRequest, IngestResponse
from rag_core.application.use_cases.ingest_document import IngestDocumentUseCase
from rag_core.domain.entities.document import Document
from rag_core.domain.interfaces.chunker_port import ChunkingConfig


@dataclass
class IngestService:
    """Service that orchestrates document ingestion.

    This service provides a simplified interface for ingesting documents
    into the RAG pipeline.
    """

    ingest_use_case: IngestDocumentUseCase

    async def ingest(self, request: IngestRequest) -> IngestResponse:
        """Ingest a document from a request.

        Args:
            request: IngestRequest with document content and options.

        Returns:
            IngestResponse with ingestion details.
        """
        # Create the document entity
        document = Document(
            content=request.content,
            source=request.source,
            metadata=request.metadata,
        )

        # Create chunking config
        config = ChunkingConfig(
            chunk_size=request.chunk_size,
            chunk_overlap=request.chunk_overlap,
        )

        # Execute ingestion
        result = await self.ingest_use_case.execute(
            document=document,
            collection_name=request.collection_name,
            chunking_config=config,
        )

        return IngestResponse(
            document_id=str(result.document_id),
            chunk_count=result.chunk_count,
            total_characters=result.total_characters,
            collection_name=result.collection_name,
            processing_time_ms=result.processing_time_ms,
            metadata=result.metadata,
        )

    async def ingest_text(
        self,
        content: str,
        source: str,
        collection_name: str | None = None,
        chunk_size: int = 512,
        chunk_overlap: int = 50,
        metadata: dict[str, Any] | None = None,
    ) -> IngestResponse:
        """Simplified method to ingest raw text.

        Args:
            content: The text content to ingest.
            source: Source identifier for the document.
            collection_name: Optional collection name.
            chunk_size: Size of chunks in characters.
            chunk_overlap: Overlap between chunks.
            metadata: Optional metadata for the document.

        Returns:
            IngestResponse with ingestion details.
        """
        request = IngestRequest(
            content=content,
            source=source,
            collection_name=collection_name,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            metadata=metadata or {},
        )
        return await self.ingest(request)

    async def ingest_batch(
        self,
        documents: list[tuple[str, str, dict[str, Any] | None]],
        collection_name: str | None = None,
        chunk_size: int = 512,
        chunk_overlap: int = 50,
    ) -> list[IngestResponse]:
        """Ingest multiple documents.

        Args:
            documents: List of (content, source, metadata) tuples.
            collection_name: Optional collection name.
            chunk_size: Size of chunks in characters.
            chunk_overlap: Overlap between chunks.

        Returns:
            List of IngestResponse for each document.
        """
        results = []
        for content, source, metadata in documents:
            response = await self.ingest_text(
                content=content,
                source=source,
                collection_name=collection_name,
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
                metadata=metadata,
            )
            results.append(response)
        return results

    async def delete(
        self,
        document_id: str,
        collection_name: str | None = None,
    ) -> int:
        """Delete a document and its chunks.

        Args:
            document_id: The document ID to delete.
            collection_name: Optional collection name.

        Returns:
            Number of chunks deleted.
        """
        return await self.ingest_use_case.delete_document(
            document_id=UUID(document_id),
            collection_name=collection_name,
        )
