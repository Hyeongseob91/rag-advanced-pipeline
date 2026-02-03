"""Advanced RAG Pipeline Facade - Simplified interface for RAG operations."""

from dataclasses import dataclass, field
from typing import Any, AsyncIterator

from rag_core.application.dto.generation_dto import IngestRequest, IngestResponse
from rag_core.application.dto.query_dto import QueryRequest, QueryResponse, SourceDTO
from rag_core.application.services.ingest_service import IngestService
from rag_core.application.services.pipeline_service import PipelineService
from rag_core.application.services.query_service import QueryService
from rag_core.application.use_cases.generate_answer import GenerateAnswerUseCase
from rag_core.application.use_cases.ingest_document import IngestDocumentUseCase
from rag_core.application.use_cases.query_rewrite import QueryRewriteUseCase
from rag_core.application.use_cases.retrieve_documents import RetrieveDocumentsUseCase
from rag_core.domain.interfaces.chunker_port import ChunkerPort
from rag_core.domain.interfaces.embedding_port import EmbeddingPort
from rag_core.domain.interfaces.llm_port import LLMPort
from rag_core.domain.interfaces.vectordb_port import VectorDBPort

from rag_infra.config.settings import (
    ChunkerType,
    EmbeddingProvider,
    LLMProvider,
    Settings,
    VectorDBProvider,
)


@dataclass
class AdvancedRAGPipeline:
    """Simplified facade for the Advanced RAG Pipeline.

    This class provides a clean, easy-to-use interface for common RAG
    operations like querying and document ingestion. It handles all the
    internal wiring of adapters and services.

    Usage:
        # Default configuration from environment
        pipeline = AdvancedRAGPipeline()
        result = await pipeline.query("What is machine learning?")

        # Custom adapters
        from rag_infra.llm import VLLMAdapter
        pipeline = AdvancedRAGPipeline(
            llm_adapter=VLLMAdapter(base_url="http://gpu:8000/v1", model="llama-70b")
        )
    """

    # Optional custom adapters (if not provided, created from settings)
    llm_adapter: LLMPort | None = None
    embedding_adapter: EmbeddingPort | None = None
    vectordb_adapter: VectorDBPort | None = None
    chunker_adapter: ChunkerPort | None = None

    # Settings (loaded from environment if not provided)
    settings: Settings | None = None

    # Internal services (created on first use)
    _pipeline_service: PipelineService | None = field(default=None, repr=False)
    _ingest_service: IngestService | None = field(default=None, repr=False)
    _query_service: QueryService | None = field(default=None, repr=False)

    def __post_init__(self) -> None:
        """Initialize the facade."""
        if self.settings is None:
            self.settings = Settings.load()

    def _create_llm_adapter(self) -> LLMPort:
        """Create LLM adapter from settings."""
        if self.llm_adapter is not None:
            return self.llm_adapter

        assert self.settings is not None
        llm_settings = self.settings.llm

        if llm_settings.provider == LLMProvider.OPENAI:
            from rag_infra.llm import OpenAIAdapter

            return OpenAIAdapter(
                api_key=llm_settings.api_key,
                model=llm_settings.model,
                base_url=llm_settings.base_url or None,
            )
        elif llm_settings.provider == LLMProvider.VLLM:
            from rag_infra.llm import VLLMAdapter

            return VLLMAdapter(
                base_url=llm_settings.effective_base_url,
                model=llm_settings.model,
                api_key=llm_settings.api_key,
            )
        elif llm_settings.provider == LLMProvider.OLLAMA:
            from rag_infra.llm import OllamaAdapter

            return OllamaAdapter(
                base_url=llm_settings.effective_base_url,
                model=llm_settings.model,
            )
        else:
            raise ValueError(f"Unsupported LLM provider: {llm_settings.provider}")

    def _create_embedding_adapter(self) -> EmbeddingPort:
        """Create embedding adapter from settings."""
        if self.embedding_adapter is not None:
            return self.embedding_adapter

        assert self.settings is not None
        emb_settings = self.settings.embedding

        if emb_settings.provider == EmbeddingProvider.OPENAI:
            from rag_infra.embedding import OpenAIEmbeddingAdapter

            return OpenAIEmbeddingAdapter(
                api_key=emb_settings.api_key,
                model=emb_settings.model,
                base_url=emb_settings.base_url or None,
                dimension=emb_settings.dimension,
            )
        elif emb_settings.provider == EmbeddingProvider.INFINITY:
            from rag_infra.embedding import InfinityAdapter

            return InfinityAdapter(
                base_url=emb_settings.effective_base_url,
                model=emb_settings.model,
                dimension=emb_settings.dimension,
            )
        elif emb_settings.provider == EmbeddingProvider.SENTENCE_TRANSFORMERS:
            from rag_infra.embedding import SentenceTransformerAdapter

            return SentenceTransformerAdapter(
                model=emb_settings.model,
            )
        else:
            raise ValueError(f"Unsupported embedding provider: {emb_settings.provider}")

    def _create_vectordb_adapter(self) -> VectorDBPort:
        """Create vector database adapter from settings."""
        if self.vectordb_adapter is not None:
            return self.vectordb_adapter

        assert self.settings is not None
        vdb_settings = self.settings.vectordb

        if vdb_settings.provider == VectorDBProvider.WEAVIATE:
            from rag_infra.vectordb import WeaviateAdapter

            return WeaviateAdapter(
                host=vdb_settings.host,
                port=vdb_settings.port,
                api_key=vdb_settings.api_key or None,
                default_collection=vdb_settings.collection_name,
            )
        elif vdb_settings.provider == VectorDBProvider.CHROMA:
            from rag_infra.vectordb import ChromaAdapter

            return ChromaAdapter(
                host=vdb_settings.host,
                port=vdb_settings.port,
                default_collection=vdb_settings.collection_name,
            )
        else:
            raise ValueError(f"Unsupported vector DB provider: {vdb_settings.provider}")

    def _create_chunker_adapter(self) -> ChunkerPort:
        """Create chunker adapter from settings."""
        if self.chunker_adapter is not None:
            return self.chunker_adapter

        assert self.settings is not None
        chunker_settings = self.settings.chunker

        if chunker_settings.type == ChunkerType.SEMANTIC:
            from rag_infra.chunking import SemanticChunker

            # Optionally use embedding adapter for semantic chunking
            embedding_adapter = (
                self._create_embedding_adapter()
                if chunker_settings.embedding_model
                else None
            )
            return SemanticChunker(embedding_port=embedding_adapter)
        elif chunker_settings.type == ChunkerType.FIXED:
            from rag_infra.chunking import FixedChunker

            return FixedChunker()
        else:
            raise ValueError(f"Unsupported chunker type: {chunker_settings.type}")

    def _get_pipeline_service(self) -> PipelineService:
        """Get or create the pipeline service."""
        if self._pipeline_service is None:
            llm = self._create_llm_adapter()
            embedding = self._create_embedding_adapter()
            vectordb = self._create_vectordb_adapter()

            self._pipeline_service = PipelineService(
                query_rewrite_use_case=QueryRewriteUseCase(llm=llm),
                retrieve_documents_use_case=RetrieveDocumentsUseCase(
                    embedding_port=embedding,
                    vectordb_port=vectordb,
                ),
                generate_answer_use_case=GenerateAnswerUseCase(llm=llm),
            )
        return self._pipeline_service

    def _get_ingest_service(self) -> IngestService:
        """Get or create the ingest service."""
        if self._ingest_service is None:
            embedding = self._create_embedding_adapter()
            vectordb = self._create_vectordb_adapter()
            chunker = self._create_chunker_adapter()

            self._ingest_service = IngestService(
                ingest_use_case=IngestDocumentUseCase(
                    chunker=chunker,
                    embedding_port=embedding,
                    vectordb_port=vectordb,
                )
            )
        return self._ingest_service

    def _get_query_service(self) -> QueryService:
        """Get or create the query service."""
        if self._query_service is None:
            embedding = self._create_embedding_adapter()
            vectordb = self._create_vectordb_adapter()

            self._query_service = QueryService(
                retrieve_use_case=RetrieveDocumentsUseCase(
                    embedding_port=embedding,
                    vectordb_port=vectordb,
                )
            )
        return self._query_service

    # Public API

    async def query(
        self,
        question: str,
        top_k: int = 5,
        rewrite_query: bool = True,
        collection_name: str | None = None,
        score_threshold: float | None = None,
        filter_metadata: dict[str, Any] | None = None,
    ) -> QueryResponse:
        """Query the RAG pipeline with a question.

        Args:
            question: The question to answer.
            top_k: Number of relevant chunks to retrieve.
            rewrite_query: Whether to rewrite the query for better retrieval.
            collection_name: Optional collection name to search.
            score_threshold: Optional minimum relevance score.
            filter_metadata: Optional metadata filters.

        Returns:
            QueryResponse with the generated answer and sources.
        """
        request = QueryRequest(
            query=question,
            top_k=top_k,
            rewrite_query=rewrite_query,
            collection_name=collection_name,
            score_threshold=score_threshold,
            filter_metadata=filter_metadata,
        )
        return await self._get_pipeline_service().execute(request)

    async def query_stream(
        self,
        question: str,
        top_k: int = 5,
        rewrite_query: bool = True,
        collection_name: str | None = None,
    ) -> AsyncIterator[str]:
        """Query the RAG pipeline with streaming response.

        Args:
            question: The question to answer.
            top_k: Number of relevant chunks to retrieve.
            rewrite_query: Whether to rewrite the query.
            collection_name: Optional collection name.

        Yields:
            String chunks of the generated answer.
        """
        request = QueryRequest(
            query=question,
            top_k=top_k,
            rewrite_query=rewrite_query,
            collection_name=collection_name,
        )
        async for chunk in self._get_pipeline_service().execute_streaming(request):
            yield chunk

    async def search(
        self,
        query: str,
        top_k: int = 5,
        collection_name: str | None = None,
        score_threshold: float | None = None,
    ) -> list[SourceDTO]:
        """Search for relevant documents without generating an answer.

        Args:
            query: The search query.
            top_k: Number of results to return.
            collection_name: Optional collection name.
            score_threshold: Optional minimum score.

        Returns:
            List of relevant source chunks.
        """
        request = QueryRequest(
            query=query,
            top_k=top_k,
            rewrite_query=False,
            collection_name=collection_name,
            score_threshold=score_threshold,
        )
        return await self._get_pipeline_service().retrieve_only(request)

    async def ingest(
        self,
        content: str,
        source: str,
        collection_name: str | None = None,
        chunk_size: int | None = None,
        chunk_overlap: int | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> IngestResponse:
        """Ingest a document into the RAG pipeline.

        Args:
            content: The document content.
            source: Source identifier (e.g., filename, URL).
            collection_name: Optional collection name.
            chunk_size: Optional chunk size (uses settings if not specified).
            chunk_overlap: Optional chunk overlap.
            metadata: Optional document metadata.

        Returns:
            IngestResponse with ingestion details.
        """
        assert self.settings is not None
        request = IngestRequest(
            content=content,
            source=source,
            collection_name=collection_name,
            chunk_size=chunk_size or self.settings.chunk.size,
            chunk_overlap=chunk_overlap or self.settings.chunk.overlap,
            metadata=metadata or {},
        )
        return await self._get_ingest_service().ingest(request)

    async def ingest_batch(
        self,
        documents: list[tuple[str, str, dict[str, Any] | None]],
        collection_name: str | None = None,
    ) -> list[IngestResponse]:
        """Ingest multiple documents.

        Args:
            documents: List of (content, source, metadata) tuples.
            collection_name: Optional collection name.

        Returns:
            List of IngestResponse for each document.
        """
        assert self.settings is not None
        return await self._get_ingest_service().ingest_batch(
            documents=documents,
            collection_name=collection_name,
            chunk_size=self.settings.chunk.size,
            chunk_overlap=self.settings.chunk.overlap,
        )

    async def delete_document(
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
        return await self._get_ingest_service().delete(document_id, collection_name)

    async def get_context(
        self,
        query: str,
        top_k: int = 5,
        collection_name: str | None = None,
    ) -> str:
        """Get relevant context for a query without generating an answer.

        Useful for custom prompting or debugging.

        Args:
            query: The query to get context for.
            top_k: Number of chunks to retrieve.
            collection_name: Optional collection name.

        Returns:
            Combined context string from relevant chunks.
        """
        return await self._get_query_service().get_context(
            query=query,
            top_k=top_k,
            collection_name=collection_name,
        )

    async def health_check(self) -> dict[str, bool]:
        """Check the health of all connected services.

        Returns:
            Dictionary with health status for each service.
        """
        results: dict[str, bool] = {}

        try:
            llm = self._create_llm_adapter()
            results["llm"] = await llm.health_check()
        except Exception:
            results["llm"] = False

        try:
            embedding = self._create_embedding_adapter()
            results["embedding"] = await embedding.health_check()
        except Exception:
            results["embedding"] = False

        try:
            vectordb = self._create_vectordb_adapter()
            results["vectordb"] = await vectordb.health_check()
        except Exception:
            results["vectordb"] = False

        return results
