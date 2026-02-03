"""
RAG Core Package - Domain and Application Layer

This package contains the core business logic with no external dependencies.
"""

from rag_core.domain.entities.document import Chunk, Document
from rag_core.domain.entities.query import Query, RewrittenQuery
from rag_core.domain.entities.response import GeneratedResponse, RetrievalResult
from rag_core.domain.interfaces.chunker_port import ChunkerPort
from rag_core.domain.interfaces.embedding_port import EmbeddingPort
from rag_core.domain.interfaces.llm_port import LLMPort, LLMResponse
from rag_core.domain.interfaces.vectordb_port import VectorDBPort
from rag_core.domain.value_objects.embedding import Embedding
from rag_core.domain.value_objects.score import SimilarityScore

__all__ = [
    # Entities
    "Document",
    "Chunk",
    "Query",
    "RewrittenQuery",
    "GeneratedResponse",
    "RetrievalResult",
    # Value Objects
    "Embedding",
    "SimilarityScore",
    # Interfaces (Ports)
    "LLMPort",
    "LLMResponse",
    "EmbeddingPort",
    "VectorDBPort",
    "ChunkerPort",
]
