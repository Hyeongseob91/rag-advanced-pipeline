"""Settings configuration using Pydantic."""

from enum import Enum
from typing import Any

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class LLMProvider(str, Enum):
    """Supported LLM providers."""

    OPENAI = "openai"
    VLLM = "vllm"
    OLLAMA = "ollama"


class EmbeddingProvider(str, Enum):
    """Supported embedding providers."""

    OPENAI = "openai"
    INFINITY = "infinity"
    SENTENCE_TRANSFORMERS = "sentence_transformers"


class VectorDBProvider(str, Enum):
    """Supported vector database providers."""

    WEAVIATE = "weaviate"
    CHROMA = "chroma"


class ChunkerType(str, Enum):
    """Supported chunking strategies."""

    SEMANTIC = "semantic"
    FIXED = "fixed"


class LLMSettings(BaseSettings):
    """LLM configuration settings."""

    model_config = SettingsConfigDict(env_prefix="LLM_", extra="ignore")

    provider: LLMProvider = Field(default=LLMProvider.OPENAI)
    model: str = Field(default="gpt-4o")
    api_key: str = Field(default="")
    base_url: str = Field(default="")
    temperature: float = Field(default=0.0, ge=0.0, le=2.0)
    max_tokens: int = Field(default=2000, gt=0)

    @property
    def effective_base_url(self) -> str:
        """Get the effective base URL based on provider."""
        if self.base_url:
            return self.base_url
        if self.provider == LLMProvider.OPENAI:
            return "https://api.openai.com/v1"
        if self.provider == LLMProvider.OLLAMA:
            return "http://localhost:11434"
        if self.provider == LLMProvider.VLLM:
            return "http://localhost:8000/v1"
        return ""


class EmbeddingSettings(BaseSettings):
    """Embedding configuration settings."""

    model_config = SettingsConfigDict(env_prefix="EMBEDDING_", extra="ignore")

    provider: EmbeddingProvider = Field(default=EmbeddingProvider.OPENAI)
    model: str = Field(default="text-embedding-3-small")
    api_key: str = Field(default="")
    base_url: str = Field(default="")
    dimension: int = Field(default=1536, gt=0)

    @property
    def effective_base_url(self) -> str:
        """Get the effective base URL based on provider."""
        if self.base_url:
            return self.base_url
        if self.provider == EmbeddingProvider.OPENAI:
            return "https://api.openai.com/v1"
        if self.provider == EmbeddingProvider.INFINITY:
            return "http://localhost:7997"
        return ""


class VectorDBSettings(BaseSettings):
    """Vector database configuration settings."""

    model_config = SettingsConfigDict(env_prefix="VECTORDB_", extra="ignore")

    provider: VectorDBProvider = Field(default=VectorDBProvider.WEAVIATE)
    host: str = Field(default="localhost")
    port: int = Field(default=8080, gt=0)
    api_key: str = Field(default="")
    collection_name: str = Field(default="documents")

    @property
    def url(self) -> str:
        """Get the full URL for the vector database."""
        return f"http://{self.host}:{self.port}"


class ChunkerSettings(BaseSettings):
    """Chunking configuration settings."""

    model_config = SettingsConfigDict(env_prefix="CHUNKER_", extra="ignore")

    type: ChunkerType = Field(default=ChunkerType.SEMANTIC)
    embedding_model: str = Field(default="")


class ChunkSettings(BaseSettings):
    """Chunk size configuration settings."""

    model_config = SettingsConfigDict(env_prefix="CHUNK_", extra="ignore")

    size: int = Field(default=512, gt=0)
    overlap: int = Field(default=50, ge=0)


class RetrievalSettings(BaseSettings):
    """Retrieval configuration settings."""

    model_config = SettingsConfigDict(env_prefix="RETRIEVAL_", extra="ignore")

    top_k: int = Field(default=5, gt=0)
    score_threshold: float = Field(default=0.7, ge=0.0, le=1.0)


class Settings(BaseSettings):
    """Main settings class aggregating all configuration."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # Sub-settings
    llm: LLMSettings = Field(default_factory=LLMSettings)
    embedding: EmbeddingSettings = Field(default_factory=EmbeddingSettings)
    vectordb: VectorDBSettings = Field(default_factory=VectorDBSettings)
    chunker: ChunkerSettings = Field(default_factory=ChunkerSettings)
    chunk: ChunkSettings = Field(default_factory=ChunkSettings)
    retrieval: RetrievalSettings = Field(default_factory=RetrievalSettings)

    # Logging
    log_level: str = Field(default="INFO")

    @classmethod
    def load(cls) -> "Settings":
        """Load settings from environment."""
        return cls(
            llm=LLMSettings(),
            embedding=EmbeddingSettings(),
            vectordb=VectorDBSettings(),
            chunker=ChunkerSettings(),
            chunk=ChunkSettings(),
            retrieval=RetrievalSettings(),
        )
