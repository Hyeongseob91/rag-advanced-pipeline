"""Generation DTOs for the application layer."""

from dataclasses import dataclass, field
from typing import Any


@dataclass
class GenerationRequest:
    """Request DTO for answer generation."""

    query: str
    context: str
    system_prompt: str | None = None
    temperature: float = 0.0
    max_tokens: int = 2000
    include_sources: bool = True
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.query or not self.query.strip():
            raise ValueError("Query cannot be empty")
        if not self.context or not self.context.strip():
            raise ValueError("Context cannot be empty")
        if self.temperature < 0 or self.temperature > 2:
            raise ValueError("Temperature must be between 0 and 2")
        if self.max_tokens <= 0:
            raise ValueError("max_tokens must be positive")


@dataclass
class GenerationResponse:
    """Response DTO for answer generation."""

    answer: str
    model: str
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    generation_time_ms: float = 0.0
    finish_reason: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.total_tokens == 0 and (self.prompt_tokens > 0 or self.completion_tokens > 0):
            object.__setattr__(
                self, "total_tokens", self.prompt_tokens + self.completion_tokens
            )


@dataclass
class IngestRequest:
    """Request DTO for document ingestion."""

    content: str
    source: str
    collection_name: str | None = None
    chunk_size: int = 512
    chunk_overlap: int = 50
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.content or not self.content.strip():
            raise ValueError("Content cannot be empty")
        if not self.source or not self.source.strip():
            raise ValueError("Source cannot be empty")
        if self.chunk_size <= 0:
            raise ValueError("chunk_size must be positive")
        if self.chunk_overlap < 0:
            raise ValueError("chunk_overlap cannot be negative")


@dataclass
class IngestResponse:
    """Response DTO for document ingestion."""

    document_id: str
    chunk_count: int
    total_characters: int
    collection_name: str
    processing_time_ms: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)
