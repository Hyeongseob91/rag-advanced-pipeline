"""LLM Port - Abstract interface for LLM adapters."""

from abc import ABC, abstractmethod
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from typing import Any


@dataclass
class LLMResponse:
    """Response from an LLM generation call."""

    content: str
    model: str = ""
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    finish_reason: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.total_tokens == 0 and (self.prompt_tokens > 0 or self.completion_tokens > 0):
            self.total_tokens = self.prompt_tokens + self.completion_tokens


@dataclass
class Message:
    """A message in a conversation."""

    role: str  # "system", "user", "assistant"
    content: str

    def __post_init__(self) -> None:
        if self.role not in ("system", "user", "assistant"):
            raise ValueError(f"Invalid role: {self.role}")
        if not self.content:
            raise ValueError("Message content cannot be empty")

    def to_dict(self) -> dict[str, str]:
        """Convert to dictionary format for API calls."""
        return {"role": self.role, "content": self.content}


class LLMPort(ABC):
    """Abstract interface for LLM adapters.

    This port defines the contract that all LLM implementations must follow.
    Implementations include OpenAI, vLLM, Ollama, etc.
    """

    @abstractmethod
    async def generate(
        self,
        prompt: str,
        system_prompt: str = "",
        temperature: float = 0.0,
        max_tokens: int = 2000,
        stop_sequences: list[str] | None = None,
    ) -> LLMResponse:
        """Generate a response from the LLM.

        Args:
            prompt: The user prompt to generate a response for.
            system_prompt: Optional system prompt to set context.
            temperature: Sampling temperature (0.0 = deterministic).
            max_tokens: Maximum tokens in the response.
            stop_sequences: Optional list of sequences to stop generation.

        Returns:
            LLMResponse containing the generated content and metadata.
        """
        pass

    @abstractmethod
    async def generate_stream(
        self,
        prompt: str,
        system_prompt: str = "",
        temperature: float = 0.0,
        max_tokens: int = 2000,
        stop_sequences: list[str] | None = None,
    ) -> AsyncIterator[str]:
        """Generate a streaming response from the LLM.

        Args:
            prompt: The user prompt to generate a response for.
            system_prompt: Optional system prompt to set context.
            temperature: Sampling temperature (0.0 = deterministic).
            max_tokens: Maximum tokens in the response.
            stop_sequences: Optional list of sequences to stop generation.

        Yields:
            String chunks of the generated response.
        """
        pass

    @abstractmethod
    async def generate_with_messages(
        self,
        messages: list[Message],
        temperature: float = 0.0,
        max_tokens: int = 2000,
        stop_sequences: list[str] | None = None,
    ) -> LLMResponse:
        """Generate a response from a list of messages.

        Args:
            messages: List of Message objects representing the conversation.
            temperature: Sampling temperature (0.0 = deterministic).
            max_tokens: Maximum tokens in the response.
            stop_sequences: Optional list of sequences to stop generation.

        Returns:
            LLMResponse containing the generated content and metadata.
        """
        pass

    @property
    @abstractmethod
    def model_name(self) -> str:
        """Return the model name/identifier."""
        pass

    async def health_check(self) -> bool:
        """Check if the LLM service is healthy.

        Returns:
            True if the service is available, False otherwise.
        """
        try:
            response = await self.generate("Hello", max_tokens=5)
            return bool(response.content)
        except Exception:
            return False
