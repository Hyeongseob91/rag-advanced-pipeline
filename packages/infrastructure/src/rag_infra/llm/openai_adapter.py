"""OpenAI LLM Adapter."""

from collections.abc import AsyncIterator
from typing import Any

from openai import AsyncOpenAI

from rag_core.domain.interfaces.llm_port import LLMPort, LLMResponse, Message


class OpenAIAdapter(LLMPort):
    """OpenAI API adapter implementing LLMPort.

    This adapter supports OpenAI's chat completion API including
    GPT-4, GPT-4o, GPT-3.5-turbo, and other compatible models.
    """

    def __init__(
        self,
        api_key: str,
        model: str = "gpt-4o",
        base_url: str | None = None,
    ):
        """Initialize the OpenAI adapter.

        Args:
            api_key: OpenAI API key.
            model: Model name to use.
            base_url: Optional custom base URL (for Azure or proxies).
        """
        self._model = model
        self._client = AsyncOpenAI(
            api_key=api_key,
            base_url=base_url,
        )

    @property
    def model_name(self) -> str:
        """Return the model name."""
        return self._model

    async def generate(
        self,
        prompt: str,
        system_prompt: str = "",
        temperature: float = 0.0,
        max_tokens: int = 2000,
        stop_sequences: list[str] | None = None,
    ) -> LLMResponse:
        """Generate a response using OpenAI API.

        Args:
            prompt: The user prompt.
            system_prompt: Optional system prompt.
            temperature: Sampling temperature.
            max_tokens: Maximum tokens in response.
            stop_sequences: Optional stop sequences.

        Returns:
            LLMResponse with the generated content.
        """
        messages: list[dict[str, str]] = []

        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        response = await self._client.chat.completions.create(
            model=self._model,
            messages=messages,  # type: ignore
            temperature=temperature,
            max_tokens=max_tokens,
            stop=stop_sequences,
        )

        choice = response.choices[0]
        usage = response.usage

        return LLMResponse(
            content=choice.message.content or "",
            model=response.model,
            prompt_tokens=usage.prompt_tokens if usage else 0,
            completion_tokens=usage.completion_tokens if usage else 0,
            total_tokens=usage.total_tokens if usage else 0,
            finish_reason=choice.finish_reason or "",
        )

    async def generate_stream(
        self,
        prompt: str,
        system_prompt: str = "",
        temperature: float = 0.0,
        max_tokens: int = 2000,
        stop_sequences: list[str] | None = None,
    ) -> AsyncIterator[str]:
        """Generate a streaming response using OpenAI API.

        Args:
            prompt: The user prompt.
            system_prompt: Optional system prompt.
            temperature: Sampling temperature.
            max_tokens: Maximum tokens in response.
            stop_sequences: Optional stop sequences.

        Yields:
            String chunks of the response.
        """
        messages: list[dict[str, str]] = []

        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        stream = await self._client.chat.completions.create(
            model=self._model,
            messages=messages,  # type: ignore
            temperature=temperature,
            max_tokens=max_tokens,
            stop=stop_sequences,
            stream=True,
        )

        async for chunk in stream:
            if chunk.choices and chunk.choices[0].delta.content:
                yield chunk.choices[0].delta.content

    async def generate_with_messages(
        self,
        messages: list[Message],
        temperature: float = 0.0,
        max_tokens: int = 2000,
        stop_sequences: list[str] | None = None,
    ) -> LLMResponse:
        """Generate a response from a list of messages.

        Args:
            messages: List of Message objects.
            temperature: Sampling temperature.
            max_tokens: Maximum tokens in response.
            stop_sequences: Optional stop sequences.

        Returns:
            LLMResponse with the generated content.
        """
        message_dicts = [msg.to_dict() for msg in messages]

        response = await self._client.chat.completions.create(
            model=self._model,
            messages=message_dicts,  # type: ignore
            temperature=temperature,
            max_tokens=max_tokens,
            stop=stop_sequences,
        )

        choice = response.choices[0]
        usage = response.usage

        return LLMResponse(
            content=choice.message.content or "",
            model=response.model,
            prompt_tokens=usage.prompt_tokens if usage else 0,
            completion_tokens=usage.completion_tokens if usage else 0,
            total_tokens=usage.total_tokens if usage else 0,
            finish_reason=choice.finish_reason or "",
        )
