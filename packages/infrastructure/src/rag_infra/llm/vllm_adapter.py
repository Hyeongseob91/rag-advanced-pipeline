"""vLLM Adapter for GPU-accelerated inference."""

from collections.abc import AsyncIterator
from typing import Any

import httpx

from rag_core.domain.interfaces.llm_port import LLMPort, LLMResponse, Message


class VLLMAdapter(LLMPort):
    """vLLM API adapter implementing LLMPort.

    This adapter connects to a vLLM server for GPU-accelerated
    inference of open-source models like Llama, Mistral, etc.
    The vLLM server exposes an OpenAI-compatible API.
    """

    def __init__(
        self,
        base_url: str = "http://localhost:8000/v1",
        model: str = "default",
        api_key: str = "",
        timeout: float = 120.0,
    ):
        """Initialize the vLLM adapter.

        Args:
            base_url: vLLM server URL.
            model: Model name (as served by vLLM).
            api_key: Optional API key.
            timeout: Request timeout in seconds.
        """
        self._base_url = base_url.rstrip("/")
        self._model = model
        self._api_key = api_key
        self._timeout = timeout
        self._client = httpx.AsyncClient(timeout=timeout)

    @property
    def model_name(self) -> str:
        """Return the model name."""
        return self._model

    def _get_headers(self) -> dict[str, str]:
        """Get request headers."""
        headers = {"Content-Type": "application/json"}
        if self._api_key:
            headers["Authorization"] = f"Bearer {self._api_key}"
        return headers

    async def generate(
        self,
        prompt: str,
        system_prompt: str = "",
        temperature: float = 0.0,
        max_tokens: int = 2000,
        stop_sequences: list[str] | None = None,
    ) -> LLMResponse:
        """Generate a response using vLLM server.

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

        payload = {
            "model": self._model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }

        if stop_sequences:
            payload["stop"] = stop_sequences

        response = await self._client.post(
            f"{self._base_url}/chat/completions",
            headers=self._get_headers(),
            json=payload,
        )
        response.raise_for_status()
        data = response.json()

        choice = data["choices"][0]
        usage = data.get("usage", {})

        return LLMResponse(
            content=choice["message"]["content"],
            model=data.get("model", self._model),
            prompt_tokens=usage.get("prompt_tokens", 0),
            completion_tokens=usage.get("completion_tokens", 0),
            total_tokens=usage.get("total_tokens", 0),
            finish_reason=choice.get("finish_reason", ""),
        )

    async def generate_stream(
        self,
        prompt: str,
        system_prompt: str = "",
        temperature: float = 0.0,
        max_tokens: int = 2000,
        stop_sequences: list[str] | None = None,
    ) -> AsyncIterator[str]:
        """Generate a streaming response using vLLM server.

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

        payload = {
            "model": self._model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": True,
        }

        if stop_sequences:
            payload["stop"] = stop_sequences

        async with self._client.stream(
            "POST",
            f"{self._base_url}/chat/completions",
            headers=self._get_headers(),
            json=payload,
        ) as response:
            response.raise_for_status()
            async for line in response.aiter_lines():
                if line.startswith("data: "):
                    data_str = line[6:]
                    if data_str.strip() == "[DONE]":
                        break
                    import json

                    try:
                        data = json.loads(data_str)
                        if data["choices"] and data["choices"][0]["delta"].get("content"):
                            yield data["choices"][0]["delta"]["content"]
                    except json.JSONDecodeError:
                        continue

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

        payload = {
            "model": self._model,
            "messages": message_dicts,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }

        if stop_sequences:
            payload["stop"] = stop_sequences

        response = await self._client.post(
            f"{self._base_url}/chat/completions",
            headers=self._get_headers(),
            json=payload,
        )
        response.raise_for_status()
        data = response.json()

        choice = data["choices"][0]
        usage = data.get("usage", {})

        return LLMResponse(
            content=choice["message"]["content"],
            model=data.get("model", self._model),
            prompt_tokens=usage.get("prompt_tokens", 0),
            completion_tokens=usage.get("completion_tokens", 0),
            total_tokens=usage.get("total_tokens", 0),
            finish_reason=choice.get("finish_reason", ""),
        )

    async def close(self) -> None:
        """Close the HTTP client."""
        await self._client.aclose()
