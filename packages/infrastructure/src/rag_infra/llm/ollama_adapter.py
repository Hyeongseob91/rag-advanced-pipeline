"""Ollama LLM Adapter for local models."""

from collections.abc import AsyncIterator
import json

import httpx

from rag_core.domain.interfaces.llm_port import LLMPort, LLMResponse, Message


class OllamaAdapter(LLMPort):
    """Ollama API adapter implementing LLMPort.

    This adapter connects to a local Ollama server for running
    open-source models locally without GPU requirements.
    """

    def __init__(
        self,
        base_url: str = "http://localhost:11434",
        model: str = "llama3.1:8b",
        timeout: float = 120.0,
    ):
        """Initialize the Ollama adapter.

        Args:
            base_url: Ollama server URL.
            model: Model name to use.
            timeout: Request timeout in seconds.
        """
        self._base_url = base_url.rstrip("/")
        self._model = model
        self._timeout = timeout
        self._client = httpx.AsyncClient(timeout=timeout)

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
        """Generate a response using Ollama.

        Args:
            prompt: The user prompt.
            system_prompt: Optional system prompt.
            temperature: Sampling temperature.
            max_tokens: Maximum tokens in response.
            stop_sequences: Optional stop sequences.

        Returns:
            LLMResponse with the generated content.
        """
        payload = {
            "model": self._model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": temperature,
                "num_predict": max_tokens,
            },
        }

        if system_prompt:
            payload["system"] = system_prompt

        if stop_sequences:
            payload["options"]["stop"] = stop_sequences

        response = await self._client.post(
            f"{self._base_url}/api/generate",
            json=payload,
        )
        response.raise_for_status()
        data = response.json()

        return LLMResponse(
            content=data.get("response", ""),
            model=data.get("model", self._model),
            prompt_tokens=data.get("prompt_eval_count", 0),
            completion_tokens=data.get("eval_count", 0),
            total_tokens=(
                data.get("prompt_eval_count", 0) + data.get("eval_count", 0)
            ),
            finish_reason="stop" if data.get("done", False) else "",
        )

    async def generate_stream(
        self,
        prompt: str,
        system_prompt: str = "",
        temperature: float = 0.0,
        max_tokens: int = 2000,
        stop_sequences: list[str] | None = None,
    ) -> AsyncIterator[str]:
        """Generate a streaming response using Ollama.

        Args:
            prompt: The user prompt.
            system_prompt: Optional system prompt.
            temperature: Sampling temperature.
            max_tokens: Maximum tokens in response.
            stop_sequences: Optional stop sequences.

        Yields:
            String chunks of the response.
        """
        payload = {
            "model": self._model,
            "prompt": prompt,
            "stream": True,
            "options": {
                "temperature": temperature,
                "num_predict": max_tokens,
            },
        }

        if system_prompt:
            payload["system"] = system_prompt

        if stop_sequences:
            payload["options"]["stop"] = stop_sequences

        async with self._client.stream(
            "POST",
            f"{self._base_url}/api/generate",
            json=payload,
        ) as response:
            response.raise_for_status()
            async for line in response.aiter_lines():
                if line:
                    try:
                        data = json.loads(line)
                        if "response" in data:
                            yield data["response"]
                        if data.get("done", False):
                            break
                    except json.JSONDecodeError:
                        continue

    async def generate_with_messages(
        self,
        messages: list[Message],
        temperature: float = 0.0,
        max_tokens: int = 2000,
        stop_sequences: list[str] | None = None,
    ) -> LLMResponse:
        """Generate a response from a list of messages using Ollama chat API.

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
            "stream": False,
            "options": {
                "temperature": temperature,
                "num_predict": max_tokens,
            },
        }

        if stop_sequences:
            payload["options"]["stop"] = stop_sequences

        response = await self._client.post(
            f"{self._base_url}/api/chat",
            json=payload,
        )
        response.raise_for_status()
        data = response.json()

        message = data.get("message", {})

        return LLMResponse(
            content=message.get("content", ""),
            model=data.get("model", self._model),
            prompt_tokens=data.get("prompt_eval_count", 0),
            completion_tokens=data.get("eval_count", 0),
            total_tokens=(
                data.get("prompt_eval_count", 0) + data.get("eval_count", 0)
            ),
            finish_reason="stop" if data.get("done", False) else "",
        )

    async def list_models(self) -> list[str]:
        """List available models in Ollama.

        Returns:
            List of model names.
        """
        response = await self._client.get(f"{self._base_url}/api/tags")
        response.raise_for_status()
        data = response.json()
        return [model["name"] for model in data.get("models", [])]

    async def close(self) -> None:
        """Close the HTTP client."""
        await self._client.aclose()
