"""Infinity Embedding Adapter for GPU-accelerated embeddings."""

import httpx

from rag_core.domain.interfaces.embedding_port import EmbeddingPort, EmbeddingResult


class InfinityAdapter(EmbeddingPort):
    """Infinity embedding server adapter implementing EmbeddingPort.

    Infinity is a high-performance embedding server that supports
    various embedding models with GPU acceleration.
    """

    def __init__(
        self,
        base_url: str = "http://localhost:7997",
        model: str = "BAAI/bge-m3",
        dimension: int = 1024,
        timeout: float = 60.0,
    ):
        """Initialize the Infinity adapter.

        Args:
            base_url: Infinity server URL.
            model: Model name to use.
            dimension: Embedding dimension.
            timeout: Request timeout in seconds.
        """
        self._base_url = base_url.rstrip("/")
        self._model = model
        self._dimension = dimension
        self._timeout = timeout
        self._client = httpx.AsyncClient(timeout=timeout)

    @property
    def model_name(self) -> str:
        """Return the model name."""
        return self._model

    @property
    def dimension(self) -> int:
        """Return the embedding dimension."""
        return self._dimension

    async def embed(self, text: str) -> list[float]:
        """Generate an embedding for a single text.

        Args:
            text: The text to embed.

        Returns:
            Embedding vector as a list of floats.
        """
        response = await self._client.post(
            f"{self._base_url}/embeddings",
            json={
                "model": self._model,
                "input": text,
            },
        )
        response.raise_for_status()
        data = response.json()
        return data["data"][0]["embedding"]

    async def embed_batch(self, texts: list[str]) -> EmbeddingResult:
        """Generate embeddings for multiple texts.

        Args:
            texts: List of texts to embed.

        Returns:
            EmbeddingResult with all embeddings.
        """
        if not texts:
            return EmbeddingResult(embeddings=[], model=self._model)

        response = await self._client.post(
            f"{self._base_url}/embeddings",
            json={
                "model": self._model,
                "input": texts,
            },
        )
        response.raise_for_status()
        data = response.json()

        # Sort by index to maintain order
        sorted_data = sorted(data["data"], key=lambda x: x["index"])
        embeddings = [item["embedding"] for item in sorted_data]

        return EmbeddingResult(
            embeddings=embeddings,
            model=data.get("model", self._model),
            total_tokens=data.get("usage", {}).get("total_tokens", 0),
        )

    async def health_check(self) -> bool:
        """Check if the Infinity server is healthy.

        Returns:
            True if healthy, False otherwise.
        """
        try:
            response = await self._client.get(f"{self._base_url}/health")
            return response.status_code == 200
        except Exception:
            return False

    async def close(self) -> None:
        """Close the HTTP client."""
        await self._client.aclose()
