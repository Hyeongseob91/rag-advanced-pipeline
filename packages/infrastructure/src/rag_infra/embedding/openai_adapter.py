"""OpenAI Embedding Adapter."""

from openai import AsyncOpenAI

from rag_core.domain.interfaces.embedding_port import EmbeddingPort, EmbeddingResult


# Known dimensions for OpenAI models
OPENAI_EMBEDDING_DIMENSIONS = {
    "text-embedding-3-small": 1536,
    "text-embedding-3-large": 3072,
    "text-embedding-ada-002": 1536,
}


class OpenAIEmbeddingAdapter(EmbeddingPort):
    """OpenAI Embedding API adapter implementing EmbeddingPort.

    Supports text-embedding-3-small, text-embedding-3-large,
    and text-embedding-ada-002 models.
    """

    def __init__(
        self,
        api_key: str,
        model: str = "text-embedding-3-small",
        base_url: str | None = None,
        dimension: int | None = None,
    ):
        """Initialize the OpenAI embedding adapter.

        Args:
            api_key: OpenAI API key.
            model: Embedding model name.
            base_url: Optional custom base URL.
            dimension: Optional custom dimension (for dimension reduction).
        """
        self._model = model
        self._dimension = dimension or OPENAI_EMBEDDING_DIMENSIONS.get(model, 1536)
        self._client = AsyncOpenAI(
            api_key=api_key,
            base_url=base_url,
        )

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
        response = await self._client.embeddings.create(
            model=self._model,
            input=text,
            dimensions=self._dimension if "3-" in self._model else None,
        )
        return response.data[0].embedding

    async def embed_batch(self, texts: list[str]) -> EmbeddingResult:
        """Generate embeddings for multiple texts.

        Args:
            texts: List of texts to embed.

        Returns:
            EmbeddingResult with all embeddings.
        """
        if not texts:
            return EmbeddingResult(embeddings=[], model=self._model)

        response = await self._client.embeddings.create(
            model=self._model,
            input=texts,
            dimensions=self._dimension if "3-" in self._model else None,
        )

        # Sort by index to maintain order
        sorted_data = sorted(response.data, key=lambda x: x.index)
        embeddings = [item.embedding for item in sorted_data]

        return EmbeddingResult(
            embeddings=embeddings,
            model=response.model,
            total_tokens=response.usage.total_tokens,
        )
