"""SentenceTransformers Embedding Adapter for local embeddings."""

import asyncio
from concurrent.futures import ThreadPoolExecutor
from typing import Any

from rag_core.domain.interfaces.embedding_port import EmbeddingPort, EmbeddingResult


class SentenceTransformerAdapter(EmbeddingPort):
    """SentenceTransformers adapter implementing EmbeddingPort.

    This adapter uses the sentence-transformers library for local
    embedding generation without requiring external services.
    """

    def __init__(
        self,
        model: str = "all-MiniLM-L6-v2",
        device: str | None = None,
        normalize: bool = True,
    ):
        """Initialize the SentenceTransformers adapter.

        Args:
            model: Model name or path.
            device: Device to use ('cpu', 'cuda', etc.). Auto-detected if None.
            normalize: Whether to normalize embeddings.
        """
        self._model_name = model
        self._normalize = normalize
        self._model: Any = None
        self._dimension: int = 0
        self._device = device
        self._executor = ThreadPoolExecutor(max_workers=2)

    def _load_model(self) -> None:
        """Lazy load the model."""
        if self._model is None:
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(
                self._model_name,
                device=self._device,
            )
            # Get dimension from a test embedding
            test_embedding = self._model.encode("test", normalize_embeddings=self._normalize)
            self._dimension = len(test_embedding)

    @property
    def model_name(self) -> str:
        """Return the model name."""
        return self._model_name

    @property
    def dimension(self) -> int:
        """Return the embedding dimension."""
        if self._dimension == 0:
            self._load_model()
        return self._dimension

    def _sync_embed(self, text: str) -> list[float]:
        """Synchronous embedding for single text."""
        self._load_model()
        embedding = self._model.encode(text, normalize_embeddings=self._normalize)
        return embedding.tolist()

    def _sync_embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Synchronous embedding for batch of texts."""
        self._load_model()
        embeddings = self._model.encode(texts, normalize_embeddings=self._normalize)
        return [emb.tolist() for emb in embeddings]

    async def embed(self, text: str) -> list[float]:
        """Generate an embedding for a single text.

        Args:
            text: The text to embed.

        Returns:
            Embedding vector as a list of floats.
        """
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(self._executor, self._sync_embed, text)

    async def embed_batch(self, texts: list[str]) -> EmbeddingResult:
        """Generate embeddings for multiple texts.

        Args:
            texts: List of texts to embed.

        Returns:
            EmbeddingResult with all embeddings.
        """
        if not texts:
            return EmbeddingResult(embeddings=[], model=self._model_name)

        loop = asyncio.get_event_loop()
        embeddings = await loop.run_in_executor(
            self._executor, self._sync_embed_batch, texts
        )

        return EmbeddingResult(
            embeddings=embeddings,
            model=self._model_name,
            total_tokens=0,  # SentenceTransformers doesn't track tokens
        )

    def close(self) -> None:
        """Cleanup resources."""
        self._executor.shutdown(wait=False)
