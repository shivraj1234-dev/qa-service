"""Embedding provider protocol and default implementations."""

from __future__ import annotations

from typing import Protocol

from qa_service.observability.logging import get_logger

logger = get_logger(__name__)


class EmbeddingProvider(Protocol):
    """Protocol for an embedding model."""

    @property
    def dimension(self) -> int:
        """Embedding vector dimension."""
        ...

    def embed(self, texts: list[str]) -> list[list[float]]:
        """Return a list of embedding vectors, one per input text."""
        ...


class SentenceTransformerEmbeddingProvider:
    """Local embedding provider backed by sentence-transformers.

    The model is loaded lazily on first use to avoid blocking startup and to
    keep test environments free of model downloads when not needed.
    """

    def __init__(self, model_name: str) -> None:
        """Initialise with a sentence-transformers model name."""
        self._model_name = model_name
        self._model: object | None = None
        self._dimension: int | None = None

    def _ensure_loaded(self) -> None:
        if self._model is not None:
            return
        from sentence_transformers import SentenceTransformer

        logger.info("loading_embedding_model", model=self._model_name)
        self._model = SentenceTransformer(self._model_name)
        self._dimension = int(self._model.get_sentence_embedding_dimension())  # type: ignore[union-attr]
        logger.info(
            "embedding_model_loaded",
            model=self._model_name,
            dimension=self._dimension,
        )

    @property
    def dimension(self) -> int:
        self._ensure_loaded()
        assert self._dimension is not None  # for type checker
        return self._dimension

    def embed(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        self._ensure_loaded()
        assert self._model is not None
        vectors = self._model.encode(  # type: ignore[union-attr]
            texts,
            convert_to_numpy=True,
            show_progress_bar=False,
            normalize_embeddings=True,
        )
        return [vector.tolist() for vector in vectors]


def build_embedding_provider(provider: str, model_name: str) -> EmbeddingProvider:
    """Construct an embedding provider from its identifier.

    Args:
        provider: Provider name (only ``"sentence-transformers"`` is bundled).
        model_name: Model identifier.

    Returns:
        A configured :class:`EmbeddingProvider`.

    Raises:
        ValueError: If ``provider`` is not supported.
    """
    if provider.lower() in {"sentence-transformers", "st", "local"}:
        return SentenceTransformerEmbeddingProvider(model_name=model_name)
    raise ValueError(f"Unsupported embedding provider: {provider!r}")
