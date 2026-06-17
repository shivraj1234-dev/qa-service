"""Application dependency wiring."""

from __future__ import annotations

from functools import lru_cache

from qa_service.config import Settings, get_settings
from qa_service.providers.embedding import EmbeddingProvider, build_embedding_provider
from qa_service.providers.llm import LLMProvider, build_llm_provider
from qa_service.repository.vector_store import ChromaVectorStore
from qa_service.services.judge import JudgeService
from qa_service.services.rag import RagService


@lru_cache(maxsize=1)
def _settings() -> Settings:
    return get_settings()


@lru_cache(maxsize=1)
def get_embedding_provider() -> EmbeddingProvider:
    """Return the configured embedding provider (singleton)."""
    settings = _settings()
    return build_embedding_provider(
        provider=settings.embedding_provider,
        model_name=settings.embedding_model,
    )


@lru_cache(maxsize=1)
def get_llm_provider() -> LLMProvider:
    """Return the configured LLM provider (singleton)."""
    settings = _settings()
    return build_llm_provider(
        provider=settings.llm_provider,
        base_url=settings.llm_base_url,
        api_key=settings.llm_api_key,
        model=settings.llm_model,
        temperature=settings.llm_temperature,
        max_tokens=settings.llm_max_tokens,
    )


@lru_cache(maxsize=1)
def get_vector_store() -> ChromaVectorStore:
    """Return the configured vector store (singleton)."""
    return ChromaVectorStore(vector_store_url=_settings().vector_store_url)


@lru_cache(maxsize=1)
def get_rag_service() -> RagService:
    """Return the configured RAG service (singleton)."""
    settings = _settings()
    return RagService(
        embedder=get_embedding_provider(),
        llm=get_llm_provider(),
        vector_store=get_vector_store(),
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
        top_k=settings.top_k,
    )


@lru_cache(maxsize=1)
def get_judge_service() -> JudgeService:
    """Return the configured judge service (singleton)."""
    return JudgeService(llm=get_llm_provider())
