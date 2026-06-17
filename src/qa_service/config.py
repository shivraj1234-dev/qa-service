"""Application configuration loaded from environment variables."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

# Resolve the project-root `.env` (two levels up from this file: src/qa_service).
# This makes local runs work regardless of the current working directory.
# Inside Docker the file is absent and env vars are injected directly, which
# pydantic-settings handles transparently.
_ENV_FILE = Path(__file__).resolve().parents[2] / ".env"


class Settings(BaseSettings):
    """Runtime settings for the QA service.

    All values are loaded from environment variables (or a local `.env` file
    in development). See `.env.example` for documentation on each variable.
    """

    model_config = SettingsConfigDict(
        env_file=_ENV_FILE,
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # LLM
    llm_provider: str = Field(default="nvidia")
    llm_base_url: str = Field(default="https://integrate.api.nvidia.com/v1")
    llm_model: str = Field(default="meta/llama-3.3-70b-instruct")
    llm_api_key: str = Field(default="")
    llm_temperature: float = Field(default=0.1, ge=0.0, le=2.0)
    llm_max_tokens: int = Field(default=512, ge=1, le=8192)

    # Embedding
    embedding_provider: str = Field(default="sentence-transformers")
    embedding_model: str = Field(default="sentence-transformers/all-MiniLM-L6-v2")

    # Vector store
    vector_store_url: str = Field(default="http://chroma:8000")

    # Chunking and retrieval
    chunk_size: int = Field(default=512, ge=64, le=4096)
    chunk_overlap: int = Field(default=50, ge=0, le=512)
    top_k: int = Field(default=4, ge=1, le=20)

    # Server
    log_level: str = Field(default="INFO")
    app_port: int = Field(default=8000)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return a cached `Settings` instance."""
    return Settings()
