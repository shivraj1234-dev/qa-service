"""LLM provider protocol and OpenAI-compatible implementation."""

from __future__ import annotations

from typing import Protocol

from openai import OpenAI

from qa_service.observability.logging import get_logger

logger = get_logger(__name__)


class LLMProvider(Protocol):
    """Protocol for chat-style LLM providers."""

    def complete(self, system: str, user: str, *, temperature: float | None = None) -> str:
        """Generate a completion given system + user prompts."""
        ...


class OpenAICompatibleProvider:
    """Chat completion provider for any OpenAI-compatible HTTP endpoint.

    This works for OpenAI itself, NVIDIA NIM, Together, Groq, vLLM,
    and most other API-compatible inference services.
    """

    def __init__(
        self,
        *,
        base_url: str,
        api_key: str,
        model: str,
        temperature: float = 0.1,
        max_tokens: int = 512,
    ) -> None:
        """Initialise the provider.

        Args:
            base_url: HTTP base URL of the OpenAI-compatible endpoint.
            api_key: API key (may be empty for self-hosted endpoints).
            model: Model identifier.
            temperature: Default sampling temperature.
            max_tokens: Default response token cap.
        """
        self._client = OpenAI(base_url=base_url, api_key=api_key or "not-needed")
        self._model = model
        self._temperature = temperature
        self._max_tokens = max_tokens

    def complete(self, system: str, user: str, *, temperature: float | None = None) -> str:
        """Generate a completion from the model.

        Args:
            system: System prompt text.
            user: User prompt text.
            temperature: Optional override for sampling temperature.

        Returns:
            The text content of the first choice, stripped of surrounding
            whitespace.

        Raises:
            RuntimeError: If the upstream response contains no choices or no
                content (network/transport errors are surfaced as the
                original openai-sdk exceptions).
        """
        response = self._client.chat.completions.create(
            model=self._model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            temperature=self._temperature if temperature is None else temperature,
            max_tokens=self._max_tokens,
        )
        if not response.choices:
            raise RuntimeError("LLM response contained no choices")
        message = response.choices[0].message
        content = (message.content or "").strip()
        if not content:
            raise RuntimeError("LLM response contained empty content")
        return content


def build_llm_provider(
    *,
    provider: str,
    base_url: str,
    api_key: str,
    model: str,
    temperature: float,
    max_tokens: int,
) -> LLMProvider:
    """Construct an :class:`LLMProvider` from configuration.

    Currently all supported providers (``nvidia``, ``openai``, ``custom``) are
    served via :class:`OpenAICompatibleProvider` since they share the same
    HTTP contract.
    """
    supported = {"nvidia", "openai", "custom", "openai-compatible"}
    if provider.lower() not in supported:
        raise ValueError(f"Unsupported LLM provider: {provider!r}. Supported: {sorted(supported)}")
    return OpenAICompatibleProvider(
        base_url=base_url,
        api_key=api_key,
        model=model,
        temperature=temperature,
        max_tokens=max_tokens,
    )
