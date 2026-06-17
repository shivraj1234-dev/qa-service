"""LLM-as-judge evaluation service."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass

from qa_service.observability.logging import get_logger
from qa_service.prompts import load_prompt
from qa_service.providers.llm import LLMProvider

logger = get_logger(__name__)

_JSON_OBJECT_RE = re.compile(r"\{.*\}", re.DOTALL)


@dataclass(frozen=True, slots=True)
class JudgeVerdict:
    """A judge's verdict on a single answer."""

    score: float
    reasoning: str


def _parse_verdict(text: str) -> JudgeVerdict | None:
    """Parse a judge response, tolerating minor formatting drift."""
    match = _JSON_OBJECT_RE.search(text)
    if match is None:
        return None
    try:
        data = json.loads(match.group(0))
    except json.JSONDecodeError:
        return None
    if not isinstance(data, dict):
        return None
    score_raw = data.get("score")
    reasoning_raw = data.get("reasoning", "")
    try:
        score = float(score_raw)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None
    score = max(0.0, min(1.0, score))
    return JudgeVerdict(score=score, reasoning=str(reasoning_raw))


class JudgeService:
    """Score generated answers against expected answers using an LLM judge."""

    def __init__(self, *, llm: LLMProvider) -> None:
        """Initialise with an :class:`LLMProvider`."""
        self._llm = llm
        self._prompt = load_prompt("judge")

    def score(
        self,
        *,
        question: str,
        expected: str,
        generated: str,
    ) -> JudgeVerdict:
        """Return a verdict for the generated answer.

        On parse failure the judge is retried once with a slightly higher
        temperature; if that also fails, a deterministic fallback verdict is
        returned so the batch can continue.
        """
        user_prompt = self._prompt["user"].format(
            question=question,
            expected=expected,
            generated=generated,
        )
        for attempt in range(2):
            response = self._llm.complete(
                system=self._prompt["system"],
                user=user_prompt,
                temperature=0.0 if attempt == 0 else 0.3,
            )
            verdict = _parse_verdict(response)
            if verdict is not None:
                return verdict
            logger.warning("judge_parse_failed", attempt=attempt + 1, response=response[:200])
        return JudgeVerdict(score=0.0, reasoning="judge response could not be parsed")
