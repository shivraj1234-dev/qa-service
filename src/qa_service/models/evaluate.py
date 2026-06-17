"""Models for the ``/evaluate`` endpoint."""

from __future__ import annotations

from pydantic import BaseModel, Field

# Upper bound on test cases per request. Each case triggers two upstream LLM
# calls (a RAG query plus a judge call), so capping the batch size prevents an
# unauthenticated caller from amplifying cost / exhausting the LLM provider.
MAX_TEST_CASES = 50


class EvalCase(BaseModel):
    """A single Q&A pair used to evaluate the RAG pipeline."""

    question: str = Field(..., min_length=1)
    expected_answer: str = Field(..., min_length=1)
    doc_id: str | None = Field(
        default=None,
        description="Optional doc_id to scope the evaluation question.",
    )


class EvalRequest(BaseModel):
    """Request payload for ``POST /evaluate``."""

    test_cases: list[EvalCase] = Field(..., min_length=1, max_length=MAX_TEST_CASES)


class EvalResultItem(BaseModel):
    """Per-case evaluation result."""

    question: str
    generated_answer: str
    score: float = Field(..., ge=0.0, le=1.0)
    reasoning: str


class EvalResponse(BaseModel):
    """Aggregate evaluation response."""

    total: int = Field(..., ge=0)
    avg_score: float = Field(..., ge=0.0, le=1.0)
    results: list[EvalResultItem]
