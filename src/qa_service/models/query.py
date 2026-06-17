"""Models for the ``/query`` endpoint."""

from __future__ import annotations

from pydantic import BaseModel, Field


class QueryRequest(BaseModel):
    """Question payload sent to ``POST /query``."""

    question: str = Field(..., min_length=1, max_length=2000)
    doc_id: str | None = Field(
        default=None,
        description="Optional document id to scope the query. If omitted, all documents are searched.",
    )


class Source(BaseModel):
    """A single retrieved chunk used to ground the answer."""

    doc_id: str
    title: str
    chunk: str


class QueryResponse(BaseModel):
    """Response returned by ``POST /query`` (matches assignment spec)."""

    answer: str
    sources: list[Source]
