"""Models for the ``/documents`` endpoint."""

from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, Field


class DocumentSummary(BaseModel):
    """Summary information about an indexed document."""

    doc_id: str
    title: str
    chunks: int = Field(..., ge=0)
    ingested_at: datetime


class DocumentsResponse(BaseModel):
    """List of indexed documents."""

    documents: list[DocumentSummary]
