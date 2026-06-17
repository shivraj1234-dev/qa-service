"""Models for the ``/ingest`` endpoint."""

from __future__ import annotations

from pydantic import BaseModel, Field


class IngestResponse(BaseModel):
    """Response payload returned after ingesting a document."""

    doc_id: str = Field(..., description="Unique identifier for the ingested document.")
    title: str = Field(..., description="Display title of the document.")
    chunks_indexed: int = Field(..., ge=0, description="Number of chunks indexed.")
    status: str = Field(default="success")
