"""Documents listing endpoint."""

from __future__ import annotations

from fastapi import APIRouter, Depends

from qa_service.dependencies import get_vector_store
from qa_service.models.document import DocumentsResponse, DocumentSummary
from qa_service.repository.vector_store import ChromaVectorStore

router = APIRouter(tags=["documents"])


@router.get("/documents", response_model=DocumentsResponse)
def list_documents(
    store: ChromaVectorStore = Depends(get_vector_store),
) -> DocumentsResponse:
    """Return summaries of all indexed documents (most recent first)."""
    summaries = [
        DocumentSummary(
            doc_id=info.doc_id,
            title=info.title,
            chunks=info.chunks,
            ingested_at=info.ingested_at,
        )
        for info in store.list_documents()
    ]
    return DocumentsResponse(documents=summaries)
