"""Query endpoint."""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, status

from qa_service.dependencies import get_rag_service
from qa_service.models.query import QueryRequest, QueryResponse
from qa_service.services.rag import RagService

router = APIRouter(tags=["query"])


@router.post("/query", response_model=QueryResponse)
def query(
    payload: QueryRequest,
    rag: RagService = Depends(get_rag_service),
) -> QueryResponse:
    """Answer a question with grounded citations.

    If ``doc_id`` is provided, only that document is searched. Otherwise the
    query is run across all indexed documents.
    """
    try:
        return rag.query(question=payload.question, doc_id=payload.doc_id)
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(exc),
        ) from exc
