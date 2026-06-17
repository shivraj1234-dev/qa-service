"""PDF ingestion endpoint."""

from __future__ import annotations

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile, status

from qa_service.dependencies import get_rag_service
from qa_service.models.ingest import IngestResponse
from qa_service.observability.logging import get_logger
from qa_service.services.pdf_parser import PdfParseError, extract_text_from_pdf
from qa_service.services.rag import RagService

router = APIRouter(tags=["ingest"])
logger = get_logger(__name__)

_MAX_PDF_BYTES = 25 * 1024 * 1024


@router.post("/ingest", response_model=IngestResponse, status_code=status.HTTP_201_CREATED)
async def ingest_pdf(
    file: UploadFile = File(..., description="PDF file to index."),
    title: str | None = Form(default=None, description="Optional display title."),
    rag: RagService = Depends(get_rag_service),
) -> IngestResponse:
    """Ingest a PDF, creating a per-document vector collection.

    The endpoint accepts a multipart upload. Each ingested document becomes
    its own vector collection that can later be queried independently or as
    part of a fan-out across all documents.
    """
    if (
        file.content_type
        and "pdf" not in file.content_type.lower()
        and not (file.filename or "").lower().endswith(".pdf")
    ):
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail="Only PDF uploads are supported.",
        )

    payload = await file.read()
    if len(payload) == 0:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Uploaded file is empty.",
        )
    if len(payload) > _MAX_PDF_BYTES:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=f"PDF exceeds maximum size of {_MAX_PDF_BYTES} bytes.",
        )

    try:
        text = extract_text_from_pdf(payload)
    except PdfParseError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Could not parse PDF: {exc}",
        ) from exc

    display_title = (title or file.filename or "untitled").strip() or "untitled"
    result = rag.ingest(title=display_title, content=text)
    return IngestResponse(
        doc_id=result.doc_id,
        title=result.title,
        chunks_indexed=result.chunks_indexed,
        status="success",
    )
