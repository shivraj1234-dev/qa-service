"""PDF parsing utilities backed by pypdf."""

from __future__ import annotations

import io

from pypdf import PdfReader
from pypdf.errors import PdfReadError

from qa_service.observability.logging import get_logger

logger = get_logger(__name__)


class PdfParseError(ValueError):
    """Raised when a PDF cannot be read or contains no extractable text."""


def extract_text_from_pdf(pdf_bytes: bytes) -> str:
    """Extract text from the bytes of a PDF document.

    Args:
        pdf_bytes: Raw bytes of a PDF file.

    Returns:
        The concatenated text content with page boundaries marked by
        double newlines.

    Raises:
        PdfParseError: If the bytes cannot be parsed as a PDF or no text is
            extractable from any page.
    """
    if not pdf_bytes:
        raise PdfParseError("PDF payload is empty")

    try:
        reader = PdfReader(io.BytesIO(pdf_bytes))
    except (PdfReadError, OSError) as exc:
        raise PdfParseError(f"Could not read PDF: {exc}") from exc

    pages: list[str] = []
    for index, page in enumerate(reader.pages):
        try:
            text = page.extract_text() or ""
        except (PdfReadError, ValueError) as exc:
            logger.warning("pdf_page_extract_failed", page=index, error=str(exc))
            text = ""
        text = text.strip()
        if text:
            pages.append(text)

    if not pages:
        raise PdfParseError("PDF contains no extractable text")

    return "\n\n".join(pages)
