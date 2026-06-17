"""RAG orchestration: ingest documents, answer questions with citations."""

from __future__ import annotations

import uuid
from dataclasses import dataclass

from qa_service.models.query import QueryResponse, Source
from qa_service.observability.logging import get_logger
from qa_service.observability.metrics import (
    DOCUMENTS_INGESTED_TOTAL,
    QUERIES_PROCESSED_TOTAL,
)
from qa_service.prompts import load_prompt
from qa_service.providers.embedding import EmbeddingProvider
from qa_service.providers.llm import LLMProvider
from qa_service.repository.vector_store import (
    ChromaVectorStore,
    StoredChunk,
)
from qa_service.services.chunker import chunk_text

logger = get_logger(__name__)


@dataclass(frozen=True, slots=True)
class IngestResult:
    """Result of a successful ingest call."""

    doc_id: str
    title: str
    chunks_indexed: int


class RagService:
    """High-level service that wires together chunking, embedding, retrieval and generation."""

    def __init__(
        self,
        *,
        embedder: EmbeddingProvider,
        llm: LLMProvider,
        vector_store: ChromaVectorStore,
        chunk_size: int,
        chunk_overlap: int,
        top_k: int,
    ) -> None:
        """Initialise with all required dependencies."""
        self._embedder = embedder
        self._llm = llm
        self._store = vector_store
        self._chunk_size = chunk_size
        self._chunk_overlap = chunk_overlap
        self._top_k = top_k

    def ingest(self, *, title: str, content: str) -> IngestResult:
        """Chunk, embed and persist a document.

        Args:
            title: Display title.
            content: Raw text content.

        Returns:
            :class:`IngestResult` with the new ``doc_id`` and chunk count.

        Raises:
            ValueError: If ``content`` produces zero chunks.
        """
        chunks = chunk_text(content, chunk_size=self._chunk_size, overlap=self._chunk_overlap)
        if not chunks:
            raise ValueError("Document produced no chunks (empty content)")

        doc_id = str(uuid.uuid4())
        texts = [chunk.text for chunk in chunks]
        embeddings = self._embedder.embed(texts)
        self._store.add_document(
            doc_id=doc_id,
            title=title,
            chunks=texts,
            embeddings=embeddings,
        )
        DOCUMENTS_INGESTED_TOTAL.inc()
        logger.info("ingest_complete", doc_id=doc_id, title=title, chunks=len(chunks))
        return IngestResult(doc_id=doc_id, title=title, chunks_indexed=len(chunks))

    def query(self, *, question: str, doc_id: str | None = None) -> QueryResponse:
        """Answer ``question`` using retrieved chunks as grounding context."""
        query_embedding = self._embedder.embed([question])[0]
        retrieved = self._store.query(
            embedding=query_embedding,
            top_k=self._top_k,
            doc_id=doc_id,
        )

        if not retrieved:
            QUERIES_PROCESSED_TOTAL.inc()
            return QueryResponse(
                answer="I don't know based on the provided documents.",
                sources=[],
            )

        prompt = load_prompt("answer")
        context = self._format_context(retrieved)
        user_prompt = prompt["user"].format(question=question, context=context)
        answer = self._llm.complete(system=prompt["system"], user=user_prompt)

        QUERIES_PROCESSED_TOTAL.inc()
        sources = [
            Source(doc_id=chunk.doc_id, title=chunk.title, chunk=chunk.text) for chunk in retrieved
        ]
        return QueryResponse(answer=answer, sources=sources)

    @staticmethod
    def _format_context(chunks: list[StoredChunk]) -> str:
        """Format retrieved chunks for inclusion in the answer prompt."""
        return "\n\n".join(
            f"[{i + 1}] (title={chunk.title})\n{chunk.text}" for i, chunk in enumerate(chunks)
        )
