"""Chroma-backed vector store repository.

Each ingested document is stored in its own collection so a single document
forms an independently queryable "vector database". The collection name is
``doc_<doc_id>`` and metadata about the document (title, ingest time, chunk
count) is stored on the collection itself.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any
from urllib.parse import urlparse

import chromadb
from chromadb import HttpClient
from chromadb.api import ClientAPI
from chromadb.config import Settings as ChromaSettings

from qa_service.observability.logging import get_logger

logger = get_logger(__name__)


COLLECTION_PREFIX = "doc_"


@dataclass(frozen=True, slots=True)
class StoredChunk:
    """A retrieved chunk plus its source metadata."""

    doc_id: str
    title: str
    chunk_index: int
    text: str
    score: float


@dataclass(frozen=True, slots=True)
class CollectionInfo:
    """Lightweight summary of a stored document collection."""

    doc_id: str
    title: str
    chunks: int
    ingested_at: datetime


def _build_client(vector_store_url: str) -> ClientAPI:
    """Build a Chroma client.

    Uses the HTTP client when ``vector_store_url`` looks like an HTTP URL,
    otherwise falls back to a persistent local client rooted at the URL path
    (useful for tests and local development without docker-compose).
    """
    parsed = urlparse(vector_store_url)
    if parsed.scheme in {"http", "https"}:
        host = parsed.hostname or "localhost"
        port = parsed.port or 8000
        ssl = parsed.scheme == "https"
        return HttpClient(
            host=host,
            port=port,
            ssl=ssl,
            settings=ChromaSettings(anonymized_telemetry=False),
        )
    path = parsed.path or vector_store_url or "./.chroma"
    return chromadb.PersistentClient(
        path=path,
        settings=ChromaSettings(anonymized_telemetry=False),
    )


class ChromaVectorStore:
    """Vector store backed by ChromaDB.

    The store uses one collection per document so that queries can be scoped
    to a single document, or fanned out across all documents when the caller
    omits a ``doc_id``.
    """

    def __init__(self, vector_store_url: str) -> None:
        """Initialise the repository with a Chroma endpoint URL."""
        self._client = _build_client(vector_store_url)
        self._url = vector_store_url

    @staticmethod
    def _collection_name(doc_id: str) -> str:
        return f"{COLLECTION_PREFIX}{doc_id.replace('-', '')}"

    def add_document(
        self,
        *,
        doc_id: str,
        title: str,
        chunks: list[str],
        embeddings: list[list[float]],
    ) -> None:
        """Create a fresh collection for ``doc_id`` and upsert its chunks."""
        if not chunks:
            raise ValueError("Cannot add a document with zero chunks")
        if len(chunks) != len(embeddings):
            raise ValueError("chunks and embeddings must have the same length")

        ingested_at = datetime.now(UTC).isoformat()
        name = self._collection_name(doc_id)
        collection = self._client.get_or_create_collection(
            name=name,
            metadata={
                "doc_id": doc_id,
                "title": title,
                "ingested_at": ingested_at,
                "chunks": len(chunks),
            },
        )
        ids = [f"{doc_id}:{i}" for i in range(len(chunks))]
        metadatas: list[dict[str, Any]] = [
            {"doc_id": doc_id, "title": title, "chunk_index": i} for i in range(len(chunks))
        ]
        collection.add(
            ids=ids,
            documents=chunks,
            embeddings=embeddings,  # type: ignore[arg-type]
            metadatas=metadatas,
        )
        logger.info("document_indexed", doc_id=doc_id, title=title, chunks=len(chunks))

    def list_documents(self) -> list[CollectionInfo]:
        """Return summary information for all indexed documents."""
        infos: list[CollectionInfo] = []
        for collection in self._client.list_collections():
            if not collection.name.startswith(COLLECTION_PREFIX):
                continue
            metadata = collection.metadata or {}
            doc_id = str(metadata.get("doc_id", ""))
            if not doc_id:
                continue
            ingested_raw = str(metadata.get("ingested_at", ""))
            try:
                ingested_at = datetime.fromisoformat(ingested_raw)
            except ValueError:
                ingested_at = datetime.now(UTC)
            infos.append(
                CollectionInfo(
                    doc_id=doc_id,
                    title=str(metadata.get("title", doc_id)),
                    chunks=int(metadata.get("chunks", 0)),
                    ingested_at=ingested_at,
                )
            )
        infos.sort(key=lambda info: info.ingested_at, reverse=True)
        return infos

    def query(
        self,
        *,
        embedding: list[float],
        top_k: int,
        doc_id: str | None = None,
    ) -> list[StoredChunk]:
        """Retrieve the most similar chunks to ``embedding``.

        When ``doc_id`` is provided only that document's collection is
        searched. Otherwise all collections are queried and the merged
        top-k by score is returned.
        """
        if doc_id is not None:
            return self._query_collection(
                name=self._collection_name(doc_id),
                embedding=embedding,
                top_k=top_k,
            )

        results: list[StoredChunk] = []
        for collection in self._client.list_collections():
            if not collection.name.startswith(COLLECTION_PREFIX):
                continue
            results.extend(
                self._query_collection(
                    name=collection.name,
                    embedding=embedding,
                    top_k=top_k,
                )
            )
        results.sort(key=lambda chunk: chunk.score)
        return results[:top_k]

    def _query_collection(
        self,
        *,
        name: str,
        embedding: list[float],
        top_k: int,
    ) -> list[StoredChunk]:
        try:
            collection = self._client.get_collection(name=name)
        except (ValueError, KeyError) as exc:
            logger.warning("collection_not_found", collection=name, error=str(exc))
            return []
        result = collection.query(
            query_embeddings=[embedding],  # type: ignore[arg-type]
            n_results=top_k,
            include=["documents", "metadatas", "distances"],
        )
        documents = (result.get("documents") or [[]])[0]
        metadatas = (result.get("metadatas") or [[]])[0]
        distances = (result.get("distances") or [[]])[0]
        chunks: list[StoredChunk] = []
        for text, metadata, distance in zip(documents, metadatas, distances, strict=False):
            metadata = metadata or {}
            chunks.append(
                StoredChunk(
                    doc_id=str(metadata.get("doc_id", "")),
                    title=str(metadata.get("title", "")),
                    chunk_index=int(metadata.get("chunk_index", 0)),
                    text=str(text),
                    score=float(distance),
                )
            )
        return chunks
