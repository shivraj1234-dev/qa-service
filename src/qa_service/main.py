"""FastAPI application entry point."""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.responses import Response

from qa_service.config import get_settings
from qa_service.observability.logging import configure_logging, get_logger
from qa_service.observability.metrics import render_metrics
from qa_service.routes import documents, evaluate, health, ingest, query


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Configure logging on startup; nothing to clean up on shutdown."""
    settings = get_settings()
    configure_logging(level=settings.log_level)
    logger = get_logger("qa_service")
    logger.info(
        "service_starting",
        llm_model=settings.llm_model,
        embedding_model=settings.embedding_model,
        vector_store_url=settings.vector_store_url,
    )
    yield
    logger.info("service_stopping")


def create_app() -> FastAPI:
    """Create and configure the FastAPI application."""
    app = FastAPI(
        title="QA Service",
        description="Document Q&A service with RAG and LLM-as-judge evaluation.",
        version="0.1.0",
        lifespan=lifespan,
    )
    app.include_router(health.router)
    app.include_router(ingest.router)
    app.include_router(query.router)
    app.include_router(documents.router)
    app.include_router(evaluate.router)

    @app.get(
        "/metrics",
        tags=["observability"],
        summary="Prometheus metrics",
        description=(
            "Prometheus exposition of `documents_ingested_total` and "
            "`queries_processed_total`."
        ),
        response_class=Response,
        responses={200: {"content": {"text/plain": {}}}},
    )
    @app.get("/metrics/", include_in_schema=False)
    def metrics_endpoint() -> Response:
        """Expose Prometheus metrics for scraping."""
        payload, content_type = render_metrics()
        return Response(content=payload, media_type=content_type)

    return app


app = create_app()
