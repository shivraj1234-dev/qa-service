"""Prometheus metrics for the QA service."""

from __future__ import annotations

from prometheus_client import (
    CollectorRegistry,
    Counter,
    disable_created_metrics,
    generate_latest,
)
from prometheus_client.exposition import CONTENT_TYPE_LATEST

# Suppress the auxiliary ``*_created`` timestamp series so ``/metrics`` only
# exposes the two counters required by the assignment.
disable_created_metrics()

REGISTRY = CollectorRegistry()

DOCUMENTS_INGESTED_TOTAL = Counter(
    "documents_ingested_total",
    "Total number of documents successfully ingested.",
    registry=REGISTRY,
)

QUERIES_PROCESSED_TOTAL = Counter(
    "queries_processed_total",
    "Total number of queries answered.",
    registry=REGISTRY,
)


def render_metrics() -> tuple[bytes, str]:
    """Render Prometheus metrics payload and its content type.

    Returns:
        Tuple of payload bytes and the Prometheus text content type.
    """
    return generate_latest(REGISTRY), CONTENT_TYPE_LATEST

