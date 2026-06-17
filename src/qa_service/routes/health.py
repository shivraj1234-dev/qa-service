"""Health route used by container orchestrators."""

from __future__ import annotations

from fastapi import APIRouter

router = APIRouter(tags=["health"])


@router.get("/health")
def health() -> dict[str, str]:
    """Return a simple liveness response."""
    return {"status": "ok"}
