"""API request/response schemas."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel


class DependencyHealth(BaseModel):
    """Health of a single downstream dependency."""

    status: Literal["up", "down"]
    latency_ms: float | None = None
    detail: str | None = None


class HealthResponse(BaseModel):
    """Aggregate service health."""

    status: Literal["ok", "degraded"]
    service: str
    version: str
    dependencies: dict[str, DependencyHealth]
