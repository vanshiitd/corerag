"""HTTP routes."""

from __future__ import annotations

from typing import Annotated

import redis.asyncio as aioredis
from fastapi import APIRouter, Depends, Response
from qdrant_client import AsyncQdrantClient

from api.deps import get_qdrant, get_redis
from api.schemas import DependencyHealth, HealthResponse
from core import __version__
from core.clients import check_qdrant, check_redis

router = APIRouter()

QdrantDep = Annotated[AsyncQdrantClient, Depends(get_qdrant)]
RedisDep = Annotated[aioredis.Redis, Depends(get_redis)]


@router.get("/")
async def root() -> dict[str, str]:
    """Minimal service banner."""
    return {"service": "corerag", "version": __version__, "docs": "/docs"}


@router.get("/health", response_model=HealthResponse)
async def health(response: Response, qdrant: QdrantDep, redis: RedisDep) -> HealthResponse:
    """Readiness probe: reports per-dependency status and returns 503 if degraded."""
    q_ok, q_ms, q_detail = await check_qdrant(qdrant)
    r_ok, r_ms, r_detail = await check_redis(redis)
    dependencies = {
        "qdrant": DependencyHealth(
            status="up" if q_ok else "down", latency_ms=round(q_ms, 2), detail=q_detail
        ),
        "redis": DependencyHealth(
            status="up" if r_ok else "down", latency_ms=round(r_ms, 2), detail=r_detail
        ),
    }
    healthy = all(dep.status == "up" for dep in dependencies.values())
    if not healthy:
        response.status_code = 503
    return HealthResponse(
        status="ok" if healthy else "degraded",
        service="corerag",
        version=__version__,
        dependencies=dependencies,
    )
