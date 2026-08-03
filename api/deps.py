"""FastAPI dependency providers -- typed accessors for objects on app.state."""

from __future__ import annotations

from typing import cast

import redis.asyncio as aioredis
from fastapi import Request
from qdrant_client import AsyncQdrantClient


def get_qdrant(request: Request) -> AsyncQdrantClient:
    """Return the process-wide async Qdrant client."""
    return cast(AsyncQdrantClient, request.app.state.qdrant)


def get_redis(request: Request) -> aioredis.Redis:
    """Return the process-wide async Redis client."""
    return cast(aioredis.Redis, request.app.state.redis)
