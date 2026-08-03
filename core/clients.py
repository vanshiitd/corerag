"""Client factories and readiness probes for external services (Qdrant, Redis)."""

from __future__ import annotations

import time

import redis.asyncio as aioredis
from qdrant_client import AsyncQdrantClient

from core.config import Settings


def make_qdrant_client(settings: Settings) -> AsyncQdrantClient:
    """Build an async Qdrant client from settings."""
    api_key = settings.qdrant_api_key.get_secret_value() if settings.qdrant_api_key else None
    return AsyncQdrantClient(url=settings.qdrant_url, api_key=api_key)


def make_redis_client(settings: Settings) -> aioredis.Redis:
    """Build an async Redis client from settings."""
    return aioredis.Redis.from_url(settings.redis_url, decode_responses=True)


async def check_qdrant(client: AsyncQdrantClient) -> tuple[bool, float, str | None]:
    """Ping Qdrant; return (ok, latency_ms, error_detail)."""
    start = time.perf_counter()
    try:
        await client.get_collections()
    except Exception as exc:
        return False, (time.perf_counter() - start) * 1000, str(exc)
    return True, (time.perf_counter() - start) * 1000, None


async def check_redis(client: aioredis.Redis) -> tuple[bool, float, str | None]:
    """Ping Redis; return (ok, latency_ms, error_detail)."""
    start = time.perf_counter()
    try:
        # redis-py's ping() is typed `Awaitable[bool] | bool` on the shared
        # sync/async command mixin (confirmed in its own source) -- imprecise for
        # any single concrete client, not a real type error for this async one.
        await client.ping()  # type: ignore[misc]
    except Exception as exc:
        return False, (time.perf_counter() - start) * 1000, str(exc)
    return True, (time.perf_counter() - start) * 1000, None
