"""Integration tests for core/rate_limit.py (P6.3) -- requires live Redis."""

from __future__ import annotations

from collections.abc import AsyncIterator

import pytest
import redis.asyncio as aioredis

from core.config import Settings, get_settings
from core.rate_limit import RateLimitExceededError, check_rate_limit


@pytest.fixture
async def redis_client() -> AsyncIterator[aioredis.Redis]:
    client = aioredis.Redis.from_url(get_settings().redis_url, decode_responses=True)
    try:
        yield client
    finally:
        await client.delete("corerag:ratelimit:test-key")
        await client.aclose()


@pytest.mark.integration
async def test_allows_requests_within_the_limit(redis_client: aioredis.Redis) -> None:
    settings = Settings(rate_limit_per_minute=3)
    for _ in range(3):
        await check_rate_limit(redis_client, settings, "test-key")  # should not raise


@pytest.mark.integration
async def test_raises_once_the_limit_is_exceeded(redis_client: aioredis.Redis) -> None:
    settings = Settings(rate_limit_per_minute=2)
    await check_rate_limit(redis_client, settings, "test-key")
    await check_rate_limit(redis_client, settings, "test-key")
    with pytest.raises(RateLimitExceededError) as exc_info:
        await check_rate_limit(redis_client, settings, "test-key")
    assert exc_info.value.retry_after_seconds > 0


@pytest.mark.integration
async def test_different_keys_have_independent_budgets(redis_client: aioredis.Redis) -> None:
    settings = Settings(rate_limit_per_minute=1)
    await check_rate_limit(redis_client, settings, "test-key")
    await check_rate_limit(redis_client, settings, "test-key-other")  # independent, no raise
    await redis_client.delete("corerag:ratelimit:test-key-other")
