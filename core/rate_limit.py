"""Per-IP rate limiting for the public hosted demo (P6.3).

Redis-backed fixed-window counter (INCR + EXPIRE) -- the simplest correct
pattern for this use case (abuse protection on a portfolio demo, not a
metered product needing precise sliding-window fairness).
"""

from __future__ import annotations

import redis.asyncio as aioredis

from core.config import Settings


class RateLimitExceededError(Exception):
    """Raised when a client has exceeded its per-minute request budget."""

    def __init__(self, retry_after_seconds: int) -> None:
        self.retry_after_seconds = retry_after_seconds
        super().__init__(f"rate limit exceeded, retry after {retry_after_seconds}s")


async def check_rate_limit(client: aioredis.Redis, settings: Settings, key: str) -> None:
    """Raise RateLimitExceededError if `key` (typically a client IP) has made more
    than settings.rate_limit_per_minute requests in the current 60s window."""
    redis_key = f"corerag:ratelimit:{key}"
    count = await client.incr(redis_key)
    if count == 1:
        # Only the request that created the counter sets its expiry, so a
        # concurrent second request landing before this line can't reset it.
        await client.expire(redis_key, 60)
    if count > settings.rate_limit_per_minute:
        ttl = await client.ttl(redis_key)
        raise RateLimitExceededError(retry_after_seconds=ttl if ttl > 0 else 60)
