"""Semantic cache interceptor: sub-second answers on repeat/near-duplicate queries.

Wraps RedisVL's SemanticCache with our own local embedder (CustomVectorizer over
data.embeddings.embed_dense) -- no new paid provider, consistent with the
hybrid-providers architecture. Threshold (settings.cache_similarity_threshold,
default 0.95) empirically validated during P4 planning against real
opposite-intent query pairs -- see PLAN.md.
"""

from __future__ import annotations

import asyncio

from pydantic import BaseModel
from redisvl.extensions.cache.llm import SemanticCache
from redisvl.utils.vectorize import CustomVectorizer

from core.config import Settings
from core.retrieval import ScoredChunk
from data.embeddings import embed_dense

# Keyed on the primitive values that actually determine cache identity, not
# @lru_cache: Settings isn't hashable (a mutable pydantic model).
_cache_instances: dict[tuple[str, str, float, int], SemanticCache] = {}


class CachedAnswer(BaseModel):
    """A cached (answer, citations) pair returned on a cache hit."""

    answer: str
    citations: list[ScoredChunk]


def get_cache(settings: Settings) -> SemanticCache:
    """Return the process-wide SemanticCache for this config, building it once.

    The cache name incorporates cache_version (same versioned-namespace pattern
    as qdrant_collection_version) -- bumping it after a re-ingest invalidates all
    prior entries for free, no manual cleanup needed.
    """
    key = (
        settings.redis_url,
        settings.cache_version,
        settings.cache_similarity_threshold,
        settings.cache_ttl_seconds,
    )
    if key not in _cache_instances:

        def embed(text: str, **kwargs: object) -> list[float]:
            return embed_dense([text], settings)[0]

        _cache_instances[key] = SemanticCache(
            name=f"corerag_cache_{settings.cache_version}",
            vectorizer=CustomVectorizer(embed),
            redis_url=settings.redis_url,
            distance_threshold=1 - settings.cache_similarity_threshold,
            ttl=settings.cache_ttl_seconds,
        )
    return _cache_instances[key]


def check_cache(query: str, settings: Settings) -> CachedAnswer | None:
    """Synchronous cache lookup. CPU-bound (local embedding) + I/O; dispatched via
    asyncio.to_thread by check_cache_async for async callers."""
    if not settings.cache_enabled:
        return None
    hits = get_cache(settings).check(prompt=query, num_results=1)
    if not hits:
        return None
    citations_data = hits[0].get("metadata", {}).get("citations") or []
    return CachedAnswer(
        answer=hits[0]["response"],
        citations=[ScoredChunk(**c) for c in citations_data],
    )


def store_in_cache(
    query: str, answer: str, citations: list[ScoredChunk], settings: Settings
) -> None:
    """Synchronous cache write-through. See check_cache re: threading."""
    if not settings.cache_enabled:
        return
    get_cache(settings).store(
        prompt=query,
        response=answer,
        metadata={"citations": [c.model_dump() for c in citations]},
    )


async def check_cache_async(query: str, settings: Settings) -> CachedAnswer | None:
    """Async wrapper: cache lookup is CPU-bound (embedding) + I/O, dispatched to
    a thread so it never blocks the event loop."""
    return await asyncio.to_thread(check_cache, query, settings)


async def store_in_cache_async(
    query: str, answer: str, citations: list[ScoredChunk], settings: Settings
) -> None:
    """Async wrapper: see check_cache_async."""
    await asyncio.to_thread(store_in_cache, query, answer, citations, settings)
