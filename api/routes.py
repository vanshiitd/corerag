"""HTTP routes."""

from __future__ import annotations

import json
import time
from collections.abc import AsyncIterator
from typing import Annotated

import redis.asyncio as aioredis
import structlog
from fastapi import APIRouter, Depends, HTTPException, Query, Request, Response
from fastapi.responses import StreamingResponse
from langgraph.graph.state import CompiledStateGraph
from qdrant_client import AsyncQdrantClient

from api.deps import get_graph, get_qdrant, get_redis, get_settings
from api.schemas import (
    DependencyHealth,
    HealthResponse,
    QueryRequest,
    SearchResponse,
    SearchTimings,
)
from core import __version__
from core.cache import CachedAnswer, check_cache_async, store_in_cache_async
from core.clients import check_qdrant, check_redis
from core.config import Settings
from core.rate_limit import RateLimitExceededError, check_rate_limit
from core.reranker import rerank_async
from core.retrieval import ScoredChunk, hybrid_search
from core.tracing import get_tracing_handler

log = structlog.get_logger()

router = APIRouter()

QdrantDep = Annotated[AsyncQdrantClient, Depends(get_qdrant)]
RedisDep = Annotated[aioredis.Redis, Depends(get_redis)]
SettingsDep = Annotated[Settings, Depends(get_settings)]
GraphDep = Annotated[CompiledStateGraph, Depends(get_graph)]


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


@router.get("/search", response_model=SearchResponse)
async def search(
    qdrant: QdrantDep,
    settings: SettingsDep,
    q: str = Query(..., min_length=1, description="Search query"),
    k: int | None = Query(None, ge=1, description="Override retrieval_k (stage-1 candidates)"),
    n: int | None = Query(None, ge=1, description="Override rerank_top_n (final results)"),
) -> SearchResponse:
    """Debug endpoint: hybrid retrieval + rerank, with pre/post candidates, full
    citation payload, and a per-stage timing breakdown."""
    effective_k = k if k is not None else settings.retrieval_k
    effective_n = n if n is not None else settings.rerank_top_n
    if effective_n > effective_k:
        raise HTTPException(400, detail="n (rerank_top_n) cannot exceed k (retrieval_k)")
    if k is not None or n is not None:
        settings = settings.model_copy(
            update={"retrieval_k": effective_k, "rerank_top_n": effective_n}
        )

    t0 = time.perf_counter()
    stage1 = await hybrid_search(qdrant, settings, q)
    t1 = time.perf_counter()
    results = await rerank_async(q, stage1, settings)
    t2 = time.perf_counter()

    return SearchResponse(
        query=q,
        retrieval_k=effective_k,
        rerank_top_n=effective_n,
        stage1_candidates=stage1,
        results=results,
        timings=SearchTimings(
            retrieval_ms=round((t1 - t0) * 1000, 1),
            rerank_ms=round((t2 - t1) * 1000, 1),
            total_ms=round((t2 - t0) * 1000, 1),
        ),
    )


def _sse(event: str, data: object) -> str:
    return f"event: {event}\ndata: {json.dumps(data)}\n\n"


def _chunk_words(text: str) -> list[str]:
    """Split into word(+space) pieces for a cache hit's simulated token stream --
    same event shape as real generation, fired rapidly with no artificial delay,
    so client-side parsing needs no special case for a hit vs a miss."""
    words = text.split(" ")
    return [f"{w} " for w in words[:-1]] + (words[-1:] if words else [])


async def _stream_cached(cached: CachedAnswer) -> AsyncIterator[str]:
    """Replay a cache hit as the same token/sources SSE shapes as a real run."""
    for piece in _chunk_words(cached.answer):
        yield _sse("token", {"content": piece})
    yield _sse("sources", {"citations": [c.model_dump() for c in cached.citations]})


async def _stream_query(
    graph: CompiledStateGraph, settings: Settings, query: str
) -> AsyncIterator[str]:
    """Yield SSE events: a 'token' event per generated token (from the *generate*
    node only -- router/grader also stream their own raw structured-output JSON,
    which must not leak into the user-facing answer), then one final 'sources'
    event once the graph completes. Writes through to the cache before that final
    yield (not after) -- guaranteed to run as long as the client stayed connected
    through the token stream, unlike code placed after the last yield, which a
    client disconnecting right at the end could skip entirely."""
    handler = get_tracing_handler(settings)
    callbacks = [handler] if handler else []
    citations: list[ScoredChunk] = []
    answer = ""

    async for event in graph.astream_events(
        {"query": query, "original_query": query, "retries": 0},
        config={"callbacks": callbacks, "recursion_limit": 20},
        version="v2",
    ):
        node = event.get("metadata", {}).get("langgraph_node")
        if event["event"] == "on_chat_model_stream" and node == "generate":
            content = event["data"]["chunk"].content
            if content:
                yield _sse("token", {"content": content})
        elif event["event"] == "on_chain_end" and node == "generate":
            output = event["data"]["output"]
            citations = output.get("citations", [])
            answer = output.get("answer", "")

    if answer:
        await store_in_cache_async(query, answer, citations, settings)

    yield _sse("sources", {"citations": [c.model_dump() for c in citations]})


def _client_ip(request: Request) -> str:
    # Prefer X-Forwarded-For: Cloud Run (and most PaaS hosts) sits behind a
    # proxy, so request.client.host alone would be the proxy's own address,
    # not the real caller's -- the same IP for every client, defeating
    # per-client rate limiting entirely.
    forwarded = request.headers.get("x-forwarded-for")
    if forwarded:
        return forwarded.split(",")[0].strip()
    return request.client.host if request.client else "unknown"


async def _enforce_rate_limit(request: Request, redis: RedisDep, settings: SettingsDep) -> None:
    try:
        await check_rate_limit(redis, settings, _client_ip(request))
    except RateLimitExceededError as exc:
        raise HTTPException(
            status_code=429,
            detail="Too many requests -- please wait before trying again.",
            headers={"Retry-After": str(exc.retry_after_seconds)},
        ) from exc


@router.post("/query", dependencies=[Depends(_enforce_rate_limit)])
async def query(body: QueryRequest, graph: GraphDep, settings: SettingsDep) -> StreamingResponse:
    """Agentic query endpoint: streams a grounded, cited answer via SSE.

    Rate-limited per client IP (settings.rate_limit_per_minute) -- this is a
    public demo, not a metered product, so the goal is abuse protection, not
    precise fairness. Checks the semantic cache first -- a hit skips the graph
    entirely (no LLM/retrieval calls) and streams the cached answer/citations
    in the same SSE shapes; a miss runs the full graph, then writes through
    after the stream.
    """
    cached = await check_cache_async(body.query, settings)
    if cached is not None:
        log.info("query.cache_hit", query=body.query[:80])
        stream = _stream_cached(cached)
    else:
        log.info("query.cache_miss", query=body.query[:80])
        stream = _stream_query(graph, settings, body.query)
    return StreamingResponse(stream, media_type="text/event-stream")
