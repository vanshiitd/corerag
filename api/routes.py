"""HTTP routes."""

from __future__ import annotations

import json
import time
from collections.abc import AsyncIterator
from typing import Annotated

import redis.asyncio as aioredis
import structlog
from fastapi import APIRouter, Depends, HTTPException, Query, Response
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
from core.clients import check_qdrant, check_redis
from core.config import Settings
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


async def _stream_query(
    graph: CompiledStateGraph, settings: Settings, query: str
) -> AsyncIterator[str]:
    """Yield SSE events: a 'token' event per generated token (from the *generate*
    node only -- router/grader also stream their own raw structured-output JSON,
    which must not leak into the user-facing answer), then one final 'sources'
    event once the graph completes."""
    handler = get_tracing_handler(settings)
    callbacks = [handler] if handler else []
    citations: list[ScoredChunk] = []

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
            citations = event["data"]["output"].get("citations", [])

    yield _sse("sources", {"citations": [c.model_dump() for c in citations]})


@router.post("/query")
async def query(body: QueryRequest, graph: GraphDep, settings: SettingsDep) -> StreamingResponse:
    """Agentic query endpoint: streams a grounded, cited answer via SSE.

    No semantic-cache interceptor here -- that's P4's job, layered in front of
    this later.
    """
    return StreamingResponse(
        _stream_query(graph, settings, body.query), media_type="text/event-stream"
    )
