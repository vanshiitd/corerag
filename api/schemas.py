"""API request/response schemas."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

from core.retrieval import ScoredChunk


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


class SearchTimings(BaseModel):
    """Per-stage latency breakdown for a /search request."""

    retrieval_ms: float
    rerank_ms: float
    total_ms: float


class SearchResponse(BaseModel):
    """Debug view of the two-stage retrieval pipeline: candidates before and after
    reranking, with full citation payload and a timing breakdown."""

    query: str
    retrieval_k: int
    rerank_top_n: int
    stage1_candidates: list[ScoredChunk]
    results: list[ScoredChunk]
    timings: SearchTimings


class QueryRequest(BaseModel):
    """A user question for the agentic /query pipeline."""

    query: str = Field(..., min_length=1)
