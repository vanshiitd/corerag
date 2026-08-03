"""Shared state threaded through the LangGraph query pipeline.

A TypedDict, not a Pydantic model: LangGraph nodes return *partial* updates that
get shallow-merged into the accumulated state, which is the idiomatic LangGraph
pattern -- ``total=False`` reflects that most fields aren't populated until their
owning node has run.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from typing import Literal, TypedDict

from core.retrieval import ScoredChunk


class GraphState(TypedDict, total=False):
    query: str  # current query -- the original, or the grader's rewrite
    original_query: str  # the user's query, unchanged across retries
    route: Literal["simple", "multi_hop"]
    sub_queries: list[str]
    candidates: list[ScoredChunk]  # stage-1 hybrid retrieval (post-merge if multi-hop)
    reranked: list[ScoredChunk]  # stage-2 cross-encoder rerank (what the grader sees)
    relevant: bool
    retries: int
    low_confidence: bool
    answer: str
    citations: list[ScoredChunk]


# LangGraph node signature: reads the accumulated state, returns a partial update
# (a subset of GraphState's keys) that gets shallow-merged in.
NodeFn = Callable[[GraphState], Awaitable[GraphState]]
