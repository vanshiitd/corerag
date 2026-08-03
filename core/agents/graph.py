"""Graph assembly: router -> retrieve -> grade -> (retry loop | generate).

Wires the already-tested P2 retrieval/rerank modules (core.retrieval,
core.reranker) and the P3 agent nodes into a compiled LangGraph state machine,
with the reflection retry bounded by settings.max_reflection_retries.
"""

from __future__ import annotations

import structlog
from langgraph.graph import END, StateGraph
from langgraph.graph.state import CompiledStateGraph
from qdrant_client import AsyncQdrantClient

from core.agents.generator import build_generator_node
from core.agents.grader import build_grader_node
from core.agents.router import build_router_node
from core.agents.state import GraphState, NodeFn
from core.config import Settings
from core.llm import get_agent_model, get_generation_model
from core.reranker import rerank_async
from core.retrieval import ScoredChunk, hybrid_search

log = structlog.get_logger()


def _dedupe(chunks: list[ScoredChunk]) -> list[ScoredChunk]:
    seen: set[str] = set()
    out: list[ScoredChunk] = []
    for c in chunks:
        if c.chunk_id not in seen:
            seen.add(c.chunk_id)
            out.append(c)
    return out


def build_retrieve_node(client: AsyncQdrantClient, settings: Settings) -> NodeFn:
    """Hybrid retrieval + rerank. Multi-hop fans out over sub_queries on the first
    pass only; a retry (after a grader rewrite) always does one focused retrieval
    with the rewritten query, since the rewrite already targets a single gap."""

    async def retrieve_node(state: GraphState) -> GraphState:
        is_retry = state.get("retries", 0) > 0
        route = state.get("route", "simple")
        sub_queries = state.get("sub_queries") or []
        queries = (
            sub_queries
            if (not is_retry and route == "multi_hop" and sub_queries)
            else [state["query"]]
        )

        all_candidates: list[ScoredChunk] = []
        for q in queries:
            all_candidates.extend(await hybrid_search(client, settings, q))
        merged = _dedupe(all_candidates)

        reranked = await rerank_async(state["query"], merged, settings)
        log.info(
            "retrieve.done", queries=len(queries), candidates=len(merged), reranked=len(reranked)
        )
        return {"candidates": merged, "reranked": reranked}

    return retrieve_node


def _route_after_grade(state: GraphState) -> str:
    # grader_node sets low_confidence only on its "proceed" branch (relevant, or
    # retries exhausted); its absence means "retry" was decided this round. A
    # single authoritative signal, computed once inside grader_node, avoids
    # re-deriving (and risking an off-by-one on) the retry threshold here too.
    return "generate" if "low_confidence" in state else "retrieve"


def build_graph(client: AsyncQdrantClient, settings: Settings) -> CompiledStateGraph:
    """Build and compile the query-time agent graph."""
    agent_model = get_agent_model(settings)
    generation_model = get_generation_model(settings)

    graph = StateGraph(GraphState)
    graph.add_node("router", build_router_node(agent_model))
    graph.add_node("retrieve", build_retrieve_node(client, settings))
    graph.add_node("grade", build_grader_node(agent_model, settings))
    graph.add_node("generate", build_generator_node(generation_model))

    graph.set_entry_point("router")
    graph.add_edge("router", "retrieve")
    graph.add_edge("retrieve", "grade")
    graph.add_conditional_edges(
        "grade", _route_after_grade, {"retrieve": "retrieve", "generate": "generate"}
    )
    graph.add_edge("generate", END)

    return graph.compile()
