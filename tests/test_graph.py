"""End-to-end tests for core/agents/graph.py (P3.6).

These run the real, compiled graph against the real 150-paper corpus with real
LLM calls -- slow (each takes several to tens of seconds, given P2's ~3.9s p50
rerank cost multiplied across the reflection loop), but they're the ones that
actually prove the agentic pipeline works, not just that its pieces do in
isolation. Run with:  make test-int
"""

from __future__ import annotations

from collections.abc import AsyncIterator

import pytest
from qdrant_client import AsyncQdrantClient

from core.agents.graph import build_graph
from core.clients import make_qdrant_client
from core.config import get_settings


@pytest.fixture
async def qdrant() -> AsyncIterator[AsyncQdrantClient]:
    client = make_qdrant_client(get_settings())
    try:
        yield client
    finally:
        await client.close()


@pytest.mark.integration
async def test_graph_answers_a_real_question_with_citations(
    qdrant: AsyncQdrantClient,
) -> None:
    """Locks in the manually-validated happy path: a real, on-topic question gets
    a grounded, cited answer without needing any reflection retries."""
    settings = get_settings()
    graph = build_graph(qdrant, settings)
    query = "techniques for reducing latency in LLM inference serving"

    result = await graph.ainvoke(
        {"query": query, "original_query": query, "retries": 0},
        config={"recursion_limit": 20},
    )

    assert result["relevant"] is True
    assert result["low_confidence"] is False
    assert result.get("retries", 0) == 0  # no reflection needed
    assert len(result["citations"]) > 0
    assert len(result["answer"]) > 0


@pytest.mark.integration
async def test_graph_off_topic_query_triggers_reflection_then_abstains(
    qdrant: AsyncQdrantClient,
) -> None:
    """Locks in the manually-validated reflection path: a question with no
    relevant sources in this corpus triggers rewrite retries, exhausts them, and
    ends in a graceful low-confidence answer -- not a hallucinated one."""
    settings = get_settings()
    graph = build_graph(qdrant, settings)
    query = "What is the best recipe for chocolate chip cookies?"

    result = await graph.ainvoke(
        {"query": query, "original_query": query, "retries": 0},
        config={"recursion_limit": 20},
    )

    assert result["relevant"] is False
    assert result["low_confidence"] is True
    assert result["retries"] == settings.max_reflection_retries
    assert "answer" in result and len(result["answer"]) > 0
