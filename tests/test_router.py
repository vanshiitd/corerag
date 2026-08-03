"""Tests for core/agents/router.py (P3.3)."""

from __future__ import annotations

import pytest

from core.agents.router import RouteDecision, build_router_node
from core.config import get_settings
from core.llm import get_agent_model


def test_route_decision_defaults_to_empty_sub_queries() -> None:
    decision = RouteDecision(route="simple")
    assert decision.sub_queries == []


@pytest.mark.integration
async def test_router_classifies_simple_query() -> None:
    node = build_router_node(get_agent_model(get_settings()))
    result = await node({"query": "what is speculative decoding?"})
    assert result["route"] == "simple"


@pytest.mark.integration
async def test_router_classifies_comparison_query_as_multi_hop() -> None:
    node = build_router_node(get_agent_model(get_settings()))
    result = await node(
        {
            "query": (
                "Compare KV cache compression techniques with speculative decoding "
                "and continuous batching -- what are the tradeoffs between all three?"
            )
        }
    )
    assert result["route"] == "multi_hop"
    assert len(result["sub_queries"]) >= 2
