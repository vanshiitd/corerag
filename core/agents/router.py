"""Router node: classify simple vs multi-hop, decompose if needed.

Uses structured output (verified working against gpt-4o-mini in P3.0) so the
decision is deterministic to parse -- no free-form LLM text to interpret.
"""

from __future__ import annotations

from typing import Literal

import structlog
from langchain_core.language_models import BaseChatModel
from pydantic import BaseModel, Field

from core.agents.state import GraphState, NodeFn

log = structlog.get_logger()

_ROUTER_PROMPT = """You are the routing component of a retrieval-augmented question \
answering system over AI-systems research papers (LLM serving, inference optimization, \
distributed training, hardware accelerators, and related systems topics).

Given a user's question, decide:
- "simple": a single focused retrieval can answer it.
- "multi_hop": answering it well requires combining information from genuinely \
distinct sub-topics, or comparing multiple techniques/systems (e.g. "compare X and Y", \
"what are the tradeoffs between A, B, and C").

If multi_hop, decompose the question into 2-4 focused, independently-retrievable \
sub-queries that together cover what's needed to answer the original question. If \
simple, leave sub_queries empty.

Question: {query}"""


class RouteDecision(BaseModel):
    route: Literal["simple", "multi_hop"]
    sub_queries: list[str] = Field(
        default_factory=list,
        description="2-4 focused sub-queries if route is multi_hop, else empty",
    )


def build_router_node(agent_model: BaseChatModel) -> NodeFn:
    """Return a LangGraph node that classifies (and decomposes) the query."""
    structured = agent_model.with_structured_output(RouteDecision)

    async def router_node(state: GraphState) -> GraphState:
        query = state["query"]
        decision = await structured.ainvoke(_ROUTER_PROMPT.format(query=query))
        assert isinstance(decision, RouteDecision)
        log.info("router.decision", route=decision.route, sub_queries=decision.sub_queries)
        return {"route": decision.route, "sub_queries": decision.sub_queries}

    return router_node
