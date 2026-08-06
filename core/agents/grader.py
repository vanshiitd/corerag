"""Grader node: assess reranked-result relevance; rewrite the query and retry if weak.

Bounded by ``settings.max_reflection_retries`` (default 2) -- a circuit breaker so
a persistently poor retrieval can't loop forever. On exhaustion, proceeds anyway
with a ``low_confidence`` flag rather than failing the request outright.
"""

from __future__ import annotations

import structlog
from langchain_core.language_models import BaseChatModel
from pydantic import BaseModel

from core.agents.state import GraphState, NodeFn
from core.config import Settings
from core.retrieval import ScoredChunk

log = structlog.get_logger()

_GRADER_PROMPT = """You are grading whether retrieved passages from AI-systems research \
papers give a reasonable basis for answering a user's question.

Question: {query}

Retrieved passages:
{passages}

Mark relevant=True if the passages substantively engage with the question's core topic, \
even if they don't cover every detail or sub-aspect -- a partial but genuinely on-topic \
answer is a pass, not a fail. Mark relevant=False only if the passages are largely \
off-topic or fail to address what's actually being asked. If not relevant, propose a \
single rewritten search query more likely to retrieve what's actually needed (more \
specific terminology, a different angle, or narrower scope) -- otherwise leave \
rewritten_query empty."""


class GradeResult(BaseModel):
    relevant: bool
    rewritten_query: str = ""


def _format_passages(chunks: list[ScoredChunk]) -> str:
    return "\n\n".join(
        f"[{i + 1}] {c.title} ({c.section or 'N/A'}):\n{c.text[:500]}" for i, c in enumerate(chunks)
    )


def build_grader_node(agent_model: BaseChatModel, settings: Settings) -> NodeFn:
    """Return a LangGraph node that grades relevance and rewrites the query on failure."""
    structured = agent_model.with_structured_output(GradeResult)

    async def grader_node(state: GraphState) -> GraphState:
        query = state["query"]
        reranked = state.get("reranked", [])
        retries = state.get("retries", 0)

        if not reranked:
            grade = GradeResult(relevant=False, rewritten_query="")
        else:
            passages = _format_passages(reranked)
            result = await structured.ainvoke(_GRADER_PROMPT.format(query=query, passages=passages))
            assert isinstance(result, GradeResult)
            grade = result

        exhausted = retries >= settings.max_reflection_retries
        if grade.relevant or exhausted:
            log.info(
                "grader.proceed", relevant=grade.relevant, exhausted=exhausted, retries=retries
            )
            return {"relevant": grade.relevant, "low_confidence": not grade.relevant}

        rewritten = grade.rewritten_query or query
        log.info("grader.retry", retries=retries + 1, rewritten_query=rewritten)
        return {"relevant": False, "retries": retries + 1, "query": rewritten}

    return grader_node
