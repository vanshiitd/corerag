"""Generator node: synthesize a cited answer from the reranked context.

Uses the generation model (Groq, low-latency token streaming) via
core.llm.get_generation_model. Token-level SSE streaming happens at the API layer
(P3.7) via LangGraph's astream_events; this node's own .ainvoke() call produces
the complete answer for graph state and any non-streaming caller.
"""

from __future__ import annotations

import structlog
from langchain_core.language_models import BaseChatModel

from core.agents.state import GraphState, NodeFn
from core.retrieval import ScoredChunk

log = structlog.get_logger()

_GENERATOR_PROMPT = """You are answering a question about AI systems research (LLM \
serving, inference optimization, distributed training, hardware accelerators, and \
related topics) using ONLY the numbered source passages below. Cite sources inline \
using [n] markers matching the passage numbers, for every claim you make.
{low_confidence_note}
Question: {query}

Sources:
{passages}

Write a clear, well-cited answer. If the sources don't fully answer the question, say \
so explicitly rather than filling gaps with unsupported claims."""

_LOW_CONFIDENCE_NOTE = (
    "\nNote: the retrieved sources may only partially cover this question. Be explicit "
    "about what the sources do and don't establish, rather than overstating confidence.\n"
)


def _format_sources(chunks: list[ScoredChunk]) -> str:
    return "\n\n".join(
        f"[{i + 1}] {c.title} ({c.section or 'N/A'}):\n{c.text[:800]}" for i, c in enumerate(chunks)
    )


def build_generator_node(generation_model: BaseChatModel) -> NodeFn:
    """Return a LangGraph node that synthesizes a cited answer."""

    async def generator_node(state: GraphState) -> GraphState:
        query = state["original_query"]  # answer the user's actual question, not a rewrite
        reranked = state.get("reranked", [])
        low_confidence = state.get("low_confidence", False)

        if not reranked:
            return {
                "answer": "I don't have relevant sources in this corpus to answer that question.",
                "citations": [],
            }

        prompt = _GENERATOR_PROMPT.format(
            query=query,
            passages=_format_sources(reranked),
            low_confidence_note=_LOW_CONFIDENCE_NOTE if low_confidence else "",
        )
        result = await generation_model.ainvoke(prompt)
        answer = str(result.content)
        log.info("generator.done", answer_len=len(answer), low_confidence=low_confidence)
        return {"answer": answer, "citations": reranked}

    return generator_node
