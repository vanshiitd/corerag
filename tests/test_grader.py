"""Tests for core/agents/grader.py (P3.4)."""

from __future__ import annotations

import pytest

from core.agents.grader import GradeResult, _format_passages, build_grader_node
from core.config import Settings, get_settings
from core.llm import get_agent_model
from core.retrieval import ScoredChunk


def _chunk(text: str, title: str = "Test Paper") -> ScoredChunk:
    return ScoredChunk(
        chunk_id="1.1::0",
        arxiv_id="1.1",
        title=title,
        authors=["A. Author"],
        abs_url="https://arxiv.org/abs/1.1",
        section="Introduction",
        index=0,
        text=text,
        score=0.9,
    )


def test_format_passages_numbers_and_includes_section() -> None:
    formatted = _format_passages([_chunk("some content"), _chunk("more content")])
    assert "[1]" in formatted
    assert "[2]" in formatted
    assert "Introduction" in formatted


@pytest.mark.integration
async def test_grader_marks_relevant_passages_relevant() -> None:
    settings = get_settings()
    node = build_grader_node(get_agent_model(settings), settings)
    chunk = _chunk(
        "Speculative decoding accelerates LLM inference by drafting candidate tokens "
        "with a cheaper model and verifying them with the target model in parallel."
    )
    result = await node(
        {"query": "what is speculative decoding?", "reranked": [chunk], "retries": 0}
    )
    assert result["relevant"] is True
    assert result["low_confidence"] is False


@pytest.mark.integration
async def test_grader_rewrites_query_on_irrelevant_passages() -> None:
    settings = get_settings()
    node = build_grader_node(get_agent_model(settings), settings)
    chunk = _chunk("This paper discusses chocolate chip cookie baking techniques.")
    result = await node(
        {"query": "what is speculative decoding?", "reranked": [chunk], "retries": 0}
    )
    assert result["relevant"] is False
    assert result["retries"] == 1
    assert result["query"] != "what is speculative decoding?"  # rewritten


@pytest.mark.integration
async def test_grader_proceeds_with_low_confidence_when_retries_exhausted() -> None:
    settings = Settings(max_reflection_retries=2)
    node = build_grader_node(get_agent_model(settings), settings)
    chunk = _chunk("This paper discusses chocolate chip cookie baking techniques.")
    result = await node(
        {"query": "what is speculative decoding?", "reranked": [chunk], "retries": 2}
    )
    assert result["relevant"] is False
    assert result["low_confidence"] is True
    assert "retries" not in result  # proceed branch doesn't touch retries


def test_grade_result_rewritten_query_defaults_empty() -> None:
    assert GradeResult(relevant=True).rewritten_query == ""
