"""Tests for core/agents/generator.py (P3.5)."""

from __future__ import annotations

import pytest

from core.agents.generator import _format_sources, build_generator_node
from core.config import get_settings
from core.llm import get_generation_model
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


def test_format_sources_numbers_entries() -> None:
    formatted = _format_sources([_chunk("a"), _chunk("b")])
    assert "[1]" in formatted
    assert "[2]" in formatted


@pytest.mark.integration
async def test_generator_produces_no_sources_answer_when_reranked_empty() -> None:
    node = build_generator_node(get_generation_model(get_settings()))
    result = await node({"original_query": "anything", "reranked": []})
    assert result["citations"] == []
    assert "don't have relevant sources" in result["answer"]


@pytest.mark.integration
async def test_generator_cites_sources_in_answer() -> None:
    node = build_generator_node(get_generation_model(get_settings()))
    chunk = _chunk(
        "Speculative decoding accelerates LLM inference by drafting candidate tokens "
        "with a cheaper model and verifying them with the target model in parallel.",
        title="SpecDecode: Fast LLM Inference",
    )
    result = await node(
        {
            "original_query": "what is speculative decoding?",
            "reranked": [chunk],
            "low_confidence": False,
        }
    )
    assert "[1]" in result["answer"]
    assert result["citations"] == [chunk]
