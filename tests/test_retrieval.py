"""Tests for core/retrieval.py (P2.1).

Unit tests cover pure logic; integration tests run hybrid/dense search against the
real 150-paper corpus and require live Qdrant.  Run integration with: make test-int
"""

from __future__ import annotations

from collections.abc import AsyncIterator

import pytest
from qdrant_client import AsyncQdrantClient, models

from core.clients import make_qdrant_client
from core.config import get_settings
from core.retrieval import ScoredChunk, _to_scored_chunk, dense_search, hybrid_search


def _point(payload: dict[str, object] | None, score: float = 0.5) -> models.ScoredPoint:
    return models.ScoredPoint(
        id="00000000-0000-0000-0000-000000000001", version=0, score=score, payload=payload
    )


def _payload() -> dict[str, object]:
    return {
        "chunk_id": "1234.5678::0",
        "arxiv_id": "1234.5678",
        "title": "Test Paper",
        "authors": ["A. Author"],
        "abs_url": "https://arxiv.org/abs/1234.5678",
        "section": None,
        "index": 0,
        "text": "some chunk text",
        "context": None,
    }


def test_to_scored_chunk_builds_scored_chunk_from_payload() -> None:
    chunk = _to_scored_chunk(_point(_payload(), score=0.83))
    assert isinstance(chunk, ScoredChunk)
    assert chunk.score == 0.83
    assert chunk.arxiv_id == "1234.5678"
    assert chunk.text == "some chunk text"


def test_to_scored_chunk_raises_on_missing_payload() -> None:
    with pytest.raises(ValueError, match="no payload"):
        _to_scored_chunk(_point(None))


@pytest.fixture
async def qdrant() -> AsyncIterator[AsyncQdrantClient]:
    client = make_qdrant_client(get_settings())
    try:
        yield client
    finally:
        await client.close()


@pytest.mark.integration
async def test_hybrid_search_returns_relevant_results_from_real_corpus(
    qdrant: AsyncQdrantClient,
) -> None:
    """Regression test locking in the retrieval quality validated manually during P1:
    this exact query surfaced ServerlessT2I / KAP / DeltaServe as top results."""
    settings = get_settings()
    results = await hybrid_search(qdrant, settings, "reducing LLM inference latency")

    assert len(results) > 0
    scores = [r.score for r in results]
    assert scores == sorted(scores, reverse=True)  # descending by fused score

    top_ids = {r.arxiv_id for r in results[:5]}
    assert top_ids & {"2607.26566", "2607.24260", "2607.28848"}

    for r in results:
        assert r.text
        assert r.title
        assert r.arxiv_id
        assert r.abs_url


@pytest.mark.integration
async def test_hybrid_search_respects_limit(qdrant: AsyncQdrantClient) -> None:
    settings = get_settings()
    results = await hybrid_search(qdrant, settings, "quantization", limit=7)
    assert len(results) <= 7


@pytest.mark.integration
async def test_dense_search_returns_relevant_results(qdrant: AsyncQdrantClient) -> None:
    settings = get_settings()
    results = await dense_search(qdrant, settings, "GPU memory management for LLM serving")
    assert len(results) > 0
    assert all(r.text for r in results)
