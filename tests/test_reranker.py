"""Tests for core/reranker.py (P2.2).

Loading the real cross-encoder is slow/heavy (network + real inference), so every
test here is marked integration.  Run with:  make test-int
"""

from __future__ import annotations

from collections.abc import AsyncIterator

import pytest
from qdrant_client import AsyncQdrantClient

from core.clients import make_qdrant_client
from core.config import Settings, get_settings
from core.reranker import rerank, rerank_async
from core.retrieval import ScoredChunk, hybrid_search


def _chunk(arxiv_id: str, text: str, score: float = 0.5) -> ScoredChunk:
    return ScoredChunk(
        chunk_id=f"{arxiv_id}::0",
        arxiv_id=arxiv_id,
        title=f"Paper {arxiv_id}",
        authors=["A. Author"],
        abs_url=f"https://arxiv.org/abs/{arxiv_id}",
        section=None,
        index=0,
        text=text,
        context=None,
        score=score,
    )


@pytest.mark.integration
def test_rerank_empty_candidates_returns_empty() -> None:
    settings = get_settings()
    assert rerank("any query", [], settings) == []


@pytest.mark.integration
def test_rerank_scores_relevant_doc_above_irrelevant() -> None:
    """Locks in the manual correctness check: a clearly relevant doc must outscore
    a clearly irrelevant one, and the reranked score must replace the stage-1 score."""
    settings = get_settings()
    query = "techniques for reducing GPU memory usage during LLM inference"
    relevant = _chunk(
        "1111.1111",
        "We propose a KV cache compression method that reduces GPU memory footprint "
        "by 40% during autoregressive LLM inference through selective quantization.",
        score=0.1,  # deliberately low stage-1 score to prove reranking overrides it
    )
    irrelevant = _chunk(
        "2222.2222",
        "This paper presents a Byzantine fault-tolerant consensus algorithm for "
        "distributed databases operating under network partitions.",
        score=0.9,  # deliberately high stage-1 score
    )

    result = rerank(query, [irrelevant, relevant], settings)

    assert result[0].arxiv_id == "1111.1111"  # relevant doc promoted to top
    assert result[0].score > result[1].score
    assert result[0].score > 0.9  # cross-encoder confidence, not the stale 0.1 stage-1 score


@pytest.mark.integration
def test_rerank_respects_top_n() -> None:
    settings = Settings(rerank_top_n=2, retrieval_k=10)
    candidates = [_chunk(f"{i}.{i}", f"chunk number {i} about various topics") for i in range(5)]
    result = rerank("some query", candidates, settings)
    assert len(result) == 2


@pytest.mark.integration
def test_onnx_mode_agrees_with_pytorch_mode_on_relevance() -> None:
    """ONNX (int8, CPUExecutionProvider) shouldn't be bit-identical to PyTorch fp32
    -- quantization causes real numerical drift -- but must agree on which
    candidate is more relevant, since that's what actually matters for ranking."""
    query = "techniques for reducing GPU memory usage during LLM inference"
    relevant = _chunk(
        "1111.1111",
        "We propose a KV cache compression method that reduces GPU memory footprint "
        "by 40% during autoregressive LLM inference through selective quantization.",
    )
    irrelevant = _chunk(
        "2222.2222",
        "This paper presents a Byzantine fault-tolerant consensus algorithm for "
        "distributed databases operating under network partitions.",
    )

    torch_settings = Settings(reranker_mode="pytorch-cpu")
    onnx_settings = Settings(reranker_mode="cpu-onnx")

    torch_result = rerank(query, [irrelevant, relevant], torch_settings)
    onnx_result = rerank(query, [irrelevant, relevant], onnx_settings)

    assert torch_result[0].arxiv_id == onnx_result[0].arxiv_id == "1111.1111"
    assert torch_result[0].score > 0.9
    assert onnx_result[0].score > 0.9


@pytest.fixture
async def qdrant() -> AsyncIterator[AsyncQdrantClient]:
    client = make_qdrant_client(get_settings())
    try:
        yield client
    finally:
        await client.close()


@pytest.mark.integration
async def test_rerank_improves_ordering_on_real_corpus(qdrant: AsyncQdrantClient) -> None:
    """Regression test locking in the manual validation: reranking a wider stage-1
    pool for "reducing latency in LLM inference serving" promotes DeltaServe (a
    paper specifically about co-serving latency-sensitive inference) into the top
    results, even though it wasn't the #1 hybrid/RRF result."""
    settings = get_settings()
    query = "techniques for reducing latency in LLM inference serving"
    candidates = await hybrid_search(qdrant, settings, query, limit=20)

    reranked = await rerank_async(query, candidates, settings)

    assert len(reranked) <= settings.rerank_top_n
    scores = [r.score for r in reranked]
    assert scores == sorted(scores, reverse=True)
    top_ids = {r.arxiv_id for r in reranked}
    assert "2607.28848" in top_ids  # DeltaServe
