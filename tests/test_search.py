"""Integration tests for GET /search (P2.5).

Require live Qdrant + Redis and the real embedded corpus; reranking is CPU-heavy
(multi-second per call), so these are slow.  Run with:  make test-int
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from api.main import app


@pytest.mark.integration
def test_search_returns_relevant_cited_results() -> None:
    """Regression test locking in the same retrieval quality validated manually
    throughout P2: this query surfaces DeltaServe/ServerlessT2I/DualDecoder."""
    with TestClient(app) as client:
        resp = client.get(
            "/search", params={"q": "techniques for reducing latency in LLM inference serving"}
        )
    assert resp.status_code == 200
    body = resp.json()

    assert body["query"] == "techniques for reducing latency in LLM inference serving"
    assert body["retrieval_k"] == 30
    assert body["rerank_top_n"] == 5
    assert len(body["stage1_candidates"]) == 30
    assert len(body["results"]) == 5

    scores = [r["score"] for r in body["results"]]
    assert scores == sorted(scores, reverse=True)

    top_ids = {r["arxiv_id"] for r in body["results"]}
    assert "2607.28848" in top_ids  # DeltaServe

    for r in body["results"]:
        assert r["text"]
        assert r["title"]
        assert r["abs_url"]

    timings = body["timings"]
    assert timings["retrieval_ms"] > 0
    assert timings["rerank_ms"] > 0
    assert timings["total_ms"] >= timings["retrieval_ms"] + timings["rerank_ms"] - 1  # rounding


@pytest.mark.integration
def test_search_respects_k_and_n_overrides() -> None:
    with TestClient(app) as client:
        resp = client.get("/search", params={"q": "quantization", "k": 10, "n": 3})
    assert resp.status_code == 200
    body = resp.json()
    assert body["retrieval_k"] == 10
    assert body["rerank_top_n"] == 3
    assert len(body["stage1_candidates"]) == 10
    assert len(body["results"]) == 3


@pytest.mark.integration
def test_search_rejects_n_greater_than_k() -> None:
    with TestClient(app) as client:
        resp = client.get("/search", params={"q": "test", "k": 5, "n": 10})
    assert resp.status_code == 400


@pytest.mark.integration
def test_search_rejects_empty_query() -> None:
    with TestClient(app) as client:
        resp = client.get("/search", params={"q": ""})
    assert resp.status_code == 422


@pytest.mark.integration
def test_search_requires_query_param() -> None:
    with TestClient(app) as client:
        resp = client.get("/search")
    assert resp.status_code == 422
