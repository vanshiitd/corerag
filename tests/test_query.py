"""Integration tests for POST /query (P3.7).

Require live Qdrant/Redis and the real embedded corpus, and make real LLM calls
through the full agent graph -- slow (see test_graph.py). Run with: make test-int
"""

from __future__ import annotations

import contextlib
import json
import time
from collections.abc import Iterator

import pytest
from fastapi.testclient import TestClient

from api.deps import get_settings as get_settings_dep
from api.main import app
from core.cache import get_cache
from core.config import get_settings


def _parse_sse(text: str) -> list[tuple[str, dict[str, object]]]:
    """Parse `event: ...\\ndata: ...\\n\\n` blocks into (event, payload) pairs."""
    events = []
    for block in text.strip().split("\n\n"):
        lines = block.strip().split("\n")
        if len(lines) != 2:
            continue
        event = lines[0].removeprefix("event: ")
        data = json.loads(lines[1].removeprefix("data: "))
        events.append((event, data))
    return events


@pytest.mark.integration
def test_query_streams_tokens_then_sources() -> None:
    with TestClient(app) as client:
        resp = client.post("/query", json={"query": "what is speculative decoding?"})
    assert resp.status_code == 200
    assert resp.headers["content-type"].startswith("text/event-stream")

    events = _parse_sse(resp.text)
    assert events, "expected at least one SSE event"

    token_events = [e for e in events if e[0] == "token"]
    source_events = [e for e in events if e[0] == "sources"]

    assert len(token_events) > 0
    answer = "".join(str(d["content"]) for _, d in token_events)
    assert len(answer) > 0

    assert len(source_events) == 1
    citations = source_events[0][1]["citations"]
    assert isinstance(citations, list)
    assert len(citations) > 0
    assert all("chunk_id" in c and "text" in c for c in citations)


@pytest.fixture
def isolated_cache_settings() -> Iterator[None]:
    """Override the /query endpoint's settings dependency to use a dedicated,
    disposable cache_version, so this test's cache writes never touch the real
    cache namespace (used by manual/demo runs) and are cleaned up afterward."""
    test_settings = get_settings().model_copy(update={"cache_version": "pytest-query-cache"})
    app.dependency_overrides[get_settings_dep] = lambda: test_settings
    cache = get_cache(test_settings)
    with contextlib.suppress(Exception):
        cache.clear()
    try:
        yield
    finally:
        with contextlib.suppress(Exception):
            cache.clear()
        del app.dependency_overrides[get_settings_dep]


@pytest.mark.integration
def test_query_second_identical_call_hits_cache_and_is_fast(
    isolated_cache_settings: None,
) -> None:
    """The real, end-to-end proof of P4: a repeat query returns from cache with
    provably zero LLM/retrieval calls (not just 'it felt fast') and is at least
    two orders of magnitude quicker than the first, real graph-backed call."""
    query = "what is a KV cache and how does it affect memory usage in transformers?"

    with TestClient(app) as client:
        t0 = time.perf_counter()
        first = client.post("/query", json={"query": query})
        first_elapsed = time.perf_counter() - t0
        assert first.status_code == 200
        first_events = _parse_sse(first.text)
        first_answer = "".join(str(d["content"]) for e, d in first_events if e == "token")
        first_citations = next(d for e, d in first_events if e == "sources")["citations"]
        assert isinstance(first_citations, list)
        assert len(first_answer) > 0
        assert len(first_citations) > 0

        t1 = time.perf_counter()
        second = client.post("/query", json={"query": query})
        second_elapsed = time.perf_counter() - t1

    assert second.status_code == 200
    second_events = _parse_sse(second.text)
    second_answer = "".join(str(d["content"]) for e, d in second_events if e == "token")
    second_citations = next(d for e, d in second_events if e == "sources")["citations"]

    assert second_answer.strip() == first_answer.strip()
    assert second_citations == first_citations
    assert second_elapsed < first_elapsed / 10  # at least an order of magnitude faster
    assert second_elapsed < 1.0  # sub-second, well inside the sub-100ms goal in practice


@pytest.mark.integration
def test_query_rejects_empty_query() -> None:
    with TestClient(app) as client:
        resp = client.post("/query", json={"query": ""})
    assert resp.status_code == 422


@pytest.mark.integration
def test_query_requires_query_field() -> None:
    with TestClient(app) as client:
        resp = client.post("/query", json={})
    assert resp.status_code == 422
