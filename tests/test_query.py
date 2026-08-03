"""Integration tests for POST /query (P3.7).

Require live Qdrant/Redis and the real embedded corpus, and make real LLM calls
through the full agent graph -- slow (see test_graph.py). Run with: make test-int
"""

from __future__ import annotations

import json

import pytest
from fastapi.testclient import TestClient

from api.main import app


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
