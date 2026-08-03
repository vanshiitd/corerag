"""Unit tests for contextualization cost estimation, document truncation, and rate limiting."""

from __future__ import annotations

import asyncio

import httpx
import pytest

from core.config import Settings
from data.contextualize import (
    _bounded_doc_text,
    _chat_completion_with_retry,
    _encode,
    _encoder,
    _TokenBucket,
    estimate_cost,
)
from data.schemas import Chunk, ParsedDoc, Section


def _doc(n_words: int) -> ParsedDoc:
    text = " ".join(f"word{i}" for i in range(n_words))
    return ParsedDoc(
        arxiv_id="1234.5678", title="Test Paper", sections=[Section(heading=None, text=text)]
    )


def _chunks(doc: ParsedDoc, n: int) -> list[Chunk]:
    return [
        Chunk(
            chunk_id=f"{doc.arxiv_id}::{i}",
            arxiv_id=doc.arxiv_id,
            title=doc.title,
            authors=["A. Author"],
            abs_url="https://arxiv.org/abs/1234.5678",
            section=None,
            index=i,
            text=f"chunk text number {i}",
        )
        for i in range(n)
    ]


def test_estimate_cost_zero_for_none_strategy() -> None:
    settings = Settings(contextualize_strategy="none")
    doc = _doc(5000)
    cost = estimate_cost(doc, _chunks(doc, 10), settings)
    assert cost == {"calls": 0, "uncached_input": 0, "cached_input": 0, "output": 0, "usd": 0.0}


def test_estimate_cost_per_chunk_scales_with_calls() -> None:
    settings = Settings(contextualize_strategy="per_chunk", contextualize_doc_token_budget=3000)
    doc = _doc(5000)
    chunks = _chunks(doc, 10)
    cost = estimate_cost(doc, chunks, settings)
    assert cost["calls"] == 10
    assert cost["usd"] > 0
    # doc resent on every call after the first -> cached_input scales with (calls - 1)
    assert cost["cached_input"] > 0


def test_estimate_cost_doc_summary_is_single_call() -> None:
    settings = Settings(contextualize_strategy="doc_summary")
    doc = _doc(5000)
    cost = estimate_cost(doc, _chunks(doc, 10), settings)
    assert cost["calls"] == 1
    assert cost["cached_input"] == 0


def test_bounded_doc_text_truncates_large_docs() -> None:
    settings = Settings(contextualize_doc_token_budget=50)
    doc = _doc(5000)  # far larger than the 50-token budget
    bounded = _bounded_doc_text(doc, settings)
    assert len(_encoder().encode(bounded)) <= 50
    assert bounded != doc.full_text


def test_bounded_doc_text_passthrough_when_under_budget() -> None:
    settings = Settings(contextualize_doc_token_budget=10_000)
    doc = _doc(50)  # far smaller than the budget
    assert _bounded_doc_text(doc, settings) == doc.full_text


def test_handles_literal_special_token_text() -> None:
    """Papers about tokenization legitimately contain strings like <|endoftext|>,
    which tiktoken rejects by default as a prompt-injection guard. Regression test
    for a real crash hit mid-run on arXiv 2607.29678 ("TokTier: Exact Stateful
    Tokenization for Agentic LLM Serving")."""
    settings = Settings(contextualize_strategy="per_chunk", contextualize_doc_token_budget=20)
    doc = ParsedDoc(
        arxiv_id="1234.5678",
        title="A Tokenization Paper",
        sections=[Section(heading=None, text="The special token <|endoftext|> marks EOS. " * 20)],
    )
    chunks = [
        Chunk(
            chunk_id=f"{doc.arxiv_id}::{i}",
            arxiv_id=doc.arxiv_id,
            title=doc.title,
            authors=["A. Author"],
            abs_url="https://arxiv.org/abs/1234.5678",
            section=None,
            index=i,
            text="discusses <|endoftext|> handling",
        )
        for i in range(3)
    ]

    assert len(_encode(_encoder(), doc.full_text)) > 0  # would raise pre-fix
    assert isinstance(_bounded_doc_text(doc, settings), str)  # truncation must not raise
    cost = estimate_cost(doc, chunks, settings)
    assert cost["calls"] == 3


def test_per_chunk_cost_respects_doc_token_budget_cap() -> None:
    """A larger document shouldn't blow past the configured per-call token budget."""
    settings = Settings(contextualize_strategy="per_chunk", contextualize_doc_token_budget=500)
    small_doc, big_doc = _doc(400), _doc(50_000)
    small_chunks, big_chunks = _chunks(small_doc, 5), _chunks(big_doc, 5)
    small_cost = estimate_cost(small_doc, small_chunks, settings)
    big_cost = estimate_cost(big_doc, big_chunks, settings)
    # Both docs exceed/approach the budget differently, but big_doc's *effective*
    # per-call doc tokens are capped -- so cached_input shouldn't scale with raw doc size.
    assert big_cost["cached_input"] <= settings.contextualize_doc_token_budget * (
        len(big_chunks) - 1
    )
    assert small_cost["calls"] == big_cost["calls"] == 5


class _FakeClock:
    """Deterministic stand-in for time.monotonic, advanced manually by tests."""

    def __init__(self) -> None:
        self.now = 0.0

    def advance(self, dt: float) -> None:
        self.now += dt

    def __call__(self) -> float:
        return self.now


async def test_token_bucket_immediate_acquire_within_capacity() -> None:
    bucket = _TokenBucket(tokens_per_minute=1000)
    await asyncio.wait_for(bucket.acquire(500), timeout=1.0)
    assert bucket.tokens == pytest.approx(500, abs=1)


async def test_token_bucket_waits_and_refills_when_over_budget(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    clock = _FakeClock()
    monkeypatch.setattr("data.contextualize.time.monotonic", clock)

    slept: list[float] = []

    async def fake_sleep(dt: float) -> None:
        slept.append(dt)
        clock.advance(dt)

    monkeypatch.setattr("data.contextualize.asyncio.sleep", fake_sleep)

    bucket = _TokenBucket(tokens_per_minute=600)  # 10 tokens/sec
    await bucket.acquire(600)  # drains the bucket exactly, no wait needed
    assert bucket.tokens == pytest.approx(0, abs=1e-6)
    assert slept == []

    await bucket.acquire(100)  # exceeds what's available -> must pace via sleep
    assert len(slept) > 0
    assert bucket.tokens == pytest.approx(0, abs=1e-3)


async def test_token_bucket_clamps_request_larger_than_capacity(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A single request bigger than total capacity must not hang forever."""
    clock = _FakeClock()
    monkeypatch.setattr("data.contextualize.time.monotonic", clock)

    async def fake_sleep(dt: float) -> None:
        clock.advance(dt)

    monkeypatch.setattr("data.contextualize.asyncio.sleep", fake_sleep)

    bucket = _TokenBucket(tokens_per_minute=60)
    await asyncio.wait_for(bucket.acquire(10_000), timeout=2.0)
    assert 0 <= bucket.tokens <= bucket.capacity


async def test_retry_recovers_from_transport_error(monkeypatch: pytest.MonkeyPatch) -> None:
    """Regression test: a real run crashed on httpx.ReadTimeout (a connection-level
    failure, not an HTTP error response), which the old retry logic -- only
    catching HTTPStatusError -- let propagate and kill the whole ingestion run."""
    calls = {"n": 0}

    async def fake_chat_completion(
        client: httpx.AsyncClient, api_key: str, model: str, prompt: str, temperature: float
    ) -> str:
        calls["n"] += 1
        if calls["n"] == 1:
            raise httpx.ReadTimeout("simulated timeout")
        return "ok"

    monkeypatch.setattr("data.contextualize._chat_completion", fake_chat_completion)

    async def _no_op_sleep(*_: object) -> None:
        return None

    monkeypatch.setattr("data.contextualize.asyncio.sleep", _no_op_sleep)

    result = await _chat_completion_with_retry(None, "key", "model", "prompt", 0.0, retries=3)  # type: ignore[arg-type]
    assert result == "ok"
    assert calls["n"] == 2


async def test_retry_gives_up_after_exhausting_transport_error_budget(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def always_fails(
        client: httpx.AsyncClient, api_key: str, model: str, prompt: str, temperature: float
    ) -> str:
        raise httpx.ConnectError("simulated connection failure")

    monkeypatch.setattr("data.contextualize._chat_completion", always_fails)

    async def _no_op_sleep(*_: object) -> None:
        return None

    monkeypatch.setattr("data.contextualize.asyncio.sleep", _no_op_sleep)

    with pytest.raises(httpx.ConnectError):
        await _chat_completion_with_retry(None, "key", "model", "prompt", 0.0, retries=3)  # type: ignore[arg-type]
