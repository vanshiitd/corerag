"""Tests for core/cache.py (P4.1).

Integration tests require live Redis 8 (RediSearch module) and make real local
embedding calls. Run with:  make test-int
"""

from __future__ import annotations

import contextlib
from collections.abc import Iterator

import pytest

from core.cache import check_cache, check_cache_async, get_cache, store_in_cache
from core.config import Settings, get_settings
from core.retrieval import ScoredChunk


def _chunk(chunk_id: str = "1.1::0") -> ScoredChunk:
    return ScoredChunk(
        chunk_id=chunk_id,
        arxiv_id="1.1",
        title="Test Paper",
        authors=["A. Author"],
        abs_url="https://arxiv.org/abs/1.1",
        section=None,
        index=0,
        text="some text",
        score=0.9,
    )


def test_get_cache_is_a_singleton_per_config() -> None:
    settings = get_settings()
    assert get_cache(settings) is get_cache(settings)


def test_get_cache_differs_across_cache_versions() -> None:
    a = Settings(cache_version="test-a")
    b = Settings(cache_version="test-b")
    assert get_cache(a) is not get_cache(b)


def test_check_cache_disabled_always_misses() -> None:
    settings = Settings(cache_enabled=False)
    assert check_cache("anything", settings) is None


@pytest.fixture
def cache_settings() -> Settings:
    # A dedicated, disposable cache_version namespace so this test file never
    # collides with real cached answers from actual /query usage.
    return get_settings().model_copy(update={"cache_version": "pytest-cache-test"})


@pytest.fixture
def clean_cache(cache_settings: Settings) -> Iterator[Settings]:
    cache = get_cache(cache_settings)
    with contextlib.suppress(Exception):
        cache.clear()
    yield cache_settings
    with contextlib.suppress(Exception):
        cache.clear()


@pytest.mark.integration
def test_store_then_check_round_trips_answer_and_citations(clean_cache: Settings) -> None:
    settings = clean_cache
    chunk = _chunk()
    store_in_cache("what is speculative decoding?", "a cached answer", [chunk], settings)

    result = check_cache("what is speculative decoding?", settings)

    assert result is not None
    assert result.answer == "a cached answer"
    assert len(result.citations) == 1
    assert result.citations[0].chunk_id == chunk.chunk_id
    assert result.citations[0].title == chunk.title


@pytest.mark.integration
def test_check_cache_misses_on_unrelated_query(clean_cache: Settings) -> None:
    settings = clean_cache
    store_in_cache("what is speculative decoding?", "a cached answer", [], settings)
    assert check_cache("what is the capital of France?", settings) is None


@pytest.mark.integration
def test_check_cache_misses_on_opposite_intent_query(clean_cache: Settings) -> None:
    """Regression test locking in the P4 planning threshold validation: real
    opposite-intent pairs must not false-hit at the configured threshold, through
    the actual module path -- not just the raw embedding-distance check."""
    settings = clean_cache
    store_in_cache(
        "What are the advantages of speculative decoding?", "advantages answer", [], settings
    )
    result = check_cache("What are the disadvantages of speculative decoding?", settings)
    assert result is None


@pytest.mark.integration
def test_check_cache_hits_on_paraphrase(clean_cache: Settings) -> None:
    settings = clean_cache
    store_in_cache("what is speculative decoding?", "cached answer", [], settings)
    result = check_cache("explain speculative decoding", settings)
    assert result is not None
    assert result.answer == "cached answer"


@pytest.mark.integration
async def test_check_cache_async_matches_sync(clean_cache: Settings) -> None:
    settings = clean_cache
    store_in_cache("what is speculative decoding?", "cached answer", [], settings)
    result = await check_cache_async("what is speculative decoding?", settings)
    assert result is not None
    assert result.answer == "cached answer"
