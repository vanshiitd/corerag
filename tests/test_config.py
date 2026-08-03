"""Tests for the application settings object (P0.3).

These exercise the real environment-loading path; the autouse ``_hermetic_env``
fixture (see ``conftest.py``) isolates each test from the developer's environment
and any on-disk ``.env``.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from core.config import Settings, get_settings


def test_defaults_match_locked_decisions() -> None:
    s = Settings()
    # Locked model choices
    assert s.dense_embed_model == "BAAI/bge-base-en-v1.5"
    assert s.reranker_model == "Alibaba-NLP/gte-reranker-modernbert-base"
    assert s.agent_model == "gpt-4o-mini"
    assert s.generation_model.startswith("llama-3.3")
    # Locked retrieval / cache knobs
    assert s.retrieval_k == 30
    assert s.rerank_top_n == 5
    assert s.cache_similarity_threshold == 0.95
    assert s.max_reflection_retries == 2
    # Corpus filter
    assert s.arxiv_categories == ["cs.DC", "cs.AR", "cs.LG"]
    assert "inference serving" in s.arxiv_cslg_keywords
    assert "inference" not in s.arxiv_cslg_keywords  # bare, overloaded term excluded


def test_secrets_do_not_leak_in_repr() -> None:
    s = Settings()
    assert s.groq_api_key.get_secret_value() == "gsk-test"  # set by the fixture
    assert "gsk-test" not in repr(s)  # SecretStr must mask its value


def test_derived_properties(monkeypatch: pytest.MonkeyPatch) -> None:
    assert Settings().qdrant_collection_name == "corerag_v1"
    assert Settings().langfuse_enabled is False
    monkeypatch.setenv("LANGFUSE_PUBLIC_KEY", "pk")
    monkeypatch.setenv("LANGFUSE_SECRET_KEY", "sk")
    assert Settings().langfuse_enabled is True


@pytest.mark.parametrize(
    ("env_var", "value"),
    [
        ("CACHE_SIMILARITY_THRESHOLD", "1.5"),
        ("GRADER_RELEVANCE_THRESHOLD", "-0.1"),
        ("MAX_REFLECTION_RETRIES", "-1"),
        ("RETRIEVAL_K", "0"),
        ("GENERATION_TEMPERATURE", "3.0"),
    ],
)
def test_rejects_out_of_range(monkeypatch: pytest.MonkeyPatch, env_var: str, value: str) -> None:
    monkeypatch.setenv(env_var, value)
    with pytest.raises(ValidationError):
        Settings()


def test_rejects_top_n_greater_than_k(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("RERANK_TOP_N", "100")
    monkeypatch.setenv("RETRIEVAL_K", "50")
    with pytest.raises(ValidationError):
        Settings()


def test_rejects_overlap_ge_chunk_size(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("CHUNK_OVERLAP_TOKENS", "512")
    monkeypatch.setenv("CHUNK_SIZE_TOKENS", "512")
    with pytest.raises(ValidationError):
        Settings()


@pytest.mark.parametrize("env_var", ["GROQ_API_KEY", "OPENAI_API_KEY"])
def test_missing_required_secret_raises(monkeypatch: pytest.MonkeyPatch, env_var: str) -> None:
    monkeypatch.delenv(env_var, raising=False)
    with pytest.raises(ValidationError):
        Settings()


def test_openrouter_key_is_optional_and_unenforced(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    s = Settings()  # must not raise -- OpenRouter is currently inactive
    assert s.openrouter_api_key is None


def test_get_settings_is_cached() -> None:
    get_settings.cache_clear()
    assert get_settings() is get_settings()
