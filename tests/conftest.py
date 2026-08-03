"""Shared pytest fixtures."""

from __future__ import annotations

import os
from pathlib import Path

import pytest
from dotenv import dotenv_values

from core.config import get_settings

# Importing `api.main` builds the app (and reads settings) at import time, which
# happens during collection -- before the per-test fixture runs. Seed real secrets
# from .env if present, falling back to dummy placeholders only when there's no
# .env (e.g. CI). A prior version hardcoded the dummy values unconditionally via
# plain `os.environ.setdefault` -- a real bug (found in P3.1): since that's a raw
# assignment, not `monkeypatch`, it permanently shadowed the real keys for the
# *entire* test session, silently breaking every integration test that needs a
# real cloud LLM/Langfuse credential, not just local services.
_dotenv_secrets = dotenv_values(".env")
os.environ.setdefault("GROQ_API_KEY", _dotenv_secrets.get("GROQ_API_KEY") or "gsk-test")
os.environ.setdefault("OPENAI_API_KEY", _dotenv_secrets.get("OPENAI_API_KEY") or "sk-test")

# Env-var prefixes owned by CoreRAG settings. Cleared before each test so a
# developer's real environment (or a local .env) cannot leak into assertions.
_OWNED_PREFIXES = (
    "APP_",
    "ENVIRONMENT",
    "LOG_",
    "ARXIV_",
    "CHUNK_",
    "DENSE_EMBED_",
    "SPARSE_EMBED_",
    "RERANKER_",
    "RETRIEVAL_",
    "RERANK_TOP_N",
    "GROQ_",
    "GENERATION_",
    "OPENAI_",
    "OPENROUTER_",
    "CONTEXTUALIZE_",
    "AGENT_",
    "MAX_REFLECTION_",
    "GRADER_",
    "CACHE_",
    "QDRANT_",
    "REDIS_",
    "GROBID_",
    "LANGFUSE_",
)


@pytest.fixture(autouse=True)
def _hermetic_env(
    request: pytest.FixtureRequest, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Isolate unit-test settings from the real environment and any on-disk .env.

    Integration tests are exempted: they exercise real services (Qdrant, Redis,
    GROBID, and -- since P3.0/P3.1 -- real cloud LLM/Langfuse APIs), so they need
    the developer's actual .env, not a hermetic dummy one. get_settings() is a
    process-wide lru_cache; clearing it here on both paths stops a Settings
    object from an earlier test (dummy or real) leaking into this one regardless
    of which env is currently active.
    """
    get_settings.cache_clear()
    if request.node.get_closest_marker("integration") is not None:
        return

    monkeypatch.chdir(tmp_path)  # env_file=".env" now resolves to an empty tmp dir
    for key in list(os.environ):
        if key.startswith(_OWNED_PREFIXES):
            monkeypatch.delenv(key, raising=False)
    # Baseline required secrets; individual tests override or unset as needed.
    monkeypatch.setenv("GROQ_API_KEY", "gsk-test")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
