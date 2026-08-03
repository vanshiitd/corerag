"""Shared pytest fixtures."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

# Importing `api.main` builds the app (and reads settings) at import time, which
# happens during collection -- before the per-test fixture runs. Seed the required
# secrets here so that import always succeeds.
os.environ.setdefault("GROQ_API_KEY", "gsk-test")
os.environ.setdefault("OPENAI_API_KEY", "sk-test")

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
def _hermetic_env(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Isolate settings tests from the real environment and any on-disk .env."""
    monkeypatch.chdir(tmp_path)  # env_file=".env" now resolves to an empty tmp dir
    for key in list(os.environ):
        if key.startswith(_OWNED_PREFIXES):
            monkeypatch.delenv(key, raising=False)
    # Baseline required secrets; individual tests override or unset as needed.
    monkeypatch.setenv("GROQ_API_KEY", "gsk-test")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
