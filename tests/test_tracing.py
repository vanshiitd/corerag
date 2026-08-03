"""Tests for core/tracing.py (P3.1).

Regression coverage for the P3.0 finding: Langfuse's SDK reads credentials from
os.environ directly, so without the bridge in get_tracing_handler, tracing
silently no-ops instead of erroring. The integration test locks in the real,
empirically-verified fix.
Run integration with:  make test-int
"""

from __future__ import annotations

import os

import pytest
from langfuse.langchain import CallbackHandler

from core.config import Settings
from core.tracing import get_tracing_handler


def test_returns_none_when_langfuse_disabled() -> None:
    settings = Settings(langfuse_public_key=None, langfuse_secret_key=None)
    assert settings.langfuse_enabled is False
    assert get_tracing_handler(settings) is None


def test_bridges_env_and_returns_handler_when_enabled(monkeypatch: pytest.MonkeyPatch) -> None:
    # Seed unique values so Settings() can populate its SecretStr fields, then
    # remove them from os.environ before calling get_tracing_handler. If the
    # assertions below pass, it's because the bridge re-established them from
    # `settings` -- not because monkeypatch.setenv left them there already.
    monkeypatch.setenv("LANGFUSE_PUBLIC_KEY", "pk-test-unique-001")
    monkeypatch.setenv("LANGFUSE_SECRET_KEY", "sk-test-unique-002")
    monkeypatch.setenv("LANGFUSE_BASE_URL", "https://example.langfuse.test")
    settings = Settings()

    monkeypatch.delenv("LANGFUSE_PUBLIC_KEY", raising=False)
    monkeypatch.delenv("LANGFUSE_SECRET_KEY", raising=False)
    monkeypatch.delenv("LANGFUSE_BASE_URL", raising=False)

    handler = get_tracing_handler(settings)

    assert isinstance(handler, CallbackHandler)
    assert os.environ.get("LANGFUSE_PUBLIC_KEY") == "pk-test-unique-001"
    assert os.environ.get("LANGFUSE_SECRET_KEY") == "sk-test-unique-002"
    assert os.environ.get("LANGFUSE_BASE_URL") == "https://example.langfuse.test"


@pytest.mark.integration
def test_real_trace_is_recorded_in_langfuse_cloud() -> None:
    """Regression test for the exact P3.0 bug: a real call must produce a genuine
    (non-zero) trace ID, not silently no-op with 'Client will be disabled'."""
    from langfuse import get_client

    from core.config import get_settings
    from core.llm import get_agent_model

    settings = get_settings()
    model = get_agent_model(settings)
    handler = get_tracing_handler(settings)
    assert handler is not None

    model.invoke("say hi", config={"callbacks": [handler]})
    get_client().flush()

    trace_id = handler.last_trace_id
    assert trace_id is not None
    assert trace_id != "00000000000000000000000000000000"
    assert len(trace_id) == 32
