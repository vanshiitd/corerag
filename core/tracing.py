"""Langfuse tracing setup.

Langfuse's SDK reads credentials from ``os.environ`` directly, but this codebase
centralizes config through :class:`core.config.Settings` (pydantic-settings loads
``.env`` internally, never exporting to the real process environment) -- so the
bridge below is required, not cosmetic. Without it, ``CallbackHandler()`` silently
disables itself ("Client will be disabled") rather than erroring -- confirmed
empirically in P3.0 (a real trace call returned an all-zero trace ID).
"""

from __future__ import annotations

import os
from functools import lru_cache

from langfuse.langchain import CallbackHandler

from core.config import Settings


@lru_cache(maxsize=1)
def _bridge_env(public_key: str, secret_key: str, base_url: str) -> None:
    """Export Langfuse credentials into the real process environment, once."""
    os.environ["LANGFUSE_PUBLIC_KEY"] = public_key
    os.environ["LANGFUSE_SECRET_KEY"] = secret_key
    os.environ["LANGFUSE_BASE_URL"] = base_url


def get_tracing_handler(settings: Settings) -> CallbackHandler | None:
    """A fresh CallbackHandler for one request, or None if tracing is disabled.

    A new instance per request/graph-invocation, not a shared singleton -- per
    Langfuse's own guidance for correct trace attribution.
    """
    if not settings.langfuse_enabled:
        return None
    # langfuse_enabled guarantees both keys are set (see Settings.langfuse_enabled).
    assert settings.langfuse_public_key is not None
    assert settings.langfuse_secret_key is not None
    _bridge_env(
        settings.langfuse_public_key.get_secret_value(),
        settings.langfuse_secret_key.get_secret_value(),
        settings.langfuse_base_url,
    )
    return CallbackHandler()
