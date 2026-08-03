"""Tests for core/llm.py (P3.1).

Unit tests cover model construction (no network call); integration tests make
real API calls to confirm the factories actually work end to end.
Run integration with:  make test-int
"""

from __future__ import annotations

import pytest
from langchain_groq import ChatGroq
from langchain_openai import ChatOpenAI

from core.config import Settings, get_settings
from core.llm import get_agent_model, get_generation_model


def test_get_agent_model_uses_configured_model_name() -> None:
    settings = Settings(agent_model="gpt-4o-mini", agent_temperature=0.3)
    model = get_agent_model(settings)
    assert isinstance(model, ChatOpenAI)
    assert model.model_name == "gpt-4o-mini"
    assert model.temperature == 0.3


def test_get_generation_model_uses_configured_model_name() -> None:
    settings = Settings(generation_model="llama-3.3-70b-versatile", generation_temperature=0.2)
    model = get_generation_model(settings)
    assert isinstance(model, ChatGroq)
    assert model.model_name == "llama-3.3-70b-versatile"
    assert model.temperature == 0.2


@pytest.mark.integration
def test_agent_model_real_invoke() -> None:
    model = get_agent_model(get_settings())
    result = model.invoke("Reply with exactly one word: OK")
    assert "OK" in str(result.content)


@pytest.mark.integration
def test_generation_model_real_invoke() -> None:
    model = get_generation_model(get_settings())
    result = model.invoke("Reply with exactly one word: OK")
    assert "OK" in str(result.content)
