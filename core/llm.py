"""Provider-agnostic chat model factories.

Two distinct LLM roles, config-driven so providers/models swap with zero code
change: a cheap, reliable model for router/grader structured output (OpenAI), and
a low-latency streaming model for final answer generation (Groq).
"""

from __future__ import annotations

from langchain_core.language_models import BaseChatModel
from langchain_groq import ChatGroq
from langchain_openai import ChatOpenAI

from core.config import Settings


def get_agent_model(settings: Settings) -> BaseChatModel:
    """Router/grader model: cheap, reliable structured output."""
    return ChatOpenAI(
        model=settings.agent_model,
        api_key=settings.openai_api_key,
        temperature=settings.agent_temperature,
    )


def get_generation_model(settings: Settings) -> BaseChatModel:
    """Final answer generation model: low-latency token streaming."""
    return ChatGroq(
        model=settings.generation_model,
        api_key=settings.groq_api_key,
        temperature=settings.generation_temperature,
        max_tokens=settings.generation_max_tokens,
    )
