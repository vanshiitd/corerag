"""FastAPI dependency providers -- typed accessors for objects on app.state."""

from __future__ import annotations

from typing import cast

import redis.asyncio as aioredis
from fastapi import Request
from langgraph.graph.state import CompiledStateGraph
from qdrant_client import AsyncQdrantClient

from core.config import Settings


def get_settings(request: Request) -> Settings:
    """Return the process-wide settings singleton."""
    return cast(Settings, request.app.state.settings)


def get_graph(request: Request) -> CompiledStateGraph:
    """Return the process-wide compiled agent graph."""
    return cast(CompiledStateGraph, request.app.state.graph)


def get_qdrant(request: Request) -> AsyncQdrantClient:
    """Return the process-wide async Qdrant client."""
    return cast(AsyncQdrantClient, request.app.state.qdrant)


def get_redis(request: Request) -> aioredis.Redis:
    """Return the process-wide async Redis client."""
    return cast(aioredis.Redis, request.app.state.redis)
