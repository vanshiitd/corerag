"""FastAPI application entrypoint."""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

import structlog
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from api.routes import router
from core import __version__
from core.clients import make_qdrant_client, make_redis_client
from core.config import get_settings
from core.logging import configure_logging

log = structlog.get_logger()


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Build clients on startup, dispose them on shutdown."""
    settings = get_settings()
    configure_logging(settings)
    app.state.settings = settings
    app.state.qdrant = make_qdrant_client(settings)
    app.state.redis = make_redis_client(settings)
    log.info(
        "startup",
        qdrant=settings.qdrant_url,
        redis=settings.redis_url,
        langfuse_enabled=settings.langfuse_enabled,
    )
    try:
        yield
    finally:
        await app.state.qdrant.close()
        await app.state.redis.aclose()
        log.info("shutdown")


def create_app() -> FastAPI:
    """Application factory."""
    settings = get_settings()
    app = FastAPI(title=settings.app_name, version=__version__, lifespan=lifespan)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],  # tighten for the hosted demo
        allow_methods=["*"],
        allow_headers=["*"],
    )
    app.include_router(router)
    return app


app = create_app()
