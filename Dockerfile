# syntax=docker/dockerfile:1
#
# Production image for the CoreRAG API (P6.1). Single-stage: uv itself is
# lightweight and nearly everything installs from wheels, so a multi-stage
# split buys little here and would complicate copying the HF/FastEmbed model
# caches (baked in below) across stages correctly.
#
# The non-root user is created FIRST, and every later COPY/RUN either runs as
# that user or writes with --chown -- not a single `chown -R` at the end.
# Confirmed live this matters, not just style: a trailing `chown -R /app` after
# the dependency-install and model-download layers added a *second* ~2.65GB
# layer (overlayfs treats a metadata-only ownership change as a copy-up of
# every file), nearly doubling image size for zero content change.
#
# Build (defaults to the current prod reranker_mode; override after P6.5's
# real x86 benchmark if it picks a different mode):
#   docker build -t corerag-api .
#   docker build -t corerag-api --build-arg RERANKER_MODE=cpu-onnx .
#
# Run locally:
#   docker run --rm -p 8000:8000 -e PORT=8000 --env-file .env corerag-api

FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=never \
    PYTHONUNBUFFERED=1

RUN useradd --create-home --uid 1000 corerag
WORKDIR /app
RUN chown corerag:corerag /app
USER corerag

# Dependency layer first (cache-friendly): pyproject.toml/uv.lock change far
# less often than application code. uid/gid on the cache mount: BuildKit cache
# mounts default to root-owned, which a non-root USER can't write into.
COPY --chown=corerag:corerag pyproject.toml uv.lock ./
RUN --mount=type=cache,target=/home/corerag/.cache/uv,uid=1000,gid=1000 \
    uv sync --frozen --no-dev --no-install-project

COPY --chown=corerag:corerag api/ ./api/
COPY --chown=corerag:corerag core/ ./core/
COPY --chown=corerag:corerag data/ ./data/
RUN rm -rf ./data/raw
RUN --mount=type=cache,target=/home/corerag/.cache/uv,uid=1000,gid=1000 \
    uv sync --frozen --no-dev

# Pre-download and warm the local models (dense/sparse embedders + reranker)
# at build time, not on first request -- HF Spaces sleeps after 48h idle
# (PLAN.md P6.0), so without this every wake-from-sleep would pay real model
# download latency on top of the load itself. Placeholder LLM keys satisfy
# Settings' presence validation without making any real API call -- this step
# only exercises local model loading, never contacts OpenAI/Groq.
#
# Cache dirs pinned explicitly under /app (not the default ~/.cache-relative
# locations): HuggingFace Hub honors HF_HOME, but FastEmbed does NOT -- verified
# directly in its source (fastembed/common/utils.py's define_cache_dir) that it
# defaults to {tempdir}/fastembed_cache regardless of HF_HOME, controlled only
# by its own FASTEMBED_CACHE_PATH. Running this step as `corerag` (see USER
# above) means both land pre-owned by the runtime user -- no chown needed.
ARG RERANKER_MODE=pytorch-cpu
ENV GROQ_API_KEY=build-time-placeholder \
    OPENAI_API_KEY=build-time-placeholder \
    RERANKER_MODE=${RERANKER_MODE} \
    HF_HOME=/app/.cache/huggingface \
    FASTEMBED_CACHE_PATH=/app/.cache/fastembed
RUN uv run python -c "\
from core.config import Settings; \
from core.reranker import rerank; \
from core.retrieval import ScoredChunk; \
from data.embeddings import embed_dense, embed_sparse; \
settings = Settings(); \
embed_dense(['warmup'], settings); \
embed_sparse(['warmup'], settings); \
chunk = ScoredChunk(chunk_id='w::0', arxiv_id='w', title='w', authors=[], abs_url='https://x', section=None, index=0, text='warmup text', score=0.0); \
rerank('warmup query', [chunk], settings)"

# Cloud Run injects its own PORT env var at deploy time (default 8080) and
# expects the container to listen on it -- the CMD below reads ${PORT}
# dynamically, so this default only matters for local `docker run`.
ENV PORT=8080
EXPOSE 8080

CMD ["sh", "-c", "uv run uvicorn api.main:app --host 0.0.0.0 --port ${PORT}"]
