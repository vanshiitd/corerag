"""Pre-warm the semantic cache with representative demo queries (P6.6).

Posts real requests to a running instance's /query endpoint (not a direct
internal call) so it exercises the exact same code path a real visitor does,
including the real write-through-to-cache logic in api/routes.py -- a
reviewer's first click on one of these questions should be a fast cache hit,
not a cold 12-24s graph run (see PLAN.md P3.8/P4.2 for those real numbers).

Run standalone:
    uv run python -m scripts.prewarm_cache                          # localhost:8000
    uv run python -m scripts.prewarm_cache --base-url https://<space>.hf.space
"""

from __future__ import annotations

import argparse
import asyncio

import httpx
import structlog

from core.config import get_settings
from core.logging import configure_logging

log = structlog.get_logger()

# A handful of genuinely good demo questions -- clearly answerable from the
# real corpus (verified throughout P1-P5's manual/eval checks), spanning
# distinct topics so a reviewer clicking around sees breadth, not one lucky hit.
DEMO_QUERIES = [
    "what is speculative decoding?",
    "how does continuous batching work in LLM serving?",
    "what is a KV cache and how does it affect memory usage in transformers?",
    "what are the tradeoffs of tensor parallelism?",
    "how does quantization reduce inference latency?",
]


async def _prewarm_one(client: httpx.AsyncClient, query: str) -> None:
    async with client.stream("POST", "/query", json={"query": query}) as resp:
        resp.raise_for_status()
        async for _ in resp.aiter_lines():
            pass  # drain the SSE stream so the endpoint's write-through fires
    log.info("prewarm_cache.done", query=query[:60])


async def prewarm(base_url: str, queries: list[str]) -> None:
    async with httpx.AsyncClient(base_url=base_url, timeout=60.0) as client:
        for query in queries:
            try:
                await _prewarm_one(client, query)
            except httpx.HTTPError as exc:
                log.warning("prewarm_cache.failed", query=query[:60], error=str(exc))


def _main() -> None:
    parser = argparse.ArgumentParser(description="Pre-warm the semantic cache")
    parser.add_argument("--base-url", default="http://localhost:8000")
    args = parser.parse_args()
    configure_logging(get_settings())
    asyncio.run(prewarm(args.base_url, DEMO_QUERIES))


if __name__ == "__main__":
    _main()
