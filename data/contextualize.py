"""LLM contextualization (Anthropic-style Contextual Retrieval), via OpenAI.

For each chunk, ask a cheap model for a short blurb situating the chunk within its
parent document; the blurb is prepended before embedding (see ``Chunk.embed_input``).
Two strategies, chosen by ``settings.contextualize_strategy``:

  - "doc_summary": one call per document; the same summary is prepended to every
    chunk. Cheap (O(1) calls/doc) but coarse -- the P5 ablation baseline.
  - "per_chunk": one call per chunk, given the *whole* document + that chunk. Finer
    grained, but O(chunks) calls/doc. The document text is an identical, stable
    prefix across a document's chunk calls, so OpenAI's automatic prompt caching
    (>=1024-token matching prefix) discounts repeat input by ~50%.

Cost is estimated up front via tiktoken; see :func:`estimate_cost`.
"""

from __future__ import annotations

import asyncio
import random
import time
from functools import lru_cache

import httpx
import structlog
import tiktoken

from core.config import Settings
from data.schemas import Chunk, ParsedDoc

log = structlog.get_logger()


class _TokenBucket:
    """Proactive TPM limiter.

    Retry-after-429 alone isn't enough once demand is *structurally* above the
    account's rate limit (observed live: a single large paper's per-chunk calls can
    approach the whole 200k/min budget by itself). This paces our own send rate
    below the configured budget instead of bursting and error-correcting after the
    fact. Lock-free by design: the check-and-decrement below never crosses an
    `await`, so concurrent callers on the same event loop can't race it.
    """

    def __init__(self, tokens_per_minute: float) -> None:
        self.capacity = tokens_per_minute
        self.tokens = tokens_per_minute
        self.last_refill = time.monotonic()

    async def acquire(self, estimated_tokens: float) -> None:
        # A request larger than total capacity could never be satisfied and would
        # loop forever; clamp it so we wait for a full bucket and proceed instead.
        estimated_tokens = min(estimated_tokens, self.capacity)
        while True:
            now = time.monotonic()
            self.tokens = min(
                self.capacity, self.tokens + (now - self.last_refill) * (self.capacity / 60)
            )
            self.last_refill = now
            if self.tokens >= estimated_tokens:
                self.tokens -= estimated_tokens
                return
            wait = (estimated_tokens - self.tokens) / (self.capacity / 60)
            await asyncio.sleep(min(max(wait, 0.1), 5))


@lru_cache(maxsize=1)
def _bucket(tokens_per_minute: int) -> _TokenBucket:
    # A plain object (no event-loop-bound asyncio primitives), so it survives across
    # the pipeline's per-paper `asyncio.run()` calls -- one shared budget per process.
    return _TokenBucket(tokens_per_minute)


_API_URL = "https://api.openai.com/v1/chat/completions"

# gpt-4o-mini pricing, $/1M tokens -- verified live 2026-08-03 (openai.com/api/pricing).
_PRICE_INPUT = 0.15
_PRICE_CACHED_INPUT = 0.075
_PRICE_OUTPUT = 0.60

_DOC_SUMMARY_PROMPT = (
    "Summarize the following paper in 1-2 sentences, stating its core contribution "
    "and problem domain. Answer only with the summary, nothing else.\n\n"
    "<document>\n{document}\n</document>"
)

_PER_CHUNK_PROMPT = (
    "<document>\n{document}\n</document>\n"
    "Here is the chunk we want to situate within the whole document\n"
    "<chunk>\n{chunk}\n</chunk>\n"
    "Please give a short succinct context (1-2 sentences) to situate this chunk "
    "within the overall document for the purposes of improving search retrieval "
    "of the chunk. Answer only with the succinct context and nothing else."
)


@lru_cache(maxsize=1)
def _encoder() -> tiktoken.Encoding:
    return tiktoken.get_encoding("o200k_base")  # gpt-4o family encoding


def _encode(enc: tiktoken.Encoding, text: str) -> list[int]:
    """Encode arbitrary paper text for local counting/truncation.

    Papers about tokenization legitimately contain literal special-token strings
    (e.g. "<|endoftext|>"), which tiktoken rejects by default as an injection
    guard. That guard protects prompt construction; here we're only counting and
    truncating text locally, so disable it and treat the string as plain text.
    """
    return enc.encode(text, disallowed_special=())


def _bounded_doc_text(doc: ParsedDoc, settings: Settings) -> str:
    """Front-truncate the document to contextualize_doc_token_budget tokens."""
    enc = _encoder()
    tokens = _encode(enc, doc.full_text)
    if len(tokens) <= settings.contextualize_doc_token_budget:
        return doc.full_text
    return enc.decode(tokens[: settings.contextualize_doc_token_budget])


def estimate_cost(doc: ParsedDoc, chunks: list[Chunk], settings: Settings) -> dict[str, float]:
    """Pre-flight token/cost estimate for contextualizing one document.

    Assumes the OpenAI prompt cache hits on every repeat of the (stable) document
    prefix -- an optimistic but directionally correct estimate, since the pipeline
    fires a document's chunk calls back-to-back. Reflects the same document
    truncation applied at call time (see ``_bounded_doc_text``).
    """
    strategy = settings.contextualize_strategy
    if strategy == "none" or not chunks:
        return {"calls": 0, "uncached_input": 0, "cached_input": 0, "output": 0, "usd": 0.0}

    enc = _encoder()

    if strategy == "doc_summary":
        calls = 1
        doc_tokens = len(_encode(enc, doc.full_text))  # single call: full text is fine
        uncached_input = doc_tokens + 30
        cached_input = 0
    else:  # per_chunk
        calls = len(chunks)
        doc_tokens = min(len(_encode(enc, doc.full_text)), settings.contextualize_doc_token_budget)
        chunk_tokens = sum(len(_encode(enc, c.text)) for c in chunks)
        uncached_input = doc_tokens + chunk_tokens  # doc paid once; chunk text never cached
        cached_input = doc_tokens * max(calls - 1, 0)  # doc resent every later call, cached

    output = calls * 40  # ~1-2 sentence replies
    usd = (
        uncached_input * _PRICE_INPUT + cached_input * _PRICE_CACHED_INPUT + output * _PRICE_OUTPUT
    ) / 1_000_000
    return {
        "calls": calls,
        "uncached_input": uncached_input,
        "cached_input": cached_input,
        "output": output,
        "usd": round(usd, 4),
    }


async def _chat_completion(
    client: httpx.AsyncClient, api_key: str, model: str, prompt: str, temperature: float
) -> str:
    resp = await client.post(
        _API_URL,
        headers={"Authorization": f"Bearer {api_key}"},
        json={
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": temperature,
            "max_tokens": 150,
        },
        timeout=60.0,
    )
    resp.raise_for_status()
    content: str = resp.json()["choices"][0]["message"]["content"]
    return content.strip()


async def _chat_completion_with_retry(
    client: httpx.AsyncClient,
    api_key: str,
    model: str,
    prompt: str,
    temperature: float,
    retries: int = 10,
) -> str:
    for attempt in range(retries):
        try:
            return await _chat_completion(client, api_key, model, prompt, temperature)
        except httpx.HTTPStatusError as exc:
            transient = exc.response.status_code in (429, 500, 502, 503)
            if not transient or attempt >= retries - 1:
                raise
            retry_after = exc.response.headers.get("retry-after")
            wait = float(retry_after) if retry_after else (2**attempt)
            log.warning(
                "contextualize.retry",
                status=exc.response.status_code,
                attempt=attempt,
                wait_s=round(wait, 1),
            )
            await asyncio.sleep(wait + random.random())
        except httpx.TransportError as exc:
            # Connection-level failures (read timeouts, resets, DNS blips -- e.g.
            # from a sleep/wake cycle or a transient network hiccup) carry no
            # Retry-After header and are always worth retrying; HTTPStatusError
            # above is a *response*, this is a failure to even get one.
            if attempt >= retries - 1:
                raise
            wait = min(2**attempt, 30)
            log.warning(
                "contextualize.retry_transport_error",
                error=str(exc),
                attempt=attempt,
                wait_s=round(wait, 1),
            )
            await asyncio.sleep(wait + random.random())
    raise RuntimeError("unreachable")  # loop always returns or raises


async def contextualize_chunks(
    doc: ParsedDoc, chunks: list[Chunk], settings: Settings
) -> list[Chunk]:
    """Populate ``chunk.context`` for every chunk, per ``settings.contextualize_strategy``."""
    strategy = settings.contextualize_strategy
    if strategy == "none" or not chunks:
        return chunks

    api_key = settings.openai_api_key.get_secret_value()
    concurrency = asyncio.Semaphore(settings.contextualize_concurrency)
    bucket = _bucket(settings.contextualize_tokens_per_minute)
    enc = _encoder()

    async with httpx.AsyncClient() as client:
        if strategy == "doc_summary":
            prompt = _DOC_SUMMARY_PROMPT.format(document=doc.full_text)
            await bucket.acquire(len(_encode(enc, prompt)) + 150)
            summary = await _chat_completion_with_retry(
                client, api_key, settings.contextualize_model, prompt, 0.0
            )
            return [c.model_copy(update={"context": summary}) for c in chunks]

        document = _bounded_doc_text(doc, settings)

        async def _one(chunk: Chunk) -> Chunk:
            prompt = _PER_CHUNK_PROMPT.format(document=document, chunk=chunk.text)
            async with concurrency:
                await bucket.acquire(len(_encode(enc, prompt)) + 150)
                context = await _chat_completion_with_retry(
                    client, api_key, settings.contextualize_model, prompt, 0.0
                )
                return chunk.model_copy(update={"context": context})

        return list(await asyncio.gather(*[_one(c) for c in chunks]))
