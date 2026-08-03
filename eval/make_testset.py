"""Golden Q/A set generation (P5.2): synthesize questions grounded in real indexed
chunks, for use as ground truth by retrieval_eval.py and ragas_eval.py.

Generates up to one question per sampled chunk (not per paper): ragas's
TestsetGenerator does its own internal document splitting, which would produce
reference contexts that don't line up with our own Qdrant chunk boundaries.
Treating each sampled chunk as its own input Document sidesteps that entirely --
the generated sample's ground truth chunk is exactly the one we sampled, no fuzzy
text-matching needed to recover it. --n is the number of chunks *sampled*, not the
guaranteed output count: ragas's CustomNodeFilter drops chunks whose content
doesn't support a good enough summary (observed live: roughly 40-50% yield), so
--n should be sampled generously above the target golden-set size.

Run standalone:
    uv run python -m eval.make_testset --n 5 --dry-run   # cost estimate, no spend
    uv run python -m eval.make_testset --n 5              # small real run
    uv run python -m eval.make_testset --n 300             # ~150-question golden set
"""

from __future__ import annotations

import argparse
import asyncio
import json
import random
from pathlib import Path
from typing import Any

import structlog
from langchain_core.documents import Document
from openai import APIConnectionError
from qdrant_client import QdrantClient

from core.config import Settings, get_settings
from core.logging import configure_logging
from data.index import make_client
from eval.ragas_compat import ragas_embeddings, ragas_llm

log = structlog.get_logger()

_DEFAULT_OUT = Path("eval/golden_set.jsonl")
# Fixed, not random: re-running without --force reproduces the same sample, so the
# golden set is a stable fixture rather than a moving target across regenerations.
_SAMPLE_SEED = 42
# gpt-4o-mini pricing, $/1M tokens -- verified live 2026-08-03 (openai.com/api/pricing),
# same constants as data/contextualize.py. TestsetGenerator's internal transforms
# (summarization, theme/keyphrase extraction, question synthesis) are all LLM calls
# against the *sampled chunk text* (small, ~512 tokens), not the full corpus.
_PRICE_INPUT = 0.15
_PRICE_OUTPUT = 0.60
# Rough per-chunk call/token budget for TestsetGenerator's transform pipeline
# (summary + headline + question synthesis + a critique/filter pass) -- an
# estimate for --dry-run, not a hard accounting like contextualize.py's exact
# tiktoken-based one; ragas doesn't expose its own token usage cheaply up front.
_EST_CALLS_PER_CHUNK = 4
_EST_INPUT_TOKENS_PER_CALL = 700
_EST_OUTPUT_TOKENS_PER_CALL = 120
# ragas's own tenacity retry (ragas.llms.base.LangchainLLMWrapper) only catches
# openai.RateLimitError, not a transient APIConnectionError; this is a backstop
# for whatever genuine network blips remain even after the fix below.
_GENERATE_MAX_ATTEMPTS = 3


def _scroll_all_chunks(client: QdrantClient, settings: Settings) -> list[dict[str, object]]:
    payloads: list[dict[str, object]] = []
    offset = None
    while True:
        points, offset = client.scroll(
            settings.qdrant_collection_name,
            limit=256,
            offset=offset,
            with_payload=True,
            with_vectors=False,
        )
        payloads.extend(p.payload for p in points if p.payload is not None)
        if offset is None:
            break
    return payloads


def sample_chunks(settings: Settings, n: int, seed: int = _SAMPLE_SEED) -> list[dict[str, object]]:
    """Reproducibly sample n chunk payloads from the real indexed corpus."""
    client = make_client(settings)
    try:
        chunks = _scroll_all_chunks(client, settings)
    finally:
        client.close()
    if n >= len(chunks):
        return chunks
    return random.Random(seed).sample(chunks, n)


def estimate_cost(n_chunks: int) -> dict[str, float]:
    calls = n_chunks * _EST_CALLS_PER_CHUNK
    input_tokens = calls * _EST_INPUT_TOKENS_PER_CALL
    output_tokens = calls * _EST_OUTPUT_TOKENS_PER_CALL
    usd = (input_tokens * _PRICE_INPUT + output_tokens * _PRICE_OUTPUT) / 1_000_000
    return {
        "calls": calls,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "usd": round(usd, 4),
    }


def generate_golden_set(
    settings: Settings, n: int, seed: int = _SAMPLE_SEED
) -> list[dict[str, object]]:
    """Generate n (question, reference_answer, relevant_chunk_id) golden samples."""
    from ragas.testset import Testset, TestsetGenerator  # deferred: needs the compat shim first
    from ragas.testset.synthesizers.single_hop.specific import (
        SingleHopSpecificQuerySynthesizer,
    )

    sampled = sample_chunks(settings, n, seed=seed)
    documents = [
        Document(
            page_content=str(c["text"]),
            metadata={"chunk_id": c["chunk_id"], "arxiv_id": c["arxiv_id"], "title": c["title"]},
        )
        for c in sampled
    ]
    log.info("make_testset.generating", n_documents=len(documents))

    llm = ragas_llm(settings)
    generator = TestsetGenerator(llm=llm, embedding_model=ragas_embeddings(settings))
    # Single-hop only: sampled chunks are independent (drawn from across the whole
    # corpus, no relation to each other), but ragas's default query_distribution
    # also mixes in multi-hop synthesizers, which -- confirmed live -- clustered
    # unrelated chunks together and synthesized questions that fictionally connect
    # two different papers' concepts (e.g. one real generated question conflated an
    # attention-head projection matrix with an unrelated statistics paper's
    # "structural floor" metric). Those have no single ground-truth chunk, which is
    # exactly what retrieval_eval.py's hit@k/MRR/nDCG need.
    query_distribution: list[tuple[Any, float]] = [
        (SingleHopSpecificQuerySynthesizer(llm=llm), 1.0)
    ]

    # Confirmed live root cause of the (otherwise mysterious, intermittent)
    # openai.APIConnectionError failures: generate_with_langchain_docs is a *sync*
    # function whose internal stages (SummaryExtractor, NERExtractor, ThemesExtractor,
    # scenario/sample generation, ...) each go through ragas.async_utils.run(), which
    # -- when no event loop is already running -- calls a *fresh* asyncio.run() per
    # stage. The shared ChatOpenAI/httpx async client we pass in gets bound to
    # whichever loop was running on first use; once that loop closes at the end of
    # its stage, the next stage's new loop can't reuse it, and the underlying socket
    # raises "RuntimeError: Event loop is closed" -> surfaces as APIConnectionError.
    # This exact failure mode is a widely-reported open ragas issue (e.g.
    # explodinggradients/ragas#1381, #1433, #1512, #1718), not specific to this
    # network. ragas's own run() has the fix already built in, just not engaged by
    # default: if a loop IS already running when it's called, it detects that and
    # applies nest_asyncio to reuse the *same* loop for every stage instead. Running
    # our whole call inside one outer asyncio.run() makes that loop "already
    # running" for every nested ragas.async_utils.run() call, so they all share it.
    # Confirmed live: 0 connection errors across 3 repeated runs after this fix,
    # vs. failing on ~2 of every 3 attempts before it.
    async def _generate() -> Any:
        return generator.generate_with_langchain_docs(
            documents, testset_size=len(documents), query_distribution=query_distribution
        )

    testset = None
    for attempt in range(_GENERATE_MAX_ATTEMPTS):
        try:
            testset = asyncio.run(_generate())
            break
        except APIConnectionError:
            if attempt >= _GENERATE_MAX_ATTEMPTS - 1:
                raise
            wait = 2**attempt
            log.warning("make_testset.retry_connection_error", attempt=attempt, wait_s=wait)
            asyncio.run(asyncio.sleep(wait))
    # return_executor defaults to False, so this is always a Testset at runtime;
    # the declared union return type is just an unresolved overload in ragas.
    assert isinstance(testset, Testset)
    df = testset.to_pandas()

    # Map each generated sample back to its source chunk_id via the document
    # metadata ragas preserves per node, not by re-matching text -- exact, not fuzzy.
    by_content = {d.page_content: d.metadata for d in documents}
    rows: list[dict[str, object]] = []
    for _, row in df.iterrows():
        contexts = row.get("reference_contexts") or []
        chunk_ids = sorted(
            {by_content[c]["chunk_id"] for c in contexts if c in by_content},
        )
        rows.append(
            {
                "question": row["user_input"],
                "reference": row.get("reference", ""),
                "relevant_chunk_ids": chunk_ids,
            }
        )
    return rows


def _main() -> None:
    parser = argparse.ArgumentParser(description="CoreRAG golden Q/A set generation")
    parser.add_argument(
        "--n",
        type=int,
        default=300,
        help="number of chunks to sample (yield is lower -- see module docstring)",
    )
    parser.add_argument("--seed", type=int, default=_SAMPLE_SEED)
    parser.add_argument("--out", type=Path, default=_DEFAULT_OUT)
    parser.add_argument("--dry-run", action="store_true", help="estimate cost; no LLM calls")
    args = parser.parse_args()
    configure_logging(get_settings())
    settings = get_settings()

    if args.dry_run:
        cost = estimate_cost(args.n)
        log.info("make_testset.dry_run", n=args.n, **cost)
        return

    rows = generate_golden_set(settings, args.n, seed=args.seed)
    with args.out.open("w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")
    n_with_ground_truth = sum(1 for r in rows if r["relevant_chunk_ids"])
    log.info(
        "make_testset.done",
        generated=len(rows),
        with_ground_truth=n_with_ground_truth,
        out=str(args.out),
    )


if __name__ == "__main__":
    _main()
