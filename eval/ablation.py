"""Ablation study (P5.5): contextualize_strategy (none/doc_summary/per_chunk) x
retrieval mode (hybrid/dense-only) x reranker (on/off) -- a real 3x2x2=12-cell
table, built entirely from code that already exists (data/contextualize.py's three
strategies, core/retrieval.py's hybrid_search/dense_search, core/reranker.py).

Scope: a reproducible paper *sample* (not the full 150), so building the "none"
and "doc_summary" variants (GROBID reparse + reindex) stays fast. All three
variants -- including per_chunk -- are built fresh over the *same* sample rather
than reusing the real `v1` production collection for per_chunk: v1 indexes all 150
papers, so retrieval against it would face more distractor candidates than the
30-paper "none"/"doc_summary" collections, biasing the comparison. Building all
three on identical corpora keeps it apples-to-apples.

chunk_id is contextualize_strategy-independent (only the LLM-generated `context`
prefix differs -- see data/schemas.py's Chunk.embed_input), so the *same* golden
set's chunk_id ground truth is valid across all three variants; this script filters
it down to samples whose paper is in the ablation sample.

Run standalone:
    uv run python -m eval.ablation --n-papers 5 --dry-run   # cost estimate, no spend
    uv run python -m eval.ablation --n-papers 40             # real ablation run
"""

from __future__ import annotations

import argparse
import asyncio
import itertools
import random
from pathlib import Path

import structlog

from core.config import Settings, get_settings
from core.logging import configure_logging
from data.contextualize import contextualize_chunks, estimate_cost
from data.embeddings import embed_dense, embed_sparse
from data.index import ensure_collection, is_paper_indexed, make_client, upsert_chunks
from data.pipeline import load_metadata, parse_and_chunk
from data.schemas import PaperMeta
from eval.retrieval_eval import GoldenSample, RetrievalMode, evaluate_retrieval, load_golden_set

log = structlog.get_logger()

STRATEGIES = ["none", "doc_summary", "per_chunk"]
_DEFAULT_GOLDEN = Path("eval/golden_set.jsonl")
# Distinct from make_testset.py's golden-set seed (42) -- these sample different
# populations (papers vs. chunks) for different purposes; no need to coordinate.
_ABLATION_SEED = 7
_COLLECTION_PREFIX = "ablation"


def sample_papers(settings: Settings, n: int, seed: int = _ABLATION_SEED) -> list[PaperMeta]:
    """Reproducibly sample n papers from the already-fetched local metadata."""
    papers = load_metadata(Path(settings.arxiv_metadata_path))
    if n >= len(papers):
        return papers
    return random.Random(seed).sample(papers, n)


def _variant_settings(base: Settings, strategy: str) -> Settings:
    return base.model_copy(
        update={
            "contextualize_strategy": strategy,
            "qdrant_collection_version": f"{_COLLECTION_PREFIX}_{strategy}",
        }
    )


def estimate_ablation_cost(base: Settings, papers: list[PaperMeta]) -> dict[str, float]:
    """Sum data.contextualize.estimate_cost across papers, for the two strategies
    that call an LLM ("none" costs $0 -- no calls at all)."""
    totals = {"calls": 0.0, "uncached_input": 0.0, "cached_input": 0.0, "usd": 0.0}
    for strategy in ("doc_summary", "per_chunk"):
        settings = _variant_settings(base, strategy)
        for meta in papers:
            parsed = parse_and_chunk(meta, settings)
            if parsed is None:
                continue
            doc, chunks = parsed
            cost = estimate_cost(doc, chunks, settings)
            for key in totals:
                totals[key] += cost[key]
    return {k: round(v, 4) if k == "usd" else v for k, v in totals.items()}


def build_ablation_collections(base: Settings, papers: list[PaperMeta]) -> None:
    """Parse each paper once, then contextualize+embed+index it into all three
    strategy-specific collections -- avoids re-running GROBID per strategy."""
    variants = {s: _variant_settings(base, s) for s in STRATEGIES}
    for settings in variants.values():
        client = make_client(settings)
        try:
            ensure_collection(client, settings, recreate=False)
        finally:
            client.close()

    for meta in papers:
        parsed = parse_and_chunk(meta, base)
        if parsed is None:
            continue
        doc, chunks = parsed

        for strategy, settings in variants.items():
            client = make_client(settings)
            try:
                if is_paper_indexed(client, settings, meta.arxiv_id):
                    continue
                strategy_chunks = chunks
                if strategy != "none":
                    strategy_chunks = asyncio.run(contextualize_chunks(doc, chunks, settings))
                inputs = [c.embed_input for c in strategy_chunks]
                dense = embed_dense(inputs, settings)
                sparse = embed_sparse(inputs, settings)
                upsert_chunks(client, settings, strategy_chunks, dense, sparse)
                log.info(
                    "ablation.indexed",
                    strategy=strategy,
                    arxiv_id=meta.arxiv_id,
                    chunks=len(strategy_chunks),
                )
            finally:
                client.close()


def _filter_golden_set(samples: list[GoldenSample], papers: list[PaperMeta]) -> list[GoldenSample]:
    """Only golden samples whose ground-truth chunk belongs to a sampled paper --
    otherwise every ablation collection (which only indexes the sample) would
    spuriously miss on questions about papers outside it."""
    sampled_ids = {p.arxiv_id for p in papers}
    return [
        s for s in samples if all(cid.split("::")[0] in sampled_ids for cid in s.relevant_chunk_ids)
    ]


async def run_ablation(
    base: Settings, papers: list[PaperMeta], golden: list[GoldenSample], k: int = 30
) -> list[dict[str, object]]:
    """Evaluate every (strategy, mode, rerank) cell; return rows for a results table."""
    rows: list[dict[str, object]] = []
    modes: list[RetrievalMode] = ["hybrid", "dense"]
    for strategy, mode, rerank in itertools.product(STRATEGIES, modes, (True, False)):
        settings = _variant_settings(base, strategy)
        metrics = await evaluate_retrieval(settings, golden, mode=mode, k=k, rerank=rerank)
        row = {"contextualize_strategy": strategy, "mode": mode, "rerank": rerank, **metrics}
        rows.append(row)
        log.info("ablation.cell", **row)
    return rows


def _main() -> None:
    parser = argparse.ArgumentParser(description="CoreRAG contextualization/retrieval ablation")
    parser.add_argument("--n-papers", type=int, default=40)
    parser.add_argument("--seed", type=int, default=_ABLATION_SEED)
    parser.add_argument("--golden", type=Path, default=_DEFAULT_GOLDEN)
    parser.add_argument("--k", type=int, default=30)
    parser.add_argument("--dry-run", action="store_true", help="estimate cost; no LLM/DB calls")
    parser.add_argument(
        "--skip-build", action="store_true", help="reuse existing ablation collections"
    )
    args = parser.parse_args()
    configure_logging(get_settings())
    base = get_settings()

    papers = sample_papers(base, args.n_papers, seed=args.seed)
    log.info("ablation.papers_sampled", n=len(papers))

    if args.dry_run:
        cost = estimate_ablation_cost(base, papers)
        log.info("ablation.dry_run", n_papers=len(papers), **cost)
        return

    if not args.skip_build:
        build_ablation_collections(base, papers)

    golden = load_golden_set(args.golden)
    filtered = _filter_golden_set(golden, papers)
    log.info("ablation.golden_filtered", total=len(golden), in_sample=len(filtered))
    if not filtered:
        log.warning("ablation.no_golden_overlap", hint="increase --n-papers or --seed")
        return

    asyncio.run(run_ablation(base, papers, filtered, k=args.k))


if __name__ == "__main__":
    _main()
