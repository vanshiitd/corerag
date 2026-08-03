"""End-to-end ingestion: fetch -> parse -> chunk -> [contextualize] -> embed -> index.

Examples:
    uv run python -m data.pipeline --no-fetch --limit 3 --dry-run   # estimate cost, no spend
    uv run python -m data.pipeline --limit 5 --recreate
    uv run python -m data.pipeline --no-fetch --recreate            # reuse metadata.jsonl
"""

from __future__ import annotations

import argparse
import asyncio
import time
from pathlib import Path

import structlog
from qdrant_client import QdrantClient

from core.config import Settings, get_settings
from core.logging import configure_logging
from data.chunk import chunk_document
from data.contextualize import contextualize_chunks, estimate_cost
from data.embeddings import embed_dense, embed_sparse
from data.fetch_arxiv import fetch_papers
from data.index import ensure_collection, is_paper_indexed, make_client, upsert_chunks
from data.parse import parse_pdf
from data.schemas import Chunk, PaperMeta, ParsedDoc

log = structlog.get_logger()

_CostTotals = dict[str, float]


def load_metadata(path: Path) -> list[PaperMeta]:
    return [
        PaperMeta.model_validate_json(line)
        for line in path.read_text().splitlines()
        if line.strip()
    ]


def parse_and_chunk(meta: PaperMeta, settings: Settings) -> tuple[ParsedDoc, list[Chunk]] | None:
    """Parse + chunk one paper (contextualize_strategy-independent -- only the
    ``context`` prefix varies by strategy, not chunk boundaries/text), so callers
    that need multiple contextualize_strategy variants of the same paper (P5.5's
    ablation) can parse once and reuse the result, rather than re-running GROBID
    once per variant."""
    if not meta.pdf_path or not Path(meta.pdf_path).exists():
        log.warning("pipeline.skip_no_pdf", arxiv_id=meta.arxiv_id)
        return None
    doc = parse_pdf(meta.pdf_path, meta.arxiv_id, meta.title, settings)
    chunks = chunk_document(doc, meta, settings)
    if not chunks:
        log.warning("pipeline.no_chunks", arxiv_id=meta.arxiv_id)
        return None
    return doc, chunks


def _dry_run(papers: list[PaperMeta], settings: Settings, start: float) -> None:
    totals: _CostTotals = {"calls": 0, "uncached_input": 0, "cached_input": 0, "usd": 0.0}
    for meta in papers:
        parsed = parse_and_chunk(meta, settings)
        if parsed is None:
            continue
        doc, chunks = parsed
        cost = estimate_cost(doc, chunks, settings)
        for key in totals:
            totals[key] += cost[key]
        log.info("pipeline.estimate", arxiv_id=meta.arxiv_id, chunks=len(chunks), **cost)

    log.info(
        "pipeline.dry_run_done",
        papers=len(papers),
        strategy=settings.contextualize_strategy,
        estimated_usd=round(totals["usd"], 4),
        llm_calls=int(totals["calls"]),
        seconds=round(time.perf_counter() - start, 1),
    )


def _ingest(
    papers: list[PaperMeta], settings: Settings, client: QdrantClient, start: float
) -> None:
    total_chunks = 0
    skipped = 0
    for meta in papers:
        if is_paper_indexed(client, settings, meta.arxiv_id):
            skipped += 1
            log.info("pipeline.skip_already_indexed", arxiv_id=meta.arxiv_id)
            continue
        parsed = parse_and_chunk(meta, settings)
        if parsed is None:
            continue
        doc, chunks = parsed

        if settings.contextualize_strategy != "none":
            chunks = asyncio.run(contextualize_chunks(doc, chunks, settings))

        inputs = [c.embed_input for c in chunks]
        dense = embed_dense(inputs, settings)
        sparse = embed_sparse(inputs, settings)
        count = upsert_chunks(client, settings, chunks, dense, sparse)
        total_chunks += count
        log.info("pipeline.indexed", arxiv_id=meta.arxiv_id, chunks=count)

    log.info(
        "pipeline.done",
        papers=len(papers),
        skipped_already_indexed=skipped,
        chunks=total_chunks,
        strategy=settings.contextualize_strategy,
        collection=settings.qdrant_collection_name,
        seconds=round(time.perf_counter() - start, 1),
    )


def run(
    *, limit: int | None = None, recreate: bool = False, fetch: bool = True, dry_run: bool = False
) -> None:
    settings = get_settings()
    start = time.perf_counter()

    if fetch:
        papers = fetch_papers(settings, limit=limit)
    else:
        papers = load_metadata(Path(settings.arxiv_metadata_path))
        if limit:
            papers = papers[:limit]

    if dry_run:
        _dry_run(papers, settings, start)
        return

    client = make_client(settings)
    ensure_collection(client, settings, recreate=recreate)
    _ingest(papers, settings, client, start)


def _main() -> None:
    parser = argparse.ArgumentParser(description="CoreRAG ingestion pipeline")
    parser.add_argument("--limit", type=int, default=None, help="max papers")
    parser.add_argument("--recreate", action="store_true", help="drop + recreate the collection")
    parser.add_argument("--no-fetch", action="store_true", help="reuse existing metadata.jsonl")
    parser.add_argument(
        "--dry-run", action="store_true", help="estimate contextualization cost; no LLM/DB calls"
    )
    args = parser.parse_args()
    configure_logging(get_settings())
    run(limit=args.limit, recreate=args.recreate, fetch=not args.no_fetch, dry_run=args.dry_run)


if __name__ == "__main__":
    _main()
