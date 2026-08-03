"""Retrieval quality metrics (P5.3): hit@k, MRR, nDCG against the golden set's
reference chunk_ids. No LLM judge -- deterministic, reproducible from a run of
real retrieval against the live corpus.

Run standalone:
    uv run python -m eval.retrieval_eval                       # hybrid+rerank, k=30/n=5
    uv run python -m eval.retrieval_eval --mode dense --k 10
    uv run python -m eval.retrieval_eval --golden eval/golden_set.jsonl
"""

from __future__ import annotations

import argparse
import asyncio
import json
import math
from pathlib import Path
from typing import Literal

import structlog

from core.clients import make_qdrant_client
from core.config import Settings, get_settings
from core.logging import configure_logging
from core.reranker import rerank_async
from core.retrieval import ScoredChunk, dense_search, hybrid_search

log = structlog.get_logger()

_DEFAULT_GOLDEN = Path("eval/golden_set.jsonl")

RetrievalMode = Literal["hybrid", "dense"]


class GoldenSample:
    """One (question, relevant_chunk_ids) ground-truth pair."""

    def __init__(self, question: str, reference: str, relevant_chunk_ids: list[str]) -> None:
        self.question = question
        self.reference = reference
        self.relevant_chunk_ids = relevant_chunk_ids


def load_golden_set(path: Path = _DEFAULT_GOLDEN) -> list[GoldenSample]:
    """Load the golden set, skipping samples with no recoverable ground truth."""
    samples = []
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if row.get("relevant_chunk_ids"):
            samples.append(
                GoldenSample(row["question"], row.get("reference", ""), row["relevant_chunk_ids"])
            )
    return samples


def hit_at_k(retrieved_ids: list[str], relevant_ids: set[str], k: int) -> float:
    """1.0 if any relevant chunk appears in the top k, else 0.0."""
    return 1.0 if set(retrieved_ids[:k]) & relevant_ids else 0.0


def reciprocal_rank(retrieved_ids: list[str], relevant_ids: set[str]) -> float:
    """1/rank of the first relevant chunk in the ranking; 0.0 if none found."""
    for i, cid in enumerate(retrieved_ids, start=1):
        if cid in relevant_ids:
            return 1.0 / i
    return 0.0


def ndcg_at_k(retrieved_ids: list[str], relevant_ids: set[str], k: int) -> float:
    """Binary-relevance nDCG@k (relevant=1, irrelevant=0)."""
    dcg = sum(
        1.0 / math.log2(i + 1)
        for i, cid in enumerate(retrieved_ids[:k], start=1)
        if cid in relevant_ids
    )
    ideal_hits = min(len(relevant_ids), k)
    idcg = sum(1.0 / math.log2(i + 1) for i in range(1, ideal_hits + 1))
    return dcg / idcg if idcg > 0 else 0.0


async def _retrieve(
    settings: Settings, query: str, mode: RetrievalMode, k: int, rerank: bool
) -> list[ScoredChunk]:
    client = make_qdrant_client(settings)
    try:
        search = hybrid_search if mode == "hybrid" else dense_search
        candidates = await search(client, settings, query, limit=k)
        if rerank:
            candidates = await rerank_async(query, candidates, settings)
        return candidates
    finally:
        await client.close()


async def evaluate_retrieval(
    settings: Settings,
    samples: list[GoldenSample],
    *,
    mode: RetrievalMode = "hybrid",
    k: int = 30,
    rerank: bool = True,
) -> dict[str, float]:
    """Average hit@k / MRR / nDCG@k over the golden set for one retrieval config."""
    hits, mrrs, ndcgs = [], [], []
    for sample in samples:
        results = await _retrieve(settings, sample.question, mode, k, rerank)
        retrieved_ids = [c.chunk_id for c in results]
        relevant = set(sample.relevant_chunk_ids)
        hits.append(hit_at_k(retrieved_ids, relevant, k))
        mrrs.append(reciprocal_rank(retrieved_ids, relevant))
        ndcgs.append(ndcg_at_k(retrieved_ids, relevant, k))

    n = len(samples)
    return {
        "n": n,
        f"hit@{k}": round(sum(hits) / n, 4) if n else 0.0,
        "mrr": round(sum(mrrs) / n, 4) if n else 0.0,
        f"ndcg@{k}": round(sum(ndcgs) / n, 4) if n else 0.0,
    }


def _main() -> None:
    parser = argparse.ArgumentParser(description="CoreRAG retrieval quality evaluation")
    parser.add_argument("--golden", type=Path, default=_DEFAULT_GOLDEN)
    parser.add_argument("--mode", choices=["hybrid", "dense"], default="hybrid")
    parser.add_argument("--k", type=int, default=30)
    parser.add_argument("--no-rerank", action="store_true", help="skip cross-encoder reranking")
    args = parser.parse_args()
    configure_logging(get_settings())
    settings = get_settings()

    samples = load_golden_set(args.golden)
    log.info("retrieval_eval.loaded", n=len(samples), golden=str(args.golden))
    result = asyncio.run(
        evaluate_retrieval(settings, samples, mode=args.mode, k=args.k, rerank=not args.no_rerank)
    )
    log.info("retrieval_eval.result", mode=args.mode, k=args.k, rerank=not args.no_rerank, **result)


if __name__ == "__main__":
    _main()
