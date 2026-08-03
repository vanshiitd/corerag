"""Latency benchmark: retrieval + reranking (P2.4), and the full agent graph (P3.8).

Run standalone:
    uv run python -m eval.latency_bench
    uv run python -m eval.latency_bench --k 30 50 100 --modes pytorch-cpu cpu-onnx
    uv run python -m eval.latency_bench --graph
"""

from __future__ import annotations

import argparse
import asyncio
import statistics
import time

import structlog

from core.agents.graph import build_graph
from core.clients import make_qdrant_client
from core.config import Settings, get_settings
from core.logging import configure_logging
from core.reranker import rerank
from core.retrieval import hybrid_search

log = structlog.get_logger()

# Two real queries per scenario -- not a full statistical sample (each full-graph
# invocation costs real LLM calls and 10-25s), but enough to see the real shape of
# the compounding-latency risk flagged in the P3 replan, honestly, not estimated.
GRAPH_BENCHMARK_QUERIES = {
    "simple": [
        "what is speculative decoding?",
        "how does continuous batching work?",
    ],
    "multi_hop": [
        "Compare KV cache compression, speculative decoding, and continuous "
        "batching -- what are the tradeoffs between all three?",
        "What are the differences between quantization and pruning for reducing "
        "model size, and how do they affect inference latency?",
    ],
    "off_topic": [
        "What is the best recipe for chocolate chip cookies?",
        "How do I train for a marathon?",
    ],
}

# Real, diverse AI-systems questions matching the corpus domain -- not synthetic
# filler, so timings reflect genuine query/candidate text, not best-case toy input.
BENCHMARK_QUERIES = [
    "techniques for reducing latency in LLM inference serving",
    "GPU memory management for LLM serving",
    "quantization methods for neural network inference",
    "speculative decoding for language models",
    "distributed training of large language models",
    "KV cache compression techniques",
    "hardware accelerators for deep learning inference",
    "continuous batching in LLM serving systems",
]


def _percentile(values: list[float], pct: float) -> float:
    """Nearest-rank percentile (no interpolation) -- fine for small benchmark samples."""
    ordered = sorted(values)
    idx = min(len(ordered) - 1, round(pct / 100 * (len(ordered) - 1)))
    return ordered[idx]


async def benchmark_rerank(
    settings: Settings, k_values: list[int], queries: list[str] | None = None
) -> dict[int, dict[str, float]]:
    """Time reranking at each K, across real queries against the live corpus.

    Each query's candidates are fetched once at max(k_values) and sliced down for
    smaller K's, so every K sees the same candidate pool -- an apples-to-apples
    comparison with retrieval cost excluded from the reranker timing.

    Returns {k: {"p50_ms", "p95_ms", "max_ms", "mean_ms"}}.
    """
    queries = queries or BENCHMARK_QUERIES
    client = make_qdrant_client(settings)
    max_k = max(k_values)
    timings: dict[int, list[float]] = {k: [] for k in k_values}

    try:
        # Warm-up: load (and JIT/cache) the model once, outside the timed loop.
        warmup = await hybrid_search(client, settings, queries[0], limit=5)
        rerank(queries[0], warmup, settings)

        for query in queries:
            candidates = await hybrid_search(client, settings, query, limit=max_k)
            for k in k_values:
                t0 = time.perf_counter()
                rerank(query, candidates[:k], settings)
                timings[k].append((time.perf_counter() - t0) * 1000)
    finally:
        await client.close()

    return {
        k: {
            "p50_ms": round(statistics.median(vals), 1),
            "p95_ms": round(_percentile(vals, 95), 1),
            "max_ms": round(max(vals), 1),
            "mean_ms": round(statistics.fmean(vals), 1),
        }
        for k, vals in timings.items()
    }


async def benchmark_mps_bonus(
    settings: Settings, k_values: list[int], queries: list[str] | None = None
) -> dict[int, dict[str, float]]:
    """Bonus, local-dev-only data point: PyTorch on Apple's MPS (Metal) backend.

    Never a real deployment option -- the P6 cloud host has no GPU/MPS -- so this
    is measured separately from the settings-driven ``reranker_mode`` machinery,
    purely to see whether local Apple Silicon acceleration is worth knowing about.
    """
    from sentence_transformers import CrossEncoder

    queries = queries or BENCHMARK_QUERIES
    client = make_qdrant_client(settings)
    max_k = max(k_values)
    timings: dict[int, list[float]] = {k: [] for k in k_values}

    try:
        model = CrossEncoder(
            settings.reranker_model, max_length=settings.reranker_max_length, device="mps"
        )
        warmup = await hybrid_search(client, settings, queries[0], limit=5)
        model.predict([(queries[0], c.text) for c in warmup])

        for query in queries:
            candidates = await hybrid_search(client, settings, query, limit=max_k)
            for k in k_values:
                pairs = [(query, c.text) for c in candidates[:k]]
                t0 = time.perf_counter()
                model.predict(pairs)
                timings[k].append((time.perf_counter() - t0) * 1000)
    finally:
        await client.close()

    return {
        k: {
            "p50_ms": round(statistics.median(vals), 1),
            "p95_ms": round(_percentile(vals, 95), 1),
            "max_ms": round(max(vals), 1),
            "mean_ms": round(statistics.fmean(vals), 1),
        }
        for k, vals in timings.items()
    }


async def benchmark_graph(
    settings: Settings, queries_by_scenario: dict[str, list[str]] | None = None
) -> dict[str, list[dict[str, object]]]:
    """Time the full agent graph across scenario categories (P3.8).

    Not a p50/p95 statistical benchmark like benchmark_rerank -- each sample is a
    real LLM-backed graph run costing real money and 10-25s, so this reports every
    individual sample's numbers plainly rather than pretending a percentile from
    n=2 means anything. That's the honest number the P3 replan asked for.
    """
    queries_by_scenario = queries_by_scenario or GRAPH_BENCHMARK_QUERIES
    client = make_qdrant_client(settings)
    graph = build_graph(client, settings)
    results: dict[str, list[dict[str, object]]] = {}

    try:
        for scenario, queries in queries_by_scenario.items():
            results[scenario] = []
            for query in queries:
                t0 = time.perf_counter()
                state = await graph.ainvoke(
                    {"query": query, "original_query": query, "retries": 0},
                    config={"recursion_limit": 20},
                )
                elapsed_ms = round((time.perf_counter() - t0) * 1000, 1)
                sample = {
                    "query": query[:60],
                    "elapsed_ms": elapsed_ms,
                    "route": state.get("route"),
                    "retries": state.get("retries", 0),
                    "low_confidence": state.get("low_confidence", False),
                }
                results[scenario].append(sample)
                log.info("latency_bench.graph_sample", scenario=scenario, **sample)
    finally:
        await client.close()

    return results


async def run(
    k_values: list[int], modes: list[str], include_mps_bonus: bool = False
) -> dict[str, dict[int, dict[str, float]]]:
    """Run the benchmark across every (mode, K) combination; log and return results."""
    base = get_settings()
    all_results: dict[str, dict[int, dict[str, float]]] = {}
    for mode in modes:
        settings = base.model_copy(update={"reranker_mode": mode})
        log.info("latency_bench.running", mode=mode, k_values=k_values)
        results = await benchmark_rerank(settings, k_values)
        all_results[mode] = results
        for k, stats in results.items():
            log.info("latency_bench.result", mode=mode, k=k, **stats)

    if include_mps_bonus:
        log.info("latency_bench.running", mode="mps-bonus", k_values=k_values)
        mps_results = await benchmark_mps_bonus(base, k_values)
        all_results["mps-bonus"] = mps_results
        for k, stats in mps_results.items():
            log.info("latency_bench.result", mode="mps-bonus", k=k, **stats)

    return all_results


def _main() -> None:
    parser = argparse.ArgumentParser(description="CoreRAG latency benchmark")
    parser.add_argument("--k", type=int, nargs="+", default=[30, 50, 100])
    parser.add_argument("--modes", nargs="+", default=["pytorch-cpu", "cpu-onnx"])
    parser.add_argument("--mps-bonus", action="store_true", help="also benchmark PyTorch on MPS")
    parser.add_argument(
        "--graph", action="store_true", help="benchmark the full agent graph instead of rerank"
    )
    args = parser.parse_args()
    configure_logging(get_settings())
    if args.graph:
        asyncio.run(benchmark_graph(get_settings()))
    else:
        asyncio.run(run(args.k, args.modes, include_mps_bonus=args.mps_bonus))


if __name__ == "__main__":
    _main()
