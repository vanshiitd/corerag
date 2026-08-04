"""RAGAS answer-quality evaluation (P5.4): faithfulness, answer relevancy, context
precision/recall over real graph runs against the golden set.

Judge = core.llm.get_agent_model (gpt-4o-mini, already used for router/grader) via
eval.ragas_compat.ragas_llm -- no new provider. Embeddings = our own local dense
embedder via ragas_embeddings.

Run standalone:
    uv run python -m eval.ragas_eval                     # full golden set
    uv run python -m eval.ragas_eval --limit 5            # small real check
"""

from __future__ import annotations

import argparse
import asyncio
from pathlib import Path
from typing import Any, cast

import structlog

from core.agents.graph import build_graph
from core.clients import make_qdrant_client
from core.config import Settings, get_settings
from core.logging import configure_logging
from eval.ragas_compat import ragas_embeddings, ragas_llm
from eval.retrieval_eval import GoldenSample, load_golden_set

log = structlog.get_logger()

_DEFAULT_GOLDEN = Path("eval/golden_set.jsonl")


async def _run_graph(settings: Settings, samples: list[GoldenSample]) -> list[Any]:
    """Run the real agent graph for each golden question; collect a SingleTurnSample
    per question (answer, contexts) ready for RAGAS scoring."""
    from ragas import SingleTurnSample

    client = make_qdrant_client(settings)
    graph = build_graph(client, settings)
    rows = []
    try:
        for i, sample in enumerate(samples):
            state = await graph.ainvoke(
                {"query": sample.question, "original_query": sample.question, "retries": 0},
                config={"recursion_limit": 20},
            )
            citations = state.get("citations", [])
            rows.append(
                SingleTurnSample(
                    user_input=sample.question,
                    response=state.get("answer", ""),
                    retrieved_contexts=[c.text for c in citations],
                    reference=sample.reference,
                )
            )
            log.info("ragas_eval.graph_sample", i=i, n=len(samples), question=sample.question[:60])
    finally:
        await client.close()
    return rows


def run_ragas_eval(settings: Settings, samples: list[GoldenSample]) -> dict[str, float]:
    """Score real graph runs over the golden set with RAGAS's core RAG metrics."""
    from ragas import EvaluationDataset, evaluate
    from ragas.dataset_schema import EvaluationResult  # not re-exported from ragas top-level
    from ragas.metrics import AnswerRelevancy, ContextPrecision, ContextRecall, Faithfulness

    rows = asyncio.run(_run_graph(settings, samples))
    dataset = EvaluationDataset(samples=rows)
    # features() is real (verified in ragas' own source) but ships no type
    # annotations. Called via getattr, not `dataset.features()` directly: mypy's
    # disallow-untyped-calls only fires when it can see the call as going through
    # a *known* untyped function -- and whether it can see that differs depending
    # on whether the `eval` dependency group is synced (confirmed by running mypy
    # in both states: a `# type: ignore[no-untyped-call]` or bare cast() is valid
    # in one state and flagged as unused in the other). getattr()'s return is
    # untyped Any in both states alike, sidestepping the asymmetry entirely.
    input_columns = cast("list[str]", getattr(dataset, "features")())  # noqa: B009

    # return_executor defaults to False, so this is always an EvaluationResult at
    # runtime; the declared union return type is just an unresolved overload.
    result = evaluate(
        dataset,
        metrics=[Faithfulness(), AnswerRelevancy(), ContextPrecision(), ContextRecall()],
        llm=ragas_llm(settings),
        embeddings=ragas_embeddings(settings),
    )
    assert isinstance(result, EvaluationResult)
    # Public API (to_pandas), not the private `_repr_dict` the library's own
    # __repr__ happens to use internally -- per-metric column mean, NaN-safe.
    df = result.to_pandas()
    metric_columns = [c for c in df.columns if c not in input_columns]
    return {col: round(float(df[col].mean()), 4) for col in metric_columns}


def _main() -> None:
    parser = argparse.ArgumentParser(description="CoreRAG RAGAS answer-quality evaluation")
    parser.add_argument("--golden", type=Path, default=_DEFAULT_GOLDEN)
    parser.add_argument("--limit", type=int, default=None, help="evaluate only the first N samples")
    args = parser.parse_args()
    configure_logging(get_settings())
    settings = get_settings()

    samples = load_golden_set(args.golden)
    if args.limit:
        samples = samples[: args.limit]
    log.info("ragas_eval.loaded", n=len(samples), golden=str(args.golden))

    scores = run_ragas_eval(settings, samples)
    log.info("ragas_eval.result", n=len(samples), **scores)


if __name__ == "__main__":
    _main()
