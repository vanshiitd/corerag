"""RAGAS answer-quality evaluation (P5.4): faithfulness, answer relevancy, context
precision/recall over real graph runs against the golden set.

Judge = core.llm.get_agent_model (gpt-4o-mini, already used for router/grader) via
eval.ragas_compat.ragas_llm -- no new provider. Embeddings = our own local dense
embedder via ragas_embeddings.

Run standalone:
    uv run python -m eval.ragas_eval                     # full golden set
    uv run python -m eval.ragas_eval --limit 5            # small real check

Generation (the graph's "generate" node) runs on Groq, whose free tier caps
llama-3.3-70b-versatile at 100,000 tokens/day *per organization* -- already
confirmed exhausted mid-eval once before (PLAN.md P5.6). A full 181-question
run can plausibly need more than a day's quota, so each completed question's
graph output is checkpointed to `--checkpoint` (append-only JSONL, flushed per
row) and skipped on a later run -- interrupting or hitting the quota loses no
already-paid-for work. Scoring (OpenAI-judged, real $ cost) only runs once
every sample is checkpointed, unless `--score-partial` is passed.
"""

from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path
from typing import Any, cast

import groq
import structlog

from core.agents.graph import build_graph
from core.clients import make_qdrant_client
from core.config import Settings, get_settings
from core.logging import configure_logging
from eval.ragas_compat import ragas_embeddings, ragas_llm
from eval.retrieval_eval import GoldenSample, load_golden_set

log = structlog.get_logger()

_DEFAULT_GOLDEN = Path("eval/golden_set.jsonl")
_DEFAULT_CHECKPOINT = Path("eval/ragas_checkpoint.jsonl")


def _load_checkpoint(path: Path) -> dict[str, dict[str, Any]]:
    """Question -> already-completed graph output, keyed for O(1) skip checks."""
    if not path.exists():
        return {}
    rows: dict[str, dict[str, Any]] = {}
    for line in path.read_text().splitlines():
        if line.strip():
            row = json.loads(line)
            rows[row["question"]] = row
    return rows


async def _run_graph(
    settings: Settings, samples: list[GoldenSample], checkpoint: Path
) -> list[Any]:
    """Run the real agent graph for each golden question not already checkpointed;
    return a SingleTurnSample per question (answer, contexts) ready for RAGAS
    scoring, drawn from the checkpoint file after this run updates it.

    Stops early -- rather than crashing or silently dropping the remainder -- if
    Groq's shared daily generation quota trips mid-run; whatever was completed
    (this run and any prior one) stays on disk for the next invocation to pick up."""
    from ragas import SingleTurnSample

    done = _load_checkpoint(checkpoint)
    remaining = [s for s in samples if s.question not in done]

    if remaining:
        client = make_qdrant_client(settings)
        graph = build_graph(client, settings)
        try:
            with checkpoint.open("a") as f:
                for sample in remaining:
                    try:
                        state = await graph.ainvoke(
                            {
                                "query": sample.question,
                                "original_query": sample.question,
                                "retries": 0,
                            },
                            config={"recursion_limit": 20},
                        )
                    except groq.RateLimitError as exc:
                        log.warning(
                            "ragas_eval.quota_exhausted",
                            completed=len(done),
                            total=len(samples),
                            detail=str(exc)[:200],
                        )
                        break
                    row = {
                        "question": sample.question,
                        "response": state.get("answer", ""),
                        "retrieved_contexts": [c.text for c in state.get("citations", [])],
                        "reference": sample.reference,
                    }
                    f.write(json.dumps(row) + "\n")
                    f.flush()
                    done[sample.question] = row
                    log.info(
                        "ragas_eval.graph_sample",
                        i=len(done),
                        n=len(samples),
                        question=sample.question[:60],
                    )
        finally:
            await client.close()

    result_rows = []
    for sample in samples:
        completed = done.get(sample.question)
        if completed is not None:
            result_rows.append(
                SingleTurnSample(
                    user_input=completed["question"],
                    response=completed["response"],
                    retrieved_contexts=completed["retrieved_contexts"],
                    reference=completed["reference"],
                )
            )
    return result_rows


def run_ragas_eval(
    settings: Settings,
    samples: list[GoldenSample],
    checkpoint: Path = _DEFAULT_CHECKPOINT,
    *,
    score_partial: bool = False,
) -> dict[str, float] | None:
    """Score real graph runs over the golden set with RAGAS's core RAG metrics.

    Returns None (does not score) if the checkpoint isn't yet complete for this
    sample set and `score_partial` wasn't requested -- scoring is OpenAI-judged,
    real-money work, not worth spending on a set that'll grow once the Groq quota
    that interrupted graph-running resets."""
    from ragas import EvaluationDataset, evaluate
    from ragas.dataset_schema import EvaluationResult  # not re-exported from ragas top-level
    from ragas.metrics import AnswerRelevancy, ContextPrecision, ContextRecall, Faithfulness

    rows = asyncio.run(_run_graph(settings, samples, checkpoint))
    if len(rows) < len(samples) and not score_partial:
        log.warning(
            "ragas_eval.incomplete_not_scored",
            completed=len(rows),
            total=len(samples),
            checkpoint=str(checkpoint),
        )
        return None
    if len(rows) < len(samples):
        log.warning("ragas_eval.scoring_partial_set", completed=len(rows), total=len(samples))

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
    parser.add_argument("--checkpoint", type=Path, default=_DEFAULT_CHECKPOINT)
    parser.add_argument(
        "--score-partial",
        action="store_true",
        help="score whatever's checkpointed even if the full set isn't done yet",
    )
    args = parser.parse_args()
    configure_logging(get_settings())
    settings = get_settings()

    samples = load_golden_set(args.golden)
    if args.limit:
        samples = samples[: args.limit]
    log.info("ragas_eval.loaded", n=len(samples), golden=str(args.golden))

    scores = run_ragas_eval(settings, samples, args.checkpoint, score_partial=args.score_partial)
    if scores is None:
        log.warning(
            "ragas_eval.not_scored",
            hint="re-run the same command once the Groq daily quota resets to continue",
        )
        return
    log.info("ragas_eval.result", n=len(samples), **scores)


if __name__ == "__main__":
    _main()
