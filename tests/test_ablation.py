"""Unit tests for eval/ablation.py's pure helpers -- no live services."""

from __future__ import annotations

from core.config import Settings
from data.schemas import PaperMeta
from eval.ablation import _filter_golden_set, _variant_settings
from eval.retrieval_eval import GoldenSample


def _paper(arxiv_id: str) -> PaperMeta:
    return PaperMeta(
        arxiv_id=arxiv_id,
        title="t",
        authors=["a"],
        published="2024-01-01",
        categories=["cs.DC"],
        abstract="abs",
        abs_url=f"https://arxiv.org/abs/{arxiv_id}",
    )


def test_variant_settings_sets_strategy_and_collection_version() -> None:
    base = Settings()
    variant = _variant_settings(base, "doc_summary")
    assert variant.contextualize_strategy == "doc_summary"
    assert variant.qdrant_collection_version == "ablation_doc_summary"
    # base is untouched -- model_copy, not mutation.
    assert base.qdrant_collection_version != variant.qdrant_collection_version


def test_filter_golden_set_keeps_samples_within_the_paper_sample() -> None:
    papers = [_paper("1.1"), _paper("2.2")]
    samples = [
        GoldenSample("q1", "r1", ["1.1::0"]),
        GoldenSample("q2", "r2", ["3.3::0"]),  # not in the sample -- excluded
        GoldenSample("q3", "r3", ["2.2::1"]),
    ]
    kept = _filter_golden_set(samples, papers)
    assert [s.question for s in kept] == ["q1", "q3"]


def test_filter_golden_set_excludes_multi_paper_samples_missing_any_paper() -> None:
    papers = [_paper("1.1")]
    samples = [GoldenSample("q", "r", ["1.1::0", "9.9::0"])]
    assert _filter_golden_set(samples, papers) == []


def test_filter_golden_set_empty_input_is_empty() -> None:
    assert _filter_golden_set([], [_paper("1.1")]) == []
