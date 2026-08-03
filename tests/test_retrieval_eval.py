"""Unit tests for the deterministic retrieval metrics (P5.3) -- no live services."""

from __future__ import annotations

import math

from eval.retrieval_eval import hit_at_k, ndcg_at_k, reciprocal_rank


def test_hit_at_k_true_when_relevant_in_top_k() -> None:
    assert hit_at_k(["a", "b", "c"], {"b"}, k=3) == 1.0


def test_hit_at_k_false_when_relevant_outside_top_k() -> None:
    assert hit_at_k(["a", "b", "c"], {"z"}, k=3) == 0.0


def test_hit_at_k_respects_k_cutoff() -> None:
    assert hit_at_k(["a", "b", "c"], {"c"}, k=2) == 0.0


def test_reciprocal_rank_first_position() -> None:
    assert reciprocal_rank(["a", "b", "c"], {"a"}) == 1.0


def test_reciprocal_rank_third_position() -> None:
    assert reciprocal_rank(["a", "b", "c"], {"c"}) == 1.0 / 3


def test_reciprocal_rank_zero_when_absent() -> None:
    assert reciprocal_rank(["a", "b", "c"], {"z"}) == 0.0


def test_reciprocal_rank_uses_earliest_relevant_hit() -> None:
    # two relevant ids present; rank should reflect the first (best) one, not the second.
    assert reciprocal_rank(["a", "b", "c"], {"b", "c"}) == 1.0 / 2


def test_ndcg_perfect_ranking_is_one() -> None:
    # the single relevant chunk is ranked first -- best possible arrangement.
    assert ndcg_at_k(["a", "b", "c"], {"a"}, k=3) == 1.0


def test_ndcg_zero_when_no_relevant_present() -> None:
    assert ndcg_at_k(["a", "b", "c"], {"z"}, k=3) == 0.0


def test_ndcg_penalizes_lower_rank() -> None:
    top = ndcg_at_k(["a", "b", "c"], {"a"}, k=3)
    lower = ndcg_at_k(["a", "b", "c"], {"c"}, k=3)
    assert lower < top


def test_ndcg_multiple_relevant_matches_hand_computed_value() -> None:
    # relevant = {a, c}; DCG = 1/log2(2) + 1/log2(4); IDCG (ideal: both relevant first)
    # = 1/log2(2) + 1/log2(3).
    retrieved = ["a", "b", "c"]
    relevant = {"a", "c"}
    dcg = 1.0 / math.log2(2) + 1.0 / math.log2(4)
    idcg = 1.0 / math.log2(2) + 1.0 / math.log2(3)
    assert ndcg_at_k(retrieved, relevant, k=3) == dcg / idcg


def test_ndcg_empty_retrieval_is_zero() -> None:
    assert ndcg_at_k([], {"a"}, k=3) == 0.0
