"""Unit tests for ingestion logic that needs no network/services."""

from __future__ import annotations

from core.config import Settings
from data.fetch_arxiv import build_query
from data.parse import _clean, _cut_references, _split_sections


def test_build_query_includes_categories_and_keywords() -> None:
    q = build_query(Settings())  # conftest seeds dummy secrets
    assert "cat:cs.DC" in q
    assert "cat:cs.AR" in q
    assert "cat:cs.LG AND" in q
    assert 'abs:"inference serving"' in q
    assert 'abs:"inference"' not in q  # bare, overloaded term stays excluded


def test_cut_references_truncates_bibliography() -> None:
    text = "Body text here.\nReferences\n[1] Foo et al., 2024."
    out = _cut_references(text)
    assert "Body text" in out
    assert "Foo et al" not in out


def test_split_sections_detects_numbered_headings() -> None:
    text = "intro para\n2 Related Work\nrelated para\n3 Method\nmethod para"
    headings = [s.heading for s in _split_sections(text)]
    assert "Related Work" in headings
    assert "Method" in headings


def test_clean_dehyphenates_linebreaks() -> None:
    assert "quantization" in _clean("quanti-\nzation")
