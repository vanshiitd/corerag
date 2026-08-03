"""Fetch AI-Systems papers from arXiv: metadata + PDFs.

Run standalone:  uv run python -m data.fetch_arxiv --limit 5
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import arxiv
import httpx
import structlog

from core.config import Settings, get_settings
from core.logging import configure_logging
from data.schemas import PaperMeta

log = structlog.get_logger()

_UA = "CoreRAG/0.1 (research use; +https://github.com)"


def build_query(settings: Settings) -> str:
    """arXiv query: cs.DC/cs.AR broadly + cs.LG narrowed to systems terms."""
    non_lg = [c for c in settings.arxiv_categories if c != "cs.LG"]
    parts: list[str] = []
    if non_lg:
        parts.append("(" + " OR ".join(f"cat:{c}" for c in non_lg) + ")")
    if "cs.LG" in settings.arxiv_categories:
        kw = " OR ".join(f'abs:"{k}"' for k in settings.arxiv_cslg_keywords)
        parts.append(f"(cat:cs.LG AND ({kw}))")
    return " OR ".join(parts)


def fetch_papers(settings: Settings, limit: int | None = None) -> list[PaperMeta]:
    """Search arXiv, download PDFs, and persist metadata. Idempotent on PDFs."""
    target = limit or settings.arxiv_max_papers
    pdf_dir = Path(settings.arxiv_pdf_dir)
    pdf_dir.mkdir(parents=True, exist_ok=True)
    meta_path = Path(settings.arxiv_metadata_path)
    meta_path.parent.mkdir(parents=True, exist_ok=True)

    query = build_query(settings)
    log.info("arxiv.search", query=query, target=target)

    client = arxiv.Client(page_size=100, delay_seconds=3.0, num_retries=3)
    search = arxiv.Search(
        query=query,
        max_results=target * 3,  # over-fetch; we filter by date and dedup below
        sort_by=arxiv.SortCriterion.SubmittedDate,
    )

    seen: set[str] = set()
    papers: list[PaperMeta] = []
    with httpx.Client(follow_redirects=True, timeout=60.0, headers={"User-Agent": _UA}) as http:
        for result in client.results(search):
            arxiv_id = result.get_short_id().split("v")[0]  # strip version suffix
            if arxiv_id in seen:
                continue
            published = result.published.date().isoformat()
            if published < settings.arxiv_date_floor:
                continue
            seen.add(arxiv_id)

            pdf_path = pdf_dir / f"{arxiv_id}.pdf"
            if not pdf_path.exists():
                try:
                    resp = http.get(result.pdf_url)
                    resp.raise_for_status()
                    pdf_path.write_bytes(resp.content)
                    time.sleep(1.0)  # courtesy delay between downloads
                except Exception as exc:
                    log.warning("arxiv.download_failed", arxiv_id=arxiv_id, error=str(exc))
                    continue

            papers.append(
                PaperMeta(
                    arxiv_id=arxiv_id,
                    title=result.title.strip(),
                    authors=[a.name for a in result.authors],
                    published=published,
                    categories=list(result.categories),
                    abstract=result.summary.strip(),
                    abs_url=result.entry_id,
                    pdf_path=str(pdf_path),
                )
            )
            log.info("arxiv.fetched", arxiv_id=arxiv_id, title=result.title[:70])
            if len(papers) >= target:
                break

    with meta_path.open("w") as f:
        for paper in papers:
            f.write(paper.model_dump_json() + "\n")
    log.info("arxiv.done", count=len(papers), metadata=str(meta_path))
    return papers


def _main() -> None:
    parser = argparse.ArgumentParser(description="Fetch arXiv papers")
    parser.add_argument("--limit", type=int, default=None, help="max papers to fetch")
    args = parser.parse_args()
    settings = get_settings()
    configure_logging(settings)
    fetch_papers(settings, limit=args.limit)


if __name__ == "__main__":
    _main()
