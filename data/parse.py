"""Parse arXiv PDFs into clean, sectioned text.

Default parser is PyMuPDF (fast, no service). GROBID (higher fidelity for scientific
PDFs) is used when ``settings.pdf_parser == "grobid"`` and reachable, falling back to
PyMuPDF on any error.
"""

from __future__ import annotations

import re
import xml.etree.ElementTree as ET
from pathlib import Path

import httpx
import pymupdf
import structlog

from core.config import Settings
from data.schemas import ParsedDoc, Section

log = structlog.get_logger()

_TEI = "{http://www.tei-c.org/ns/1.0}"
_HEADING_RE = re.compile(r"^\s*(\d{1,2}(?:\.\d{1,2})*)\.?\s+([A-Z][A-Za-z0-9 ,:&/-]{2,60})\s*$")
_REF_RE = re.compile(r"^\s*(references|bibliography)\s*$", re.IGNORECASE | re.MULTILINE)


def _clean(text: str) -> str:
    text = re.sub(r"-\n(\w)", r"\1", text)  # join hyphenated line breaks
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _cut_references(text: str) -> str:
    matches = list(_REF_RE.finditer(text))
    return text[: matches[-1].start()].strip() if matches else text


def _split_sections(text: str) -> list[Section]:
    sections: list[Section] = []
    heading: str | None = None
    buf: list[str] = []

    def flush() -> None:
        body = "\n".join(buf).strip()
        if body:
            sections.append(Section(heading=heading, text=body))

    for line in text.split("\n"):
        match = _HEADING_RE.match(line)
        if match:
            flush()
            heading = match.group(2).strip()
            buf = []
        else:
            buf.append(line)
    flush()
    return sections or [Section(heading=None, text=text)]


def parse_pymupdf(pdf_path: str, arxiv_id: str, title: str) -> ParsedDoc:
    """Fast, service-free parse. Imperfect on multi-column layouts."""
    doc = pymupdf.open(pdf_path)
    try:
        raw = "\n".join(page.get_text("text") for page in doc)
    finally:
        doc.close()
    text = _cut_references(_clean(raw))
    return ParsedDoc(arxiv_id=arxiv_id, title=title, sections=_split_sections(text))


def parse_grobid(pdf_path: str, arxiv_id: str, title: str, settings: Settings) -> ParsedDoc:
    """High-fidelity parse via a running GROBID service (TEI XML -> sections)."""
    with Path(pdf_path).open("rb") as fh:
        resp = httpx.post(
            f"{settings.grobid_url}/api/processFulltextDocument",
            files={"input": fh},
            data={"segmentSentences": "0"},
            timeout=180.0,
        )
    resp.raise_for_status()
    root = ET.fromstring(resp.content)
    body = root.find(f".//{_TEI}text/{_TEI}body")
    sections: list[Section] = []
    if body is not None:
        for div in body.findall(f"{_TEI}div"):
            head = div.find(f"{_TEI}head")
            heading = head.text.strip() if head is not None and head.text else None
            paras = ["".join(p.itertext()).strip() for p in div.findall(f"{_TEI}p")]
            joined = "\n".join(p for p in paras if p)
            if joined:
                sections.append(Section(heading=heading, text=joined))
    if not sections:
        raise ValueError("GROBID returned no usable sections")
    return ParsedDoc(arxiv_id=arxiv_id, title=title, sections=sections)


def parse_pdf(pdf_path: str, arxiv_id: str, title: str, settings: Settings) -> ParsedDoc:
    """Parse a PDF, honoring settings.pdf_parser with a safe PyMuPDF fallback."""
    if settings.pdf_parser == "grobid":
        try:
            return parse_grobid(pdf_path, arxiv_id, title, settings)
        except Exception as exc:
            log.warning("parse.grobid_fallback", arxiv_id=arxiv_id, error=str(exc))
    return parse_pymupdf(pdf_path, arxiv_id, title)
