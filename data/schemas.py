"""Pydantic models shared across the ingestion pipeline."""

from __future__ import annotations

from pydantic import BaseModel


class PaperMeta(BaseModel):
    """Metadata for one arXiv paper."""

    arxiv_id: str
    title: str
    authors: list[str]
    published: str  # ISO date, YYYY-MM-DD
    categories: list[str]
    abstract: str
    abs_url: str
    pdf_path: str | None = None


class Section(BaseModel):
    """A parsed document section."""

    heading: str | None
    text: str


class ParsedDoc(BaseModel):
    """A parsed paper: ordered sections of clean text."""

    arxiv_id: str
    title: str
    sections: list[Section]

    @property
    def full_text(self) -> str:
        return "\n\n".join(s.text for s in self.sections)


class Chunk(BaseModel):
    """A retrievable chunk carrying full citation payload."""

    chunk_id: str  # f"{arxiv_id}::{index}"
    arxiv_id: str
    title: str
    authors: list[str]
    abs_url: str
    section: str | None
    index: int
    text: str  # raw chunk content (shown in citations)
    context: str | None = None  # optional LLM-generated context prefix

    @property
    def embed_input(self) -> str:
        """Text fed to the embedder: context-prefixed when contextualized."""
        return f"{self.context}\n\n{self.text}" if self.context else self.text
