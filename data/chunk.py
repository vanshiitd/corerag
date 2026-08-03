"""Token-based, section-aware chunking using the dense model's tokenizer.

Chunks are sliced by *character offsets* derived from token offsets, so stored text
is the exact original (no tokenizer round-trip / lowercasing).
"""

from __future__ import annotations

from functools import lru_cache

from tokenizers import Tokenizer

from core.config import Settings
from data.schemas import Chunk, PaperMeta, ParsedDoc


@lru_cache(maxsize=2)
def _tokenizer(model_name: str) -> Tokenizer:
    return Tokenizer.from_pretrained(model_name)


def _chunk_text(tok: Tokenizer, text: str, size: int, overlap: int) -> list[str]:
    offsets = tok.encode(text, add_special_tokens=False).offsets
    if not offsets:
        return []
    step = max(1, size - overlap)
    pieces: list[str] = []
    for start in range(0, len(offsets), step):
        window = offsets[start : start + size]
        if not window:
            break
        piece = text[window[0][0] : window[-1][1]].strip()
        if piece:
            pieces.append(piece)
        if start + size >= len(offsets):
            break
    return pieces


def chunk_document(doc: ParsedDoc, meta: PaperMeta, settings: Settings) -> list[Chunk]:
    """Split a parsed doc into overlapping, section-tagged chunks with citation payload."""
    tok = _tokenizer(settings.dense_embed_model)
    chunks: list[Chunk] = []
    index = 0
    for section in doc.sections:
        for piece in _chunk_text(
            tok, section.text, settings.chunk_size_tokens, settings.chunk_overlap_tokens
        ):
            chunks.append(
                Chunk(
                    chunk_id=f"{doc.arxiv_id}::{index}",
                    arxiv_id=doc.arxiv_id,
                    title=meta.title,
                    authors=meta.authors,
                    abs_url=meta.abs_url,
                    section=section.heading,
                    index=index,
                    text=piece,
                )
            )
            index += 1
    return chunks
