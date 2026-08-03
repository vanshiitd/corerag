"""Stage-2 retrieval: cross-encoder reranking.

Cuts K stage-1 candidates down to the top N by scoring full (query, document) pairs
with joint attention -- higher precision than the bi-encoder's independent
embeddings, at higher per-pair compute cost. CPU-bound, so dispatched via
``asyncio.to_thread`` to avoid blocking the event loop.
"""

from __future__ import annotations

import asyncio
from functools import lru_cache

from sentence_transformers import CrossEncoder

from core.config import Settings
from core.retrieval import ScoredChunk


@lru_cache(maxsize=2)
def _model(model_name: str, max_length: int, mode: str, onnx_file: str) -> CrossEncoder:
    if mode == "cpu-onnx":
        # provider is forced explicitly: the default execution provider on Apple
        # Silicon prefers CoreML, which crashes/hangs under asyncio.to_thread (see
        # core/config.py's reranker_mode comment). CPUExecutionProvider is stable
        # and portable to the eventual cloud host.
        return CrossEncoder(
            model_name,
            backend="onnx",
            max_length=max_length,
            model_kwargs={"file_name": onnx_file, "provider": "CPUExecutionProvider"},
        )
    # device="cpu" is required, not cosmetic: sentence-transformers silently
    # defaults to MPS on Apple Silicon when no device is given, which would make
    # "pytorch-cpu" measure GPU-accelerated latency -- not what the mode name
    # promises, and not representative of the CPU-only cloud host this targets.
    return CrossEncoder(model_name, max_length=max_length, device="cpu")


def rerank(query: str, candidates: list[ScoredChunk], settings: Settings) -> list[ScoredChunk]:
    """Score candidates by cross-encoder relevance; return the top ``rerank_top_n``.

    Each chunk's stage-1 (RRF fusion) score is replaced by the reranker's relevance
    score -- a different scale and meaning, and the one that drives final ordering.
    """
    if not candidates:
        return []
    model = _model(
        settings.reranker_model,
        settings.reranker_max_length,
        settings.reranker_mode,
        settings.reranker_onnx_file,
    )
    pairs = [(query, c.text) for c in candidates]
    scores = model.predict(pairs)
    scored = [
        c.model_copy(update={"score": float(s)}) for c, s in zip(candidates, scores, strict=True)
    ]
    scored.sort(key=lambda c: c.score, reverse=True)
    return scored[: settings.rerank_top_n]


async def rerank_async(
    query: str, candidates: list[ScoredChunk], settings: Settings
) -> list[ScoredChunk]:
    """Async wrapper: reranking is CPU-bound, dispatched to a thread."""
    return await asyncio.to_thread(rerank, query, candidates, settings)
