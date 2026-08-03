"""Local dense + sparse embeddings via FastEmbed (ONNX, no torch)."""

from __future__ import annotations

from functools import lru_cache

from fastembed import SparseTextEmbedding, TextEmbedding

from core.config import Settings


@lru_cache(maxsize=2)
def _dense(model_name: str) -> TextEmbedding:
    return TextEmbedding(model_name=model_name)


@lru_cache(maxsize=2)
def _sparse(model_name: str) -> SparseTextEmbedding:
    return SparseTextEmbedding(model_name=model_name)


def embed_dense(texts: list[str], settings: Settings) -> list[list[float]]:
    """Dense vectors for a batch of texts."""
    model = _dense(settings.dense_embed_model)
    return [vec.tolist() for vec in model.embed(texts)]


def embed_sparse(texts: list[str], settings: Settings) -> list[tuple[list[int], list[float]]]:
    """Sparse (BM25) vectors as (indices, values) for a batch of texts."""
    model = _sparse(settings.sparse_embed_model)
    return [(emb.indices.tolist(), emb.values.tolist()) for emb in model.embed(texts)]
