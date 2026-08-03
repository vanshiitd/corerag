"""Stage-1 retrieval: Qdrant hybrid search (dense + BM25 sparse, RRF fusion).

High recall, fast -- returns K candidates for the reranker (stage 2) to cut down to
N. Embedding a query is CPU-bound (ONNX inference), so it's dispatched via
``asyncio.to_thread`` to avoid blocking the event loop.
"""

from __future__ import annotations

import asyncio

from qdrant_client import AsyncQdrantClient, models

from core.config import Settings
from data.embeddings import embed_dense, embed_sparse
from data.schemas import Chunk


class ScoredChunk(Chunk):
    """A retrieved chunk carrying its retrieval score."""

    score: float


def _to_scored_chunk(point: models.ScoredPoint) -> ScoredChunk:
    if point.payload is None:
        # Shouldn't happen with with_payload=True; fail loudly rather than crash
        # obscurely on `**None`, and rather than silently drop a search result.
        raise ValueError(f"Qdrant point {point.id} has no payload (with_payload=True set?)")
    return ScoredChunk(**point.payload, score=point.score)


async def hybrid_search(
    client: AsyncQdrantClient,
    settings: Settings,
    query: str,
    limit: int | None = None,
) -> list[ScoredChunk]:
    """Dense + sparse hybrid search with Reciprocal Rank Fusion.

    Returns up to ``limit`` (default ``settings.retrieval_k``) candidates, ordered
    by descending fused score, each with full citation payload.
    """
    k = limit or settings.retrieval_k
    (dense_vecs, sparse_vecs) = await asyncio.gather(
        asyncio.to_thread(embed_dense, [query], settings),
        asyncio.to_thread(embed_sparse, [query], settings),
    )
    dense_vec = dense_vecs[0]
    indices, values = sparse_vecs[0]

    result = await client.query_points(
        settings.qdrant_collection_name,
        prefetch=[
            models.Prefetch(query=dense_vec, using="dense", limit=k),
            models.Prefetch(
                query=models.SparseVector(indices=indices, values=values),
                using="sparse",
                limit=k,
            ),
        ],
        query=models.FusionQuery(fusion=models.Fusion.RRF),
        limit=k,
        with_payload=True,
    )
    return [_to_scored_chunk(point) for point in result.points]


async def dense_search(
    client: AsyncQdrantClient,
    settings: Settings,
    query: str,
    limit: int | None = None,
) -> list[ScoredChunk]:
    """Dense-only search (no sparse/fusion) -- useful for ablation and debugging."""
    k = limit or settings.retrieval_k
    dense_vec = (await asyncio.to_thread(embed_dense, [query], settings))[0]

    result = await client.query_points(
        settings.qdrant_collection_name,
        query=dense_vec,
        using="dense",
        limit=k,
        with_payload=True,
    )
    return [_to_scored_chunk(point) for point in result.points]
