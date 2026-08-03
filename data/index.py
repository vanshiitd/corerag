"""Qdrant hybrid collection management and chunk upsert."""

from __future__ import annotations

import uuid

from qdrant_client import QdrantClient, models

from core.config import Settings
from data.schemas import Chunk


def make_client(settings: Settings) -> QdrantClient:
    """Synchronous Qdrant client for the offline ingestion pipeline."""
    api_key = settings.qdrant_api_key.get_secret_value() if settings.qdrant_api_key else None
    return QdrantClient(url=settings.qdrant_url, api_key=api_key)


def ensure_collection(client: QdrantClient, settings: Settings, *, recreate: bool = False) -> None:
    """Create the hybrid (dense + BM25 sparse) collection if needed."""
    name = settings.qdrant_collection_name
    if recreate and client.collection_exists(name):
        client.delete_collection(name)
    if not client.collection_exists(name):
        client.create_collection(
            name,
            vectors_config={
                "dense": models.VectorParams(
                    size=settings.dense_embed_dim, distance=models.Distance.COSINE
                )
            },
            sparse_vectors_config={
                "sparse": models.SparseVectorParams(modifier=models.Modifier.IDF)
            },
        )


def is_paper_indexed(client: QdrantClient, settings: Settings, arxiv_id: str) -> bool:
    """True if any chunk for this paper is already in the collection.

    Lets the pipeline skip (and avoid re-billing LLM contextualization for) papers
    already ingested -- e.g. after resuming an interrupted run.
    """
    name = settings.qdrant_collection_name
    if not client.collection_exists(name):
        return False
    # exact=True is required for correctness: Qdrant's approximate count mode
    # (exact=False) is unreliable on small collections -- it returned a nonzero
    # count for an arxiv_id that plainly didn't exist, which would have skipped
    # (and silently never indexed) every paper in a real ingestion run.
    count = client.count(
        name,
        count_filter=models.Filter(
            must=[models.FieldCondition(key="arxiv_id", match=models.MatchValue(value=arxiv_id))]
        ),
        exact=True,
    )
    return count.count > 0


def upsert_chunks(
    client: QdrantClient,
    settings: Settings,
    chunks: list[Chunk],
    dense_vecs: list[list[float]],
    sparse_vecs: list[tuple[list[int], list[float]]],
) -> int:
    """Upsert chunks with deterministic (idempotent) point IDs."""
    points: list[models.PointStruct] = []
    for chunk, dense, (indices, values) in zip(chunks, dense_vecs, sparse_vecs, strict=True):
        points.append(
            models.PointStruct(
                id=str(uuid.uuid5(uuid.NAMESPACE_URL, chunk.chunk_id)),
                vector={
                    "dense": dense,
                    "sparse": models.SparseVector(indices=indices, values=values),
                },
                payload=chunk.model_dump(),
            )
        )
    client.upsert(settings.qdrant_collection_name, points=points)
    return len(points)
