"""Integration tests for Qdrant collection/index management (P1).

Require live Qdrant. Run with:  make test-int   (i.e. uv run pytest -m integration)
"""

from __future__ import annotations

from collections.abc import Iterator

import pytest
from qdrant_client import QdrantClient, models

from core.config import Settings, get_settings
from data.index import ensure_collection, is_paper_indexed, make_client


@pytest.fixture
def isolated_client() -> Iterator[tuple[QdrantClient, Settings]]:
    """A throwaway collection, cleaned up after the test."""
    base = get_settings()
    settings = base.model_copy(update={"qdrant_collection_version": "test-is-paper-indexed"})
    client = make_client(settings)
    ensure_collection(client, settings, recreate=True)
    try:
        yield client, settings
    finally:
        client.delete_collection(settings.qdrant_collection_name)


@pytest.mark.integration
def test_is_paper_indexed_reflects_real_membership(
    isolated_client: tuple[QdrantClient, Settings],
) -> None:
    """Regression test: exact=False (Qdrant's approximate count) returned a nonzero
    count for an arxiv_id that didn't exist, which would have skipped every paper
    in a real ingestion run without ever indexing them."""
    client, settings = isolated_client

    assert is_paper_indexed(client, settings, "1234.5678") is False

    client.upsert(
        settings.qdrant_collection_name,
        points=[
            models.PointStruct(
                id="00000000-0000-0000-0000-000000000001",
                vector={
                    "dense": [0.0] * settings.dense_embed_dim,
                    "sparse": models.SparseVector(indices=[0], values=[1.0]),
                },
                payload={"arxiv_id": "1234.5678"},
            )
        ],
    )

    assert is_paper_indexed(client, settings, "1234.5678") is True
    assert is_paper_indexed(client, settings, "9999.99999") is False
