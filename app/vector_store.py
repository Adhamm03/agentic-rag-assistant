"""
vector_store.py — Qdrant connection, hybrid collection setup, and upsert.

The collection holds two named vectors per point:
    dense  : cosine-distance bge-m3 vector
    sparse : BM25 vector with Qdrant's IDF modifier enabled
Together these form the hybrid index queried in retriever.py.
"""

from __future__ import annotations

import uuid

from . import config
from .embeddings import embed_dense, embed_sparse_docs


def get_client():
    """Connect to Qdrant.

    QDRANT_URL set -> server/cloud connection. Otherwise embedded on-disk mode
    (QDRANT_PATH) — an in-process store like ChromaDB's PersistentClient.

    Note: embedded mode holds an exclusive file lock on the folder, so only one
    process (a CLI OR the API server) can open it at a time.
    """
    from qdrant_client import QdrantClient

    if config.QDRANT_URL:
        print(f"Connecting to Qdrant server: {config.QDRANT_URL}")
        return QdrantClient(url=config.QDRANT_URL, api_key=config.QDRANT_API_KEY)

    print(f"Using embedded Qdrant at: {config.QDRANT_PATH}")
    return QdrantClient(path=config.QDRANT_PATH)


def reset_embedded_store() -> None:
    """Delete the embedded on-disk store for a truly clean rebuild.

    In embedded (local) mode, Qdrant's delete_collection doesn't reliably purge
    the persisted folder, so a real ``--recreate`` wipes the directory before the
    client opens it. No-op when using a Qdrant server/cloud (QDRANT_URL set).
    """
    import shutil
    from pathlib import Path

    if config.QDRANT_URL:
        return
    path = Path(config.QDRANT_PATH)
    if path.exists():
        shutil.rmtree(path, ignore_errors=True)
        print(f"Wiped embedded store: {path}")


def ensure_collection(client, collection: str, recreate: bool = False) -> None:
    """Create the hybrid collection if needed (optionally recreating it)."""
    from qdrant_client import models

    exists = client.collection_exists(collection)
    if exists and recreate:
        print(f"Recreating collection '{collection}' ...")
        client.delete_collection(collection)
        exists = False

    if not exists:
        client.create_collection(
            collection_name=collection,
            vectors_config={
                config.DENSE_VECTOR_NAME: models.VectorParams(
                    size=config.DENSE_VECTOR_SIZE,
                    distance=models.Distance.COSINE,
                )
            },
            sparse_vectors_config={
                config.SPARSE_VECTOR_NAME: models.SparseVectorParams(
                    modifier=models.Modifier.IDF,
                )
            },
        )
        print(f"Created collection '{collection}'")
    else:
        print(f"Using existing collection '{collection}'")


def upsert_chunks(client, collection: str, chunks: list,
                  dense_model, sparse_model) -> int:
    """Embed (dense + sparse) and upsert all chunks; returns the number indexed."""
    from qdrant_client import models

    total = 0
    for start in range(0, len(chunks), config.UPSERT_BATCH_SIZE):
        batch = chunks[start:start + config.UPSERT_BATCH_SIZE]
        texts = [c.page_content for c in batch]

        dense_vecs = embed_dense(dense_model, texts)
        sparse_vecs = embed_sparse_docs(sparse_model, texts)

        points = []
        for chunk, dense_vec, (sp_idx, sp_val) in zip(batch, dense_vecs, sparse_vecs):
            meta = chunk.metadata
            points.append(
                models.PointStruct(
                    id=str(uuid.uuid4()),
                    vector={
                        config.DENSE_VECTOR_NAME: dense_vec,
                        config.SPARSE_VECTOR_NAME: models.SparseVector(
                            indices=sp_idx, values=sp_val),
                    },
                    payload={
                        "source": meta.get("source", "unknown"),
                        "page": int(meta.get("page", -1)),
                        "chunk_id": meta.get("chunk_id"),
                        "text": chunk.page_content,
                    },
                )
            )

        client.upsert(collection_name=collection, points=points)
        total += len(points)
        print(f"  upserted {total}/{len(chunks)} chunks")

    return total
