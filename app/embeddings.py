"""
embeddings.py — dense + sparse embedding models and helpers.

- Dense:  BAAI/bge-m3 via sentence-transformers (multilingual, 1024-dim), used
          with cosine distance; embeddings are L2-normalised.
- Sparse: Qdrant/bm25 via fastembed; term-frequency sparse vectors that Qdrant
          weights with its server-side IDF modifier -> true BM25 scoring.

Kept Qdrant-agnostic: sparse helpers return plain (indices, values) tuples so
callers build whatever SparseVector representation they need.
"""

from __future__ import annotations

from . import config


# --------------------------------------------------------------------------- #
# Model loaders
# --------------------------------------------------------------------------- #

def load_dense_model():
    from sentence_transformers import SentenceTransformer

    print(f"Loading dense model:  {config.DENSE_MODEL_NAME} ...")
    return SentenceTransformer(config.DENSE_MODEL_NAME)


def load_sparse_model():
    from fastembed import SparseTextEmbedding

    print(f"Loading sparse model: {config.SPARSE_MODEL_NAME} ...")
    return SparseTextEmbedding(model_name=config.SPARSE_MODEL_NAME)


# --------------------------------------------------------------------------- #
# Dense embedding
# --------------------------------------------------------------------------- #

def embed_dense(model, texts: list[str]) -> list[list[float]]:
    """L2-normalised dense embeddings for a batch of documents."""
    vectors = model.encode(
        texts,
        batch_size=config.EMBED_BATCH_SIZE,
        normalize_embeddings=True,
        show_progress_bar=True,
        convert_to_numpy=True,
    )
    return vectors.tolist()


def embed_dense_query(model, query: str) -> list[float]:
    """L2-normalised dense embedding for a single query."""
    return model.encode(
        query, normalize_embeddings=True, convert_to_numpy=True
    ).tolist()


# --------------------------------------------------------------------------- #
# Sparse embedding (returns (indices, values) tuples)
# --------------------------------------------------------------------------- #

def embed_sparse_docs(model, texts: list[str]) -> list[tuple[list[int], list[float]]]:
    return [(sv.indices.tolist(), sv.values.tolist()) for sv in model.embed(texts)]


def embed_sparse_query(model, query: str) -> tuple[list[int], list[float]]:
    sv = next(iter(model.query_embed(query)))
    return sv.indices.tolist(), sv.values.tolist()
