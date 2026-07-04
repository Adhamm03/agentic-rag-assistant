"""
retriever.py — Hybrid retrieval + reranking.

    query -> embed (dense + sparse)
          -> Qdrant hybrid search: top-N dense AND top-N sparse, fused with RRF
          -> rerank fused candidates with a cross-encoder (bge-reranker-v2-m3)
          -> keep top-k

RRF fusion runs server-side in a single query_points call (prefetch on the dense
and sparse vectors + FusionQuery(RRF)).

Run as a module:
    python -m app.retriever "What is the refund policy?"
"""

from __future__ import annotations

import argparse

from . import config
from .embeddings import (
    embed_dense_query,
    embed_sparse_query,
    load_dense_model,
    load_sparse_model,
)
from .vector_store import get_client


class Retriever:
    """Loads the query embedders, reranker, and Qdrant client once, then serves
    hybrid+reranked retrieval per query. Instantiate once and reuse.
    """

    def __init__(self, collection: str = config.DEFAULT_COLLECTION):
        from sentence_transformers import CrossEncoder

        self.collection = collection
        self.client = get_client()
        self.dense_model = load_dense_model()
        self.sparse_model = load_sparse_model()

        print(f"Loading reranker:     {config.RERANKER_MODEL_NAME} ...")
        self.reranker = CrossEncoder(config.RERANKER_MODEL_NAME)

    # ------------------------- hybrid + rerank ----------------------------- #

    def hybrid_search(self, query: str, candidates: int = config.FUSED_CANDIDATES,
                      prefetch_limit: int = config.PREFETCH_LIMIT) -> list[dict]:
        """Dense + sparse retrieval fused with RRF (server-side). Returns fused
        candidates as dicts with the RRF score and payload."""
        from qdrant_client import models

        dense_vec = embed_dense_query(self.dense_model, query)
        sp_idx, sp_val = embed_sparse_query(self.sparse_model, query)
        sparse_vec = models.SparseVector(indices=sp_idx, values=sp_val)

        response = self.client.query_points(
            collection_name=self.collection,
            prefetch=[
                models.Prefetch(query=dense_vec, using=config.DENSE_VECTOR_NAME,
                                limit=prefetch_limit),
                models.Prefetch(query=sparse_vec, using=config.SPARSE_VECTOR_NAME,
                                limit=prefetch_limit),
            ],
            query=models.FusionQuery(fusion=models.Fusion.RRF),
            limit=candidates,
            with_payload=True,
        )

        return [
            {
                "rrf_score": point.score,
                "source": point.payload.get("source"),
                "page": point.payload.get("page"),
                "chunk_id": point.payload.get("chunk_id"),
                "text": point.payload.get("text", ""),
            }
            for point in response.points
        ]

    def rerank(self, query: str, candidates: list[dict],
               top_k: int = config.TOP_K) -> list[dict]:
        """Cross-encoder rerank; attaches ``rerank_score`` and returns top_k."""
        if not candidates:
            return []

        pairs = [(query, c["text"]) for c in candidates]
        scores = self.reranker.predict(pairs)
        for cand, score in zip(candidates, scores):
            cand["rerank_score"] = float(score)

        ranked = sorted(candidates, key=lambda c: c["rerank_score"], reverse=True)
        return ranked[:top_k]

    def retrieve(self, query: str, top_k: int = config.TOP_K,
                 candidates: int = config.FUSED_CANDIDATES) -> list[dict]:
        """End-to-end: hybrid search -> rerank -> top_k."""
        fused = self.hybrid_search(query, candidates=candidates)
        return self.rerank(query, fused, top_k=top_k)


# --------------------------------------------------------------------------- #
# CLI checkpoint — ask a question, print the reranked chunks + scores
# --------------------------------------------------------------------------- #

def main() -> None:
    parser = argparse.ArgumentParser(description="Hybrid retrieve + rerank against Qdrant.")
    parser.add_argument("query", help="The question to retrieve context for")
    parser.add_argument("--collection", default=config.DEFAULT_COLLECTION)
    parser.add_argument("--top-k", type=int, default=config.TOP_K)
    parser.add_argument("--candidates", type=int, default=config.FUSED_CANDIDATES,
                        help="Fused candidates to rerank before trimming to top-k")
    args = parser.parse_args()

    retriever = Retriever(collection=args.collection)
    results = retriever.retrieve(args.query, top_k=args.top_k, candidates=args.candidates)

    print("=" * 72)
    print(f"Query: {args.query}")
    print(f"Top {len(results)} reranked chunks:")
    print("=" * 72)
    for i, r in enumerate(results, 1):
        preview = " ".join(r["text"].split())[:300]
        print(f"\n[{i}] rerank={r['rerank_score']:.4f}  rrf={r['rrf_score']:.4f}  "
              f"[{r['source']}:{r['page']}]  ({r['chunk_id']})")
        print(f"    {preview}...")


if __name__ == "__main__":
    main()
