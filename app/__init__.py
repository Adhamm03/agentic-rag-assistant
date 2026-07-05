"""Agentic RAG Assistant — backend package.

Modules:
    config        central, env-driven settings (paths, models, retrieval params)
    embeddings    dense (bge-m3) + sparse (bm25) model loaders and embed functions
    vector_store  Qdrant client, collection setup, and upsert
    ingest        load PDFs -> chunk -> embed + upsert (CLI)
    retriever     hybrid dense+sparse retrieval fused with RRF, then rerank
    agent         LangGraph agent: route -> retrieve -> grounding -> generate
    api           FastAPI app exposing /ingest and /chat
"""

# Windows: pre-load pyarrow before torch/sentence-transformers/datasets. When
# pyarrow's native extension is loaded late and re-entrantly (via the
# sentence_transformers -> datasets -> pandas -> pyarrow chain, after torch/
# scipy/sklearn are already resident), its init hits a 0xC0000005 access
# violation that silently kills the process. Loading it first avoids the crash.
import sys as _sys

if _sys.platform == "win32":
    import pyarrow.dataset  # noqa: F401
