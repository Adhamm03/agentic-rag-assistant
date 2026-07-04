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
