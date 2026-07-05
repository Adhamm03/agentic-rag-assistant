"""
config.py — central, env-driven configuration for the Agentic RAG Assistant.

Everything tunable lives here and is read from environment variables (loaded
from the project-root .env). Paths are resolved relative to the project root, so
the code works no matter which directory it's launched from.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

from dotenv import load_dotenv

# Project root = the parent of this app/ package.
BASE_DIR = Path(__file__).resolve().parent.parent

load_dotenv(BASE_DIR / ".env")

# UTF-8 stdout so multilingual output (e.g. Arabic) prints on Windows consoles
# that default to cp1252. Centralised here since every module imports config.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

# --------------------------------------------------------------------------- #
# Qdrant storage
# --------------------------------------------------------------------------- #
# QDRANT_URL set  -> local Docker server or Qdrant Cloud
# QDRANT_URL unset -> embedded on-disk mode at QDRANT_PATH (no Docker/cloud)
QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
QDRANT_PATH = os.getenv("QDRANT_PATH") or str(BASE_DIR / "qdrant_storage")

# --------------------------------------------------------------------------- #
# Models
# --------------------------------------------------------------------------- #
DENSE_MODEL_NAME = os.getenv("DENSE_MODEL", "BAAI/bge-m3")
SPARSE_MODEL_NAME = os.getenv("SPARSE_MODEL", "Qdrant/bm25")
RERANKER_MODEL_NAME = os.getenv("RERANKER_MODEL", "BAAI/bge-reranker-v2-m3")
DENSE_VECTOR_SIZE = 1024  # bge-m3 output dimension

# Named vectors inside each Qdrant point (dense + sparse = hybrid index).
DENSE_VECTOR_NAME = "dense"
SPARSE_VECTOR_NAME = "sparse"

# --------------------------------------------------------------------------- #
# Collection + chunking + batching
# --------------------------------------------------------------------------- #
DEFAULT_COLLECTION = os.getenv("COLLECTION_NAME", "rag_docs")
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "800"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "120"))
EMBED_BATCH_SIZE = int(os.getenv("EMBED_BATCH_SIZE", "32"))
UPSERT_BATCH_SIZE = int(os.getenv("UPSERT_BATCH_SIZE", "128"))

# --------------------------------------------------------------------------- #
# Retrieval
# --------------------------------------------------------------------------- #
# Each arm (dense/sparse) fetches PREFETCH_LIMIT; RRF fuses to FUSED_CANDIDATES,
# which the cross-encoder reranks down to TOP_K. Rerank is the CPU bottleneck,
# so the fused set is kept smaller than the arms.
PREFETCH_LIMIT = int(os.getenv("PREFETCH_LIMIT", "20"))
FUSED_CANDIDATES = int(os.getenv("FUSED_CANDIDATES", "10"))
TOP_K = int(os.getenv("TOP_K", "5"))

# --------------------------------------------------------------------------- #
# Agent
# --------------------------------------------------------------------------- #
# Below this best-rerank score, retrieval is treated as too weak to answer from
# and the agent returns "not found" instead of risking a hallucination.
# Calibrated to this corpus: genuinely off-topic queries score <=0.0004, while
# anything plausibly in the docs (after the agent's query rewrite) scores >=0.01.
# 0.01 sits in that gap; the generate prompt is the second line of defence.
GROUNDING_THRESHOLD = float(os.getenv("GROUNDING_THRESHOLD", "0.01"))
NOT_FOUND_MESSAGE = os.getenv(
    "NOT_FOUND_MESSAGE",
    "I couldn't find this in the provided documents.",
)
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "groq")
LLM_MODEL = os.getenv("LLM_MODEL")  # provider-specific default applied in agent

# RAGAS judge — kept separate from the generator so evaluation isn't self-graded.
JUDGE_PROVIDER = os.getenv("JUDGE_PROVIDER", "groq")
JUDGE_MODEL = os.getenv("JUDGE_MODEL")

# --------------------------------------------------------------------------- #
# API / CORS
# --------------------------------------------------------------------------- #
# The frontend is deployed on a different origin (e.g. Render), so allow it to
# call this API. Comma-separated list, or "*" for any origin (default).
CORS_ORIGINS = [o.strip() for o in os.getenv("CORS_ORIGINS", "*").split(",") if o.strip()]

# --------------------------------------------------------------------------- #
# Paths
# --------------------------------------------------------------------------- #
DATA_DIR = BASE_DIR / "data" / "pdfs"    # default corpus for the ingest CLI
EVAL_DIR = BASE_DIR / "eval"             # ground-truth QA lives here
