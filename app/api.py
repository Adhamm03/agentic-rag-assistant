"""
api.py — FastAPI backend for the Agentic RAG Assistant (single-user).

Endpoints (all under /api):
    GET  /api/health   liveness + readiness probe
    POST /api/chat     {question} -> {answer, route, grounded, sources}
    POST /api/ingest   multipart PDFs -> {sources, pages, chunks_indexed}
    POST /api/eval     RAGAS evaluation (Milestone 5 — stub)

API only — the React frontend is deployed separately and calls this API via its
absolute URL (CORS is enabled). The models are loaded ONCE at startup and shared
across requests.

Run (from project root):
    ./venv/Scripts/python.exe -m uvicorn app.api:app --host 127.0.0.1 --port 8000
"""

from __future__ import annotations

import shutil
import tempfile
import threading
from pathlib import Path

from fastapi import APIRouter, FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from . import config
from .agent import build_agent
from .ingest import load_pdfs, split_documents
from .retriever import Retriever
from .vector_store import ensure_collection, upsert_chunks

# --------------------------------------------------------------------------- #
# App state — loaded once at startup
# --------------------------------------------------------------------------- #

STATE: dict = {}          # holds "retriever" and "agent"
LOCK = threading.Lock()   # serialize access to the store + models

app = FastAPI(title="Agentic RAG Assistant", version="1.0")

# CORS — the frontend is a separate origin (Render / Vite dev), so allow it.
# Configure via CORS_ORIGINS env (comma-separated), defaults to "*".
app.add_middleware(
    CORSMiddleware,
    allow_origins=config.CORS_ORIGINS,
    allow_methods=["*"],
    allow_headers=["*"],
)

api = APIRouter(prefix="/api")


@app.on_event("startup")
def _startup() -> None:
    print("Loading models + Qdrant (one-time startup)...")
    retriever = Retriever(collection=config.DEFAULT_COLLECTION)
    STATE["retriever"] = retriever
    STATE["agent"] = build_agent(retriever=retriever)  # share client + models
    print("Ready.")


# --------------------------------------------------------------------------- #
# Schemas
# --------------------------------------------------------------------------- #

class ChatRequest(BaseModel):
    question: str


class Source(BaseModel):
    source: str | None
    page: int | None
    chunk_id: str | None


class ChatResponse(BaseModel):
    answer: str
    route: str
    grounded: bool
    sources: list[Source]


class IngestResponse(BaseModel):
    sources: list[str]
    pages: int
    chunks_indexed: int


# --------------------------------------------------------------------------- #
# Endpoints (mounted under /api)
# --------------------------------------------------------------------------- #

@api.get("/health")
def health() -> dict:
    return {"status": "ok", "ready": bool(STATE.get("agent"))}


@api.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest) -> ChatResponse:
    if not req.question.strip():
        raise HTTPException(status_code=400, detail="Question is empty.")

    print(f"[chat] question={req.question!r}")  # confirm clean UTF-8 at the edge

    with LOCK:
        result = STATE["agent"].invoke({"question": req.question})

    contexts = result.get("contexts") or []
    seen, sources = set(), []
    for c in contexts:  # dedup sources by (source, page), preserve order
        key = (c.get("source"), c.get("page"))
        if key not in seen:
            seen.add(key)
            sources.append(Source(source=c.get("source"), page=c.get("page"),
                                   chunk_id=c.get("chunk_id")))

    return ChatResponse(
        answer=result.get("answer", ""),
        route=result.get("route", "direct"),
        grounded=bool(result.get("grounded")),
        sources=sources,
    )


@api.post("/ingest", response_model=IngestResponse)
def ingest(files: list[UploadFile] = File(...)) -> IngestResponse:
    pdfs = [f for f in files if (f.filename or "").lower().endswith(".pdf")]
    if not pdfs:
        raise HTTPException(status_code=400, detail="No PDF files uploaded.")

    retriever: Retriever = STATE["retriever"]
    tmp_dir = Path(tempfile.mkdtemp(prefix="rag_upload_"))
    try:
        for f in pdfs:
            dest = tmp_dir / Path(f.filename).name
            with dest.open("wb") as out:
                shutil.copyfileobj(f.file, out)

        with LOCK:
            docs = load_pdfs(tmp_dir)
            chunks = split_documents(docs)
            ensure_collection(retriever.client, config.DEFAULT_COLLECTION, recreate=False)
            n = upsert_chunks(retriever.client, config.DEFAULT_COLLECTION, chunks,
                              retriever.dense_model, retriever.sparse_model)
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)

    return IngestResponse(
        sources=sorted({Path(f.filename).name for f in pdfs}),
        pages=len(docs),
        chunks_indexed=n,
    )


@api.post("/eval")
def eval_endpoint() -> dict:
    raise HTTPException(status_code=501, detail="RAGAS eval not implemented yet (Milestone 5).")


app.include_router(api)
