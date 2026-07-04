"""
api.py — FastAPI backend for the Agentic RAG Assistant (single-user).

Endpoints:
    GET  /health   liveness + readiness probe
    POST /chat     {question} -> {answer, route, grounded, sources}
    POST /ingest   multipart PDFs -> {sources, pages, chunks_indexed}
    POST /eval     RAGAS evaluation (Milestone 5 — stub)

Embedded Qdrant (Case A) allows only ONE process to hold the store, so this
server is that single owner: it loads the retriever (client + models) and the
agent ONCE at startup and shares them across /chat and /ingest. Don't run the
CLIs while the server is up. A lock serializes access to the store + models.

Run (from project root):
    ./venv/Scripts/python.exe -m uvicorn app.api:app --host 127.0.0.1 --port 8000
"""

from __future__ import annotations

import shutil
import tempfile
import threading
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
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
LOCK = threading.Lock()   # serialize access to the embedded store + models

app = FastAPI(title="Agentic RAG Assistant", version="1.0")

# Dev CORS: allow the Vite dev server (for the frontend we'll add later).
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_methods=["*"],
    allow_headers=["*"],
)


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
# Endpoints
# --------------------------------------------------------------------------- #

@app.get("/health")
def health() -> dict:
    return {"status": "ok", "ready": bool(STATE.get("agent"))}


@app.post("/chat", response_model=ChatResponse)
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


@app.post("/ingest", response_model=IngestResponse)
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


@app.post("/eval")
def eval_endpoint() -> dict:
    raise HTTPException(status_code=501, detail="RAGAS eval not implemented yet (Milestone 5).")
