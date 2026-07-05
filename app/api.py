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
READY = threading.Event() # set the instant models finish loading ("Ready.")
LOAD_ERROR: dict = {}     # holds {"error": msg} if startup loading failed

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


def _load_models() -> None:
    """Load models + Qdrant. Runs in a background thread so the HTTP server (and
    the /api/health probe) is available immediately, reporting "loading" until
    READY is set here — i.e. exactly when the log prints "Ready."."""
    try:
        print("Loading models + Qdrant (one-time startup)...")
        retriever = Retriever(collection=config.DEFAULT_COLLECTION)
        STATE["retriever"] = retriever
        STATE["agent"] = build_agent(retriever=retriever)  # share client + models
        READY.set()
        print("Ready.")
    except Exception as e:  # surface via /api/health instead of a silent crash
        LOAD_ERROR["error"] = str(e)
        print(f"Startup failed: {e}")


@app.on_event("startup")
def _startup() -> None:
    # Load in the background: the server binds and serves the health probe right
    # away, so the frontend sees a clean "loading" state instead of a hanging /
    # refused connection while the (multi-GB) models load on first boot.
    threading.Thread(target=_load_models, name="model-loader", daemon=True).start()


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
    """Readiness probe. ready=True only once the models have finished loading
    (mirrors the backend log's "Ready."); "loading" until then, "error" if
    startup failed. The frontend polls this and gates its ready indicator on it.
    """
    if READY.is_set():
        return {"status": "ready", "ready": True}
    if LOAD_ERROR:
        return {"status": "error", "ready": False, "error": LOAD_ERROR["error"]}
    return {"status": "loading", "ready": False}


def _require_ready() -> None:
    """503 while models are still loading, so early calls get a clean signal."""
    if not READY.is_set():
        if LOAD_ERROR:
            raise HTTPException(status_code=503, detail=f"Backend failed to start: {LOAD_ERROR['error']}")
        raise HTTPException(status_code=503, detail="Models are still loading, please wait.")


@api.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest) -> ChatResponse:
    _require_ready()
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
    _require_ready()
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
