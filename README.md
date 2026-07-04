# Agentic RAG Assistant

Hybrid-retrieval RAG over your PDFs with reranking, an agentic answer layer that
decides when to retrieve and refuses to hallucinate, and (planned) a RAGAS
evaluation. Multilingual (Arabic + English).

## Architecture

```
PDFs ─▶ ingest ─▶ chunk ─▶ embed (bge-m3 dense + bm25 sparse) ─▶ Qdrant (hybrid)

Q ─▶ agent ─┬─ route ── direct ─▶ answer
            └─ retrieve ─▶ hybrid search (RRF) ─▶ rerank (bge-reranker-v2-m3)
                        ─▶ grounding check ─▶ generate cited answer / "not found"
```

## Project layout

```
frontend/            React + TypeScript (Vite) UI — "Sanad" chat interface
app/                 backend package
  config.py          env-driven settings (paths, models, retrieval params)
  embeddings.py      dense (bge-m3) + sparse (bm25) loaders and embed helpers
  vector_store.py    Qdrant client, collection setup, upsert
  ingest.py          load PDFs -> chunk -> embed + upsert (CLI)
  retriever.py       hybrid dense+sparse retrieval fused with RRF, then rerank
  agent.py           LangGraph agent: route -> retrieve -> grounding -> generate
  api.py             FastAPI app: /health /chat /ingest /eval
data/pdfs/           the document corpus
eval/                ground-truth QA (ground_truth_qa.json / .csv) for RAGAS
qdrant_storage/      embedded on-disk vector index (gitignored, regenerable)
```

## Setup

```bash
python -m venv venv
./venv/Scripts/pip install -r requirements.txt   # Windows
cp .env.example .env                              # then add your GROQ_API_KEY
```

Storage defaults to **embedded** Qdrant (no Docker). To use a server or Qdrant
Cloud instead, set `QDRANT_URL` (+ `QDRANT_API_KEY`) in `.env`.

## Usage

Run everything as modules from the project root:

```bash
# Ingest the corpus (‑‑recreate rebuilds the index from scratch)
python -m app.ingest --pdf-dir data/pdfs --recreate

# Inspect retrieval for one query
python -m app.retriever "What does the policy cover?"

# Full agentic answer in the terminal
python -m app.agent "ما هي الاستثناءات في الوثيقة؟"

# Serve the API (docs at http://127.0.0.1:8000/docs)
python -m uvicorn app.api:app --host 127.0.0.1 --port 8000
```

### Frontend (React UI)

```bash
cd frontend
npm install        # first time
npm run dev        # http://localhost:5173  (proxies /api -> :8000)
```

Run the API (port 8000) and the frontend (port 5173) together — the Vite dev
server proxies `/api/*` to the backend. Features: EN/AR toggle with RTL, light/dark
themes, PDF upload, and cited answers with a collapsible Sources panel.

### API

| Method | Path      | Body                    | Returns |
|--------|-----------|-------------------------|---------|
| GET    | `/health` | —                       | readiness |
| POST   | `/chat`   | `{ "question": "..." }` | `{ answer, route, grounded, sources }` |
| POST   | `/ingest` | multipart PDF file(s)   | `{ sources, pages, chunks_indexed }` |
| POST   | `/eval`   | —                       | RAGAS eval (planned) |

> The embedded vector store is single-process: run the CLIs **or** the API
> server, not both at once (they contend for the same `qdrant_storage` lock).

## Run with Docker

Full stack (Qdrant + backend + nginx-served frontend) via Docker Compose:

```bash
# needs a .env at the repo root with OPENAI_API_KEY (the generator)
docker compose up --build
```

- Frontend → **http://localhost:5173** (nginx proxies `/api` to the backend)
- Backend → **http://localhost:8000** (`/docs`)

Then populate the index (Qdrant starts empty):

```bash
docker compose exec backend python -m app.ingest --pdf-dir data/pdfs
```

Notes:
- The backend uses the **Qdrant service** (`QDRANT_URL=http://qdrant:6333`), not
  embedded mode — no single-process lock, and data persists in the `qdrant_data`
  volume.
- **First startup is slow:** the backend downloads the bge-m3 + reranker models
  (~4 GB) into the `model_cache` volume; subsequent starts reuse it.
- torch is installed CPU-only to keep the image lean; there's no GPU dependency.

## Evaluation (RAGAS)

`python -m app.evaluate` scores retrieval quality — **naive vector search vs
hybrid + rerank** — on the ground-truth QA in `eval/`, using an **independent
judge** (different model family from the generator, so answers aren't
self-graded). Metrics: faithfulness, answer_relevancy, context_precision,
context_recall. Outputs: `eval/results.json`, `eval/report.md`,
`eval/ragas_comparison.png`, presented in `eval/ragas_eval.ipynb`.

> ⚠️ The committed numbers reflect **2 questions only** — the full 16-question
> run was cut short by free-tier LLM-judge **rate limits** (a local judge is
> impractically slow on CPU). They're an illustrative smoke test, not a
> statistically meaningful result; re-run with a higher-quota judge to reproduce
> the full evaluation. See [eval/report.md](eval/report.md).

## Notes

- On CPU, `bge-m3` + `bge-reranker-v2-m3` make ingestion and each answer take
  tens of seconds — expected; a GPU makes it fast.
- LLM provider is configurable (`LLM_PROVIDER=groq` default, or `gemini`).
