# Single-container full app: build the React UI, then run FastAPI which serves
# both the API (/api) and the built UI (/). CPU-only. Works locally and on
# Hugging Face Spaces (which runs containers as a non-root user, UID 1000).

# ---- Stage 1: build the React frontend ----
FROM node:20-alpine AS frontend
WORKDIR /ui
COPY frontend/package*.json ./
RUN npm ci
COPY frontend/ ./
RUN npm run build            # -> /ui/dist

# ---- Stage 2: Python backend serving the API + the built UI ----
FROM python:3.11-slim

RUN apt-get update \
 && apt-get install -y --no-install-recommends libgomp1 \
 && rm -rf /var/lib/apt/lists/*

# Install CPU-only torch first (never pull the multi-GB CUDA build), then deps.
COPY requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu \
 && pip install --no-cache-dir -r /tmp/requirements.txt

# HF Spaces run as UID 1000 — use a writable home for model/cache downloads.
RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user \
    PYTHONUNBUFFERED=1 \
    HF_HOME=/home/user/.cache/huggingface \
    PATH=/home/user/.local/bin:$PATH
WORKDIR /home/user/app

COPY --chown=user app  ./app
COPY --chown=user data ./data
COPY --chown=user eval ./eval
COPY --chown=user --from=frontend /ui/dist ./static   # served at / by FastAPI

# Models (bge-m3, reranker, bm25) download on first startup into HF_HOME.
EXPOSE 7860
CMD python -m uvicorn app.api:app --host 0.0.0.0 --port ${PORT:-7860}
