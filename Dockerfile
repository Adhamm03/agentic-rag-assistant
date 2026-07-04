# Backend image — FastAPI + the RAG pipeline (CPU-only, API only).
# The frontend is deployed separately (e.g. Render static site).
# Works on Hugging Face Spaces (runs containers as non-root, UID 1000).

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

# Models (bge-m3, reranker, bm25) download on first startup into HF_HOME.
EXPOSE 7860
CMD python -m uvicorn app.api:app --host 0.0.0.0 --port ${PORT:-7860}
