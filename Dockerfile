# Backend image — FastAPI + the RAG pipeline (CPU-only).
FROM python:3.11-slim

# libgomp1 is required by torch / onnxruntime at runtime.
RUN apt-get update \
 && apt-get install -y --no-install-recommends libgomp1 \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install the CPU-only torch first so the CUDA build (multi-GB) is never pulled,
# then the rest of the dependencies (sentence-transformers etc. see torch satisfied).
COPY requirements.txt .
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu \
 && pip install --no-cache-dir -r requirements.txt

# App code + the corpus / eval fixtures.
COPY app ./app
COPY data ./data
COPY eval ./eval

ENV PYTHONUNBUFFERED=1 \
    HF_HOME=/root/.cache/huggingface

# Models (bge-m3, reranker, bm25) download on first startup into HF_HOME — mount
# a volume there (see docker-compose) so they persist across restarts.
EXPOSE 8000
CMD ["python", "-m", "uvicorn", "app.api:app", "--host", "0.0.0.0", "--port", "8000"]
