# ── Stage 1: builder ──────────────────────────────────────────────────────────
FROM python:3.11-slim AS builder

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

RUN pip install --no-cache-dir \
    torch --index-url https://download.pytorch.org/whl/cpu \
    && pip install --no-cache-dir -r requirements.txt

# ── Stage 2: runtime ─────────────────────────────────────────────────────────
FROM python:3.11-slim AS runtime

LABEL maintainer="your-email@example.com"
LABEL description="PDF RAG Assistant — Ministral + ChromaDB CLI"

WORKDIR /app

COPY --from=builder /usr/local/lib/python3.11 /usr/local/lib/python3.11
COPY --from=builder /usr/local/bin /usr/local/bin

COPY src/       src/
COPY scripts/   scripts/
COPY configs/   configs/
COPY pyproject.toml .

RUN pip install --no-cache-dir -e .

RUN mkdir -p data/documents data/vectorstore .cache_embeddings

ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV EMBEDDING_DEVICE=cpu

CMD ["rag-query"]
