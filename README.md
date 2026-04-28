# PDF RAG Assistant

A local Retrieval-Augmented Generation (RAG) pipeline that runs **entirely on your machine** — no API keys, no cloud. Ingest PDF documents and ask natural-language questions about them via the terminal, powered by **Ministral 8B** running via llama.cpp and **ChromaDB** for semantic retrieval.

## Features

- **100% local inference** — Ministral 8B GGUF via llama-cpp-python; model downloads once (~5 GB) and runs offline after that
- **Multi-format ingestion** — PDF, TXT, Markdown, DOCX
- **Streaming responses** — token-by-token output in the terminal
- **Inline citations** — every answer links back to the source chunk and page number
- **Multi-turn chat** — conversation history injected into the model context
- **Upsert deduplication** — re-ingesting the same document won't create duplicate chunks
- **Two CLI commands** — batch-ingest documents and an interactive query REPL
- **Configurable** — all parameters (model, chunk size, retrieval threshold, …) via `.env`
- **Docker-ready** — two-stage Dockerfile + Compose file included

---

## Architecture

```
PDF / TXT / DOCX
      │
      ▼
 ┌─────────────┐   chunk + embed (local)   ┌──────────────────┐
 │   Ingest    │ ─────────────────────────▶│  ChromaDB        │
 │             │   sentence-transformers   │  (on disk)       │
 └─────────────┘                           └──────────────────┘
                                                    │
                                         similarity search (k chunks)
                                                    │
                           ┌────────────────────────▼──────────────────────┐
                           │              Retriever                        │
                           │   filters by L2 distance threshold            │
                           └────────────────────────┬──────────────────────┘
                                                    │
                                         context + question
                                                    │
                           ┌────────────────────────▼──────────────────────┐
                           │     Ministral 8B (local GGUF via llama.cpp)   │
                           │     Metal / CUDA / CPU — streaming output     │
                           └────────────────────────┬──────────────────────┘
                                                    │
                                          answer + citations
```

| Component | Technology |
|---|---|
| LLM | Ministral-8B-Instruct-2410 (GGUF, quantised Q4_K_M) |
| Embeddings | `sentence-transformers/all-MiniLM-L6-v2` (local) |
| Vector store | ChromaDB (persisted on disk) |
| Inference engine | llama-cpp-python (Metal / CUDA / CPU) |
| Orchestration | LangChain |
| CLI | Rich + argparse |

---

## Installation

### Prerequisites — install these first (platform-specific)

**PyTorch:**
```bash
# Apple Silicon
pip install torch
# CUDA 12.x
pip install torch --index-url https://download.pytorch.org/whl/cu121
# CPU only
pip install torch --index-url https://download.pytorch.org/whl/cpu
```

**llama-cpp-python (with GPU acceleration):**
```bash
# Apple Silicon (Metal — ~5× faster than CPU)
CMAKE_ARGS="-DGGML_METAL=on" pip install llama-cpp-python
# CUDA
CMAKE_ARGS="-DGGML_CUDA=on" pip install llama-cpp-python
# CPU only
pip install llama-cpp-python
```

### Install the package

**Option A — from GitHub (end users):**
```bash
pip install "rag-project @ git+https://github.com/AdelAlmutairii/rag-project.git"
```

**Option B — editable install from a local clone (development):**
```bash
git clone https://github.com/AdelAlmutairii/rag-project.git
cd rag-project
pip install -e ".[dev]"
```

After installation two commands are available in your shell:

| Command | What it does |
|---|---|
| `rag-ingest` | Batch-ingest documents into the vector store |
| `rag-query` | Interactive terminal REPL |

---

## Quick Start

### 1. Configure

Copy `.env.example` to `.env` in your working directory:
```bash
cp .env.example .env
```

The defaults work out of the box — the model is downloaded automatically on first run (~5 GB, cached in `~/.cache/huggingface/`).

### 2. Add documents

```bash
mkdir -p data/documents
cp /path/to/your/files.pdf data/documents/
```

### 3. Ingest

```bash
rag-ingest                          # reads data/documents/ by default
rag-ingest --file report.pdf        # single file
rag-ingest --dir /path/to/pdfs      # custom directory
rag-ingest --reset --dir ./docs     # wipe store and re-ingest
```

### 4. Query

```bash
rag-query                                        # interactive REPL
rag-query --question "What is the main finding?" # single question
rag-query --source report.pdf                    # restrict to one document
```

**Inside the REPL:**

| Command | Action |
|---|---|
| Type a question + Enter | Ask anything |
| `/sources` | List indexed documents |
| `/reset` | Clear chat history |
| `/quit` | Exit |

---

## Configuration

All settings are read from environment variables or a `.env` file. See [`.env.example`](.env.example) for the full list.

| Variable | Default | Description |
|---|---|---|
| `MODEL_REPO` | `bartowski/Ministral-8B-Instruct-2410-GGUF` | HuggingFace repo for the GGUF model |
| `MODEL_FILENAME` | `Ministral-8B-Instruct-2410-Q4_K_M.gguf` | GGUF filename (supports `*` wildcards) |
| `MODEL_PATH` | *(unset)* | Absolute path to a local `.gguf` file — skips the HF download |
| `N_CTX` | `8192` | Context window in tokens |
| `N_GPU_LAYERS` | `-1` | Layers offloaded to GPU (`-1` = all, `0` = CPU only) |
| `N_THREADS` | `8` | CPU threads for inference |
| `MAX_TOKENS` | `1024` | Maximum generated tokens per response |
| `TEMPERATURE` | `0.1` | Sampling temperature (0 = deterministic) |
| `EMBEDDING_MODEL` | `sentence-transformers/all-MiniLM-L6-v2` | Local embedding model |
| `EMBEDDING_DEVICE` | `cpu` | `cpu`, `cuda`, or `mps` |
| `CHUNK_SIZE` | `1000` | Characters per chunk |
| `CHUNK_OVERLAP` | `200` | Overlap between chunks |
| `RETRIEVAL_K` | `5` | Chunks retrieved per query |
| `MAX_DISTANCE` | `1.4` | Chroma L2 distance cut-off (lower = stricter) |
| `VECTORSTORE_DIR` | `data/vectorstore` | ChromaDB persistence path |
| `DOCUMENTS_DIR` | `data/documents` | Default ingestion directory |

### Alternative GGUF models

Any GGUF model that follows the Mistral/ChatML chat template works:

| Model | Size (Q4_K_M) | Repo |
|---|---|---|
| Ministral 3B *(faster, lighter)* | ~2 GB | `bartowski/Ministral-3B-Instruct-2410-GGUF` |
| **Ministral 8B *(default)*** | ~5 GB | `bartowski/Ministral-8B-Instruct-2410-GGUF` |
| Mistral 7B v0.3 | ~4.4 GB | `bartowski/Mistral-7B-Instruct-v0.3-GGUF` |

---

## Project Structure

```
rag-project/
├── src/
│   └── rag/                     # installable Python package
│       ├── __init__.py
│       ├── config.py            # pydantic-settings (reads .env)
│       ├── prompts.py           # system prompt + context formatter
│       ├── embeddings.py        # HuggingFace embeddings (cached)
│       ├── ingest.py            # document loading + chunking
│       ├── vectorstore.py       # ChromaDB wrapper
│       ├── llm.py               # llama-cpp-python wrapper (sync + stream)
│       ├── retriever.py         # semantic search + distance filter
│       ├── pipeline.py          # RAGPipeline orchestrator
│       └── cli/
│           ├── __init__.py
│           ├── ingest.py        # rag-ingest entry point
│           └── query.py         # rag-query entry point
│
├── scripts/
│   ├── ingest_docs.py           # thin wrapper (backward compat)
│   └── query.py                 # thin wrapper (backward compat)
│
├── tests/
│   ├── conftest.py
│   ├── test_ingest.py
│   ├── test_retriever.py
│   └── test_pipeline.py
│
├── configs/default.yaml         # parameter documentation + defaults
├── data/documents/              # drop your files here (git-ignored)
├── .env.example
├── conftest.py                  # puts src/ on sys.path for pytest
├── Dockerfile
├── docker-compose.yml
├── Makefile
├── pyproject.toml
└── requirements.txt
```

---

## Development

```bash
# Run tests (no model or API needed — LLM is mocked)
make test

# With coverage
make test-cov

# Lint + format
make lint
make format
```

---

## Docker

```bash
# Build the image
docker compose build

# Ingest documents
docker compose run rag-ingest

# Interactive query REPL
docker compose run rag-query
```

Model weights are cached in a named Docker volume so they survive container restarts.

---

## Tuning Retrieval Quality

| Problem | Fix |
|---|---|
| Answers say "not found" too often | Raise `MAX_DISTANCE` (e.g. `1.7`) |
| Answers contain off-topic content | Lower `MAX_DISTANCE` (e.g. `1.0`) |
| Missing context in answers | Raise `RETRIEVAL_K` (e.g. `8`) |
| Slow inference | Lower `N_CTX`, or switch to Ministral 3B |
| Low-quality PDF extraction | Use `CHUNK_SIZE=600` for denser docs |

---

## License

MIT — see [LICENSE](LICENSE) for details.
