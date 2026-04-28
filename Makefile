# ════════════════════════════════════════════════════════════════════════════
# PDF RAG Assistant — Makefile
#
# Works in two modes:
#   1. After `pip install -e .`  → uses the installed console scripts
#   2. Direct (no install)       → sets PYTHONPATH=src and calls modules
# ════════════════════════════════════════════════════════════════════════════

.PHONY: help install install-dev ingest ingest-reset ingest-file query app \
        test test-cov lint format type-check clean reset-store

PYTHON    ?= python
PIP       ?= pip
DOCS_DIR  ?= data/documents

# Add src/ to the path so the package is importable without pip install
export PYTHONPATH := src

# ── Help ─────────────────────────────────────────────────────────────────────

help:
	@echo ""
	@echo "  PDF RAG Assistant"
	@echo ""
	@echo "  Setup"
	@echo "    install        Install runtime dependencies (+ editable package)"
	@echo "    install-dev    Install runtime + dev tools"
	@echo ""
	@echo "  Data pipeline"
	@echo "    ingest         Ingest all docs in data/documents/"
	@echo "    ingest-reset   Wipe vector store, then re-ingest"
	@echo "    ingest-file    Ingest a single file: make ingest-file FILE=path/to/doc.pdf"
	@echo ""
	@echo "  Run"
	@echo "    app            Launch Streamlit web UI"
	@echo "    query          Launch interactive terminal REPL"
	@echo ""
	@echo "  Quality"
	@echo "    test           Run test suite"
	@echo "    test-cov       Run tests with coverage report"
	@echo "    lint           Check code style (ruff)"
	@echo "    format         Auto-format code (ruff)"
	@echo "    type-check     Run mypy"
	@echo ""
	@echo "  Maintenance"
	@echo "    clean          Remove Python cache files"
	@echo "    reset-store    Delete the vector store (requires confirmation)"
	@echo ""

# ── Setup ────────────────────────────────────────────────────────────────────

install:
	@echo "Installing PyTorch and llama-cpp-python for your platform first."
	@echo "See README for GPU-specific build flags, then run:"
	@echo "  pip install -e ."
	$(PIP) install -e .

install-dev:
	$(PIP) install -e ".[dev]"

# ── Data pipeline ────────────────────────────────────────────────────────────

ingest:
	$(PYTHON) -m rag.cli.ingest --dir $(DOCS_DIR)

ingest-reset:
	$(PYTHON) -m rag.cli.ingest --dir $(DOCS_DIR) --reset

ingest-file:
	@test -n "$(FILE)" || (echo "Usage: make ingest-file FILE=path/to/doc.pdf" && exit 1)
	$(PYTHON) -m rag.cli.ingest --file $(FILE)

# ── Run ──────────────────────────────────────────────────────────────────────

app:
	streamlit run src/rag/app/main.py

query:
	$(PYTHON) -m rag.cli.query

# ── Testing ──────────────────────────────────────────────────────────────────

test:
	pytest tests/ -v

test-cov:
	pytest tests/ -v --cov=rag --cov-report=term-missing --cov-report=html

# ── Code quality ──────────────────────────────────────────────────────────────

lint:
	ruff check src/ tests/

format:
	ruff format src/ tests/
	ruff check --fix src/ tests/

type-check:
	mypy src/

# ── Maintenance ───────────────────────────────────────────────────────────────

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -name "*.pyc" -delete 2>/dev/null || true
	rm -rf .pytest_cache .mypy_cache .ruff_cache htmlcov .coverage dist build

reset-store:
	@echo "WARNING: This will permanently delete data/vectorstore/"
	@read -p "Are you sure? [y/N] " c && [ "$$c" = "y" ]
	rm -rf data/vectorstore/
	@echo "Vector store cleared."
