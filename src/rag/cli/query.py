#!/usr/bin/env python3
"""CLI: interactive REPL for querying the RAG pipeline.

After `pip install`, this is available as the `rag-query` command.

Usage:
    rag-query
    rag-query --question "What is DRAGEN?"
    rag-query --source myfile.pdf --no-history
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.prompt import Prompt
from rich.rule import Rule
from rich.table import Table

from rag.config import Settings
from rag.pipeline import RAGPipeline

console = Console()
logging.basicConfig(level=logging.WARNING)

_BANNER = """\
[bold cyan]PDF RAG Assistant[/bold cyan] — local Ministral inference

Type your question and press Enter.
  [bold]/sources[/bold]  list indexed documents
  [bold]/reset[/bold]   clear chat history
  [bold]/quit[/bold]    exit
"""


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Interactive RAG query REPL.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--question", "-q", type=str, default=None,
                        help="Single question (non-interactive)")
    parser.add_argument("--source", type=str, default=None,
                        help="Restrict retrieval to a specific filename")
    parser.add_argument("--no-history", action="store_true",
                        help="Disable multi-turn memory")
    parser.add_argument("--k", type=int, default=None, help="Override retrieval_k")
    parser.add_argument("--verbose", "-v", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.verbose:
        logging.basicConfig(level=logging.INFO, force=True)

    try:
        base_settings = Settings()
    except Exception as exc:
        console.print(f"[red]Configuration error:[/red] {exc}")
        sys.exit(1)

    # Use model_copy() rather than mutating the Settings instance directly.
    # Direct attribute assignment on a Pydantic model is an antipattern and
    # bypasses validation; model_copy creates a properly validated new instance.
    overrides: dict = {}
    if args.k:
        overrides["retrieval_k"] = args.k
    settings = base_settings.model_copy(update=overrides) if overrides else base_settings

    with console.status("[dim]Loading model…[/dim]"):
        pipeline = RAGPipeline(settings)

    if not pipeline.is_ready():
        console.print(
            "[yellow]No documents indexed.[/yellow] "
            "Run [bold]rag-ingest[/bold] first."
        )
        sys.exit(1)

    if args.question:
        _single_query(args.question, pipeline, args.source)
    else:
        _repl(pipeline, args.source, use_history=not args.no_history)


def _single_query(question: str, pipeline: RAGPipeline, source_filter: str | None) -> None:
    result = pipeline.query(question, source_filter=source_filter)
    console.print(Rule("Answer"))
    console.print(Markdown(result.answer))
    if result.sources:
        console.print(Rule("Sources"))
        _print_sources(result.sources)


def _repl(pipeline: RAGPipeline, source_filter: str | None, use_history: bool) -> None:
    console.print(_BANNER)
    sources = pipeline.list_sources()
    console.print(
        f"[dim]{len(sources)} document(s) indexed | "
        f"model: {pipeline.settings.model_filename.replace('.gguf', '')}[/dim]"
    )
    if source_filter:
        console.print(f"[dim]Filtered to: {source_filter}[/dim]")

    # Each entry is a full injected message dict (role + content with sources).
    # Storing the complete content — not just the bare question — ensures the
    # LLM can see the supporting context from prior turns when generating
    # follow-up answers, preventing groundless references to earlier answers.
    chat_history: list[dict[str, str]] = []

    while True:
        try:
            question = Prompt.ask("\n[bold cyan]You[/bold cyan]")
        except (KeyboardInterrupt, EOFError):
            console.print("\n[dim]Goodbye.[/dim]")
            break

        question = question.strip()
        if not question:
            continue

        if question.lower() in {"/quit", "/exit", "/q"}:
            console.print("[dim]Goodbye.[/dim]")
            break

        if question.lower() == "/sources":
            srcs = pipeline.list_sources()
            if srcs:
                t = Table(title="Indexed Documents", show_header=False)
                t.add_column("Source")
                for s in srcs:
                    t.add_row(s)
                console.print(t)
            else:
                console.print("[yellow]No documents indexed.[/yellow]")
            continue

        if question.lower() == "/reset":
            chat_history.clear()
            console.print("[dim]Chat history cleared.[/dim]")
            continue

        try:
            with console.status("[dim]Thinking…[/dim]"):
                result = pipeline.query(
                    question,
                    chat_history=chat_history if use_history else None,
                    source_filter=source_filter,
                )
        except Exception as exc:
            console.print(f"[red]Error:[/red] {exc}")
            continue

        console.print(Panel(
            Markdown(result.answer),
            title="[bold green]Assistant[/bold green]",
            border_style="green",
        ))

        if result.sources:
            _print_sources(result.sources)

        r = result.retrieval
        console.print(
            f"[dim]best L2: {r.best_score:.4f} | threshold: {r.threshold:.2f} "
            f"| chunks used: {len(r.filtered)}"
            + (" | reranked" if r.reranked else "")
            + "[/dim]"
        )

        if use_history:
            # Store the full injected user message (with source context) so
            # future turns have the grounding material, not just the bare question.
            injected_user_content = (
                f"Sources:\n{result.injected_context}\n\nQuestion: {question}"
                if result.injected_context
                else question
            )
            chat_history.append({"role": "user", "content": injected_user_content})
            chat_history.append({"role": "assistant", "content": result.answer})


def _print_sources(sources: list[dict]) -> None:
    t = Table(show_header=True, header_style="bold", show_lines=True)
    t.add_column("#", style="dim", width=3)
    t.add_column("Source")
    t.add_column("Page", justify="center")
    t.add_column("Match", justify="center")
    t.add_column("Excerpt")
    for i, src in enumerate(sources, 1):
        score = src["score"]
        # Correct cosine similarity from L2 distance on normalized embeddings:
        #   cos_sim = 1 - (L2² / 2)
        # The old formula `1 - L2` understated similarity by up to 50 percentage
        # points at mid-range distances (e.g., L2=0.5 → was shown as 50%, actual 87%).
        if src.get("reranked"):
            # After reranking, score is a cross-encoder logit (higher = better).
            # Sigmoid maps it to a 0–100% probability-like display value.
            import math
            display_pct = 1.0 / (1.0 + math.exp(-score))
        else:
            cos_sim = max(0.0, 1.0 - score ** 2 / 2.0)
            display_pct = cos_sim
        excerpt = src["text"][:120].replace("\n", " ")
        if len(src["text"]) > 120:
            excerpt += "…"
        t.add_row(str(i), src["source"], str(src["page"]), f"{display_pct:.0%}", excerpt)
    console.print(t)


if __name__ == "__main__":
    main()
