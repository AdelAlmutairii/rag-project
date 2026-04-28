#!/usr/bin/env python3
"""CLI: batch-ingest documents from a directory into the vector store.

After `pip install`, this is available as the `rag-ingest` command.

Usage:
    rag-ingest
    rag-ingest --dir path/to/pdfs
    rag-ingest --file path/to/single.pdf
    rag-ingest --reset --dir path/to/pdfs
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from rich.console import Console
from rich.panel import Panel
from rich.progress import BarColumn, MofNCompleteColumn, Progress, TextColumn, TimeElapsedColumn
from rich.table import Table

from rag.config import Settings
from rag.ingest import ingest_directory, ingest_file
from rag.vectorstore import VectorStore

console = Console()
logging.basicConfig(level=logging.WARNING)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Ingest documents into the RAG vector store.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--dir", type=Path, default=None, metavar="PATH",
                        help="Directory of documents to ingest")
    source.add_argument("--file", type=Path, default=None, metavar="FILE",
                        help="Single file to ingest")
    parser.add_argument("--reset", action="store_true",
                        help="Wipe the existing vector store before ingesting")
    parser.add_argument("--chunk-size", type=int, default=None)
    parser.add_argument("--chunk-overlap", type=int, default=None)
    parser.add_argument("--device", choices=["cpu", "cuda", "mps"], default=None,
                        help="Device for the embedding model")
    parser.add_argument("--verbose", "-v", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.verbose:
        logging.basicConfig(level=logging.INFO, force=True)

    try:
        settings = Settings()
    except Exception as exc:
        console.print(f"[red]Configuration error:[/red] {exc}")
        console.print("Create a [bold].env[/bold] file in the current directory.")
        sys.exit(1)

    if args.chunk_size is not None:
        settings.chunk_size = args.chunk_size
    if args.chunk_overlap is not None:
        settings.chunk_overlap = args.chunk_overlap
    if args.device is not None:
        settings.embedding_device = args.device

    vectorstore = VectorStore(settings)

    if args.reset:
        console.print("[yellow]Resetting vector store…[/yellow]")
        vectorstore.reset()

    if args.file:
        _ingest_single(args.file, settings, vectorstore)
    else:
        _ingest_directory(args.dir or settings.documents_dir, settings, vectorstore)


def _ingest_single(file_path: Path, settings: Settings, vectorstore: VectorStore) -> None:
    if not file_path.exists():
        console.print(f"[red]File not found:[/red] {file_path}")
        sys.exit(1)

    console.print(f"[bold]Ingesting[/bold] {file_path.name}…")
    result = ingest_file(file_path, settings)

    if result.errors:
        for fname, err in result.errors.items():
            console.print(f"[red]{fname}:[/red] {err}")
        sys.exit(1)

    if not result.chunks:
        console.print("[yellow]No content extracted.[/yellow]")
        return

    with console.status("Embedding and storing…"):
        vectorstore.add_documents(result.chunks)

    console.print(Panel(
        f"[green]✔ {result.chunk_count} chunks from {file_path.name}[/green]\n"
        f"Total in store: [bold]{vectorstore.count()}[/bold]",
        title="Done",
    ))


def _ingest_directory(directory: Path, settings: Settings, vectorstore: VectorStore) -> None:
    if not directory.exists():
        console.print(f"[red]Directory not found:[/red] {directory}")
        sys.exit(1)

    console.print(f"[bold]Scanning[/bold] {directory.resolve()}…")

    with console.status("Loading documents…"):
        result = ingest_directory(directory, settings)

    table = Table(title="Ingestion Summary", show_header=True, header_style="bold cyan")
    table.add_column("Metric", style="dim")
    table.add_column("Value", justify="right")
    table.add_row("Documents loaded", str(result.loaded_count))
    table.add_row("Chunks created", str(result.chunk_count))
    table.add_row("Skipped (unsupported)", str(len(result.skipped)))
    table.add_row("Errors", str(len(result.errors)))
    console.print(table)

    if result.skipped:
        console.print(f"[dim]Skipped:[/dim] {', '.join(result.skipped)}")
    for fname, err in result.errors.items():
        console.print(f"[red]{fname}:[/red] {err}")

    if not result.chunks:
        console.print("[yellow]No chunks to store.[/yellow]")
        return

    with Progress(
        TextColumn("[progress.description]{task.description}"),
        BarColumn(), MofNCompleteColumn(), TimeElapsedColumn(),
        console=console,
    ) as progress:
        task = progress.add_task("Embedding and storing…", total=1)
        vectorstore.add_documents(result.chunks)
        progress.update(task, advance=1)

    console.print(Panel(
        f"[green]✔ {result.chunk_count} chunks stored.[/green]\n"
        f"Total in store: [bold]{vectorstore.count()}[/bold]",
        title="Done",
    ))


if __name__ == "__main__":
    main()
