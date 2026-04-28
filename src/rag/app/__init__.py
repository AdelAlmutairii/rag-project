"""Streamlit web UI — launchable as a console script entry point."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def launch() -> None:
    """Entry point for the `rag-app` console script."""
    app_file = Path(__file__).parent / "main.py"
    raise SystemExit(
        subprocess.call(
            ["streamlit", "run", str(app_file), "--"] + sys.argv[1:],
        )
    )
