"""Thin wrapper — kept for backward compatibility with `python scripts/query.py`.

The real implementation lives in src/rag/cli/query.py.
After `pip install -e .`, use the `rag-query` console script instead.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from rag.cli.query import main  # noqa: E402

if __name__ == "__main__":
    main()
