"""Thin wrapper — kept for backward compatibility with `streamlit run app/main.py`.

The real implementation lives in src/rag/app/main.py.
After `pip install -e .`, use the `rag-app` console script instead.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from rag.app.main import main  # noqa: E402

if __name__ == "__main__":
    main()
