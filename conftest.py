"""Root conftest — adds src/ to sys.path so `import rag` works without pip install."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))
