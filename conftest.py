"""Ensure the repo root is importable so tests can import ``toolgym`` and
``examples`` packages regardless of where pytest is invoked from.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
