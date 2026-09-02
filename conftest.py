"""Ensure the repository root is importable so ``import src.*`` works in tests.

Without this, ``pytest`` (invoked as ``pytest`` rather than ``python -m pytest``)
would not find the ``src`` package because it is not pip-installed.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))