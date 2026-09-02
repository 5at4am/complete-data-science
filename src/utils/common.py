"""Shared utilities: seeds, timing, and model/dataset persistence.

Everything here is dependency-light (numpy / joblib only) so it can be
imported from any notebook cheaply.
"""

from __future__ import annotations

import contextlib
import os
import time
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any

_PROJECT_ROOT = Path(__file__).resolve().parents[2]


def project_root() -> Path:
    """Return the absolute path of the repository root."""
    return _PROJECT_ROOT


def set_seed(seed: int = 42) -> None:
    """Seed numpy, Python's random, and scikit-learn for reproducibility."""
    import random

    import numpy as np

    np.random.seed(seed)
    random.seed(seed)
    with contextlib.suppress(ImportError):
        from sklearn.utils import check_random_state

        check_random_state(seed)


@contextlib.contextmanager
def timing(name: str = "task") -> Iterator[Callable[[], float]]:
    """Time a block of code.

    Example:
        with timing("training") as elapsed:
            model.fit(X, y)
        print(f"took {elapsed()}s")
    """
    start = time.perf_counter()

    def _elapsed() -> float:
        return round(time.perf_counter() - start, 3)

    try:
        yield _elapsed
    finally:
        print(f"{name} took {_elapsed()}s")


def save_object(obj: Any, path: str | os.PathLike) -> str:
    """Persist any picklable object with joblib.

    Creates parent directories as needed and returns the resolved path.
    """
    import joblib

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(obj, path)
    print(f"Saved to: {path} ({path.stat().st_size / 1024:.1f} KB)")
    return str(path)


def load_object(path: str | os.PathLike) -> Any:
    """Load an object previously persisted with :func:`save_object`."""
    import joblib

    return joblib.load(path)
