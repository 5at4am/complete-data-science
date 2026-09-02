"""Tests for src.utils.common and src.utils."""

import os


def test_project_root_points_at_repo():
    from src.utils.common import project_root

    root = project_root()
    assert (root / "README.md").exists()
    assert (root / "src").exists()


def test_set_seed_runs_and_is_reproducible():
    from src.utils.common import set_seed

    import numpy as np

    set_seed(42)
    a = np.random.rand(5)
    set_seed(42)
    b = np.random.rand(5)
    assert (a == b).all()


def test_save_and_load_roundtrip(tmp_path):
    from src.utils.common import load_object, save_object

    payload = {"models": ["lr", "rf"], "best": 0.9}
    path = save_object(payload, str(tmp_path / "nested" / "model.joblib"))
    assert os.path.exists(path)  # parent was created automatically
    assert load_object(path) == payload