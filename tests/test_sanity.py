"""Basic sanity test to verify the test infrastructure works."""


def test_environment_imports():
    """Verify core packages are importable."""
    import numpy
    import pandas
    import sklearn

    assert numpy is not None
    assert pandas is not None
    assert sklearn is not None


def test_numpy_basic():
    """Verify NumPy works."""
    import numpy as np

    a = np.array([1, 2, 3])
    assert a.sum() == 6
    assert a.shape == (3,)
