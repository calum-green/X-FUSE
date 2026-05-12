# tests/test_loaders.py
from unittest.mock import MagicMock, patch
import numpy as np
import pytest

from x_fuse.loaders import load_xct


def _make_mock_h5(data: np.ndarray):
    """Return a mock h5py File whose ['key']['data'] returns data."""
    mock_ds = MagicMock()
    mock_ds.__getitem__ = MagicMock(return_value=data)
    mock_file = MagicMock()
    mock_file.__enter__ = MagicMock(return_value=mock_file)
    mock_file.__exit__ = MagicMock(return_value=False)
    mock_file.__getitem__ = MagicMock(return_value=mock_ds)
    return mock_file


@patch("x_fuse.loaders.h5py.File")
def test_load_xct_tomopy(mock_h5):
    data = np.zeros((10, 20, 30))  # (Z, Y, X)
    mock_h5.return_value = _make_mock_h5(data)
    result = load_xct("fake.h5", recon="Tomopy")
    assert result.shape == (20, 30, 10)  # transposed to (Y, X, Z)


@patch("x_fuse.loaders.h5py.File")
def test_load_xct_astra(mock_h5):
    data = np.zeros((10, 20, 30))
    mock_h5.return_value = _make_mock_h5(data)
    result = load_xct("fake.h5", recon="Astra")
    assert result.shape == (20, 30, 10)


def test_load_xct_rejects_non_h5():
    with pytest.raises(AssertionError):
        load_xct("fake.nxs")
