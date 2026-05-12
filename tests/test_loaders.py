# tests/test_loaders.py
from unittest.mock import MagicMock, patch
import numpy as np
import pytest

from x_fuse.loaders import load_diad_xrdct, load_xct, load_xrdct_phase


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


# ── load_xrdct_phase ──────────────────────────────────────────────────────────


def _mock_listdir(phase: str, n_files: int):
    return [f"{phase}_file_{i:02d}.nxs" for i in range(n_files)]


@patch("x_fuse.loaders.os.listdir")
@patch("x_fuse.loaders.h5py.File")
def test_load_xrdct_phase_returns_one_array_per_phase(mock_h5, mock_ls):
    phases = ["Zn", "Na"]
    shape = (3, 4, 4)
    entry_names = {"Zn": "entry/peak at q~1.656", "Na": "entry/peak at q~1.651"}

    mock_ls.return_value = _mock_listdir("Zn", 3) + _mock_listdir("Na", 3)

    mock_file = MagicMock()
    mock_file.__enter__ = MagicMock(return_value=mock_file)
    mock_file.__exit__ = MagicMock(return_value=False)
    mock_file.__getitem__ = MagicMock(return_value=np.zeros((1, 4, 4)))
    mock_h5.return_value = mock_file

    result = load_xrdct_phase(
        "fake_folder", phases=phases, shape=shape, entry_names=entry_names
    )
    assert len(result) == len(
        phases
    ), f"Expected {len(phases)} arrays, got {len(result)}"


# ── load_diad_xrdct ───────────────────────────────────────────────────────────


@patch("x_fuse.loaders.load_xrdct_phase")
def test_load_diad_xrdct_returns_dict_keyed_by_phase(mock_load):
    phases = ["Zn", "Na"]
    mock_load.return_value = [np.ones((15, 20, 20)), np.ones((15, 20, 20)) * 2]

    result = load_diad_xrdct("fake_folder", phases=phases)

    assert isinstance(result, dict)
    assert set(result.keys()) == set(phases)
    assert result["Zn"].shape == (15, 20, 20)
