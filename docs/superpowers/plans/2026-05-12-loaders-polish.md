# Loaders Polish Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Polish `x_fuse/loaders.py` — fix three bugs in the existing functions, remove dead code, and tighten type annotations. No functions are rewritten from scratch.

**Architecture:** Two-layer flat file: generic loaders (`load_xct`, `load_xrdct_phase`) at the top, dataset-specific wrappers below. Notebooks import wrappers directly via `from x_fuse.loaders import <function>`.

**Tech Stack:** Python 3.10+, numpy, h5py, pytest, unittest.mock

---

## Files

- Modify: `x_fuse/loaders.py` — fix bugs, remove dead code, tighten annotations
- Modify: `tests/test_loaders.py` — new file with unit tests for the fixed behaviour (uses mocks, no real data files needed)

---

### Task 1: Fix `load_xct` — broken `recon_methods` lookup and unclosed file

**Bug:** `recon_methods` is a `list[dict]` but is accessed as `recon_methods[recon]` where `recon` is a string (e.g. `"Astra"`). This raises `TypeError` at runtime. Additionally, `h5py.File` is opened without a context manager so the file handle is never closed. The `crop` parameter is accepted but never used — remove it.

**Files:**
- Modify: `x_fuse/loaders.py`
- Test: `tests/test_loaders.py`

- [ ] **Step 1: Create `tests/test_loaders.py` and write a failing test for `load_xct`**

```python
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
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pytest tests/test_loaders.py -v
```

Expected: FAIL — `TypeError: list indices must be integers or slices, not str`

- [ ] **Step 3: Fix `load_xct` in `x_fuse/loaders.py`**

Replace the function with the corrected version (keep the same signature minus unused `crop`):

```python
def load_xct(xct_path: str, recon: str = "Tomopy", **kwargs) -> np.ndarray:
    assert xct_path.endswith(".h5"), "Expected a .h5 file for the xct data."

    recon_methods = {
        "Tomopy": "4-TomopyRecon-tomo",
        "Astra": "4-AstraReconGpu-tomo",
    }

    with h5py.File(xct_path, "r") as f:
        return np.transpose(np.array(f[recon_methods[recon]]["data"]), (1, 2, 0))
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
pytest tests/test_loaders.py::test_load_xct_tomopy tests/test_loaders.py::test_load_xct_astra tests/test_loaders.py::test_load_xct_rejects_non_h5 -v
```

Expected: all three PASS

- [ ] **Step 5: Commit**

```bash
git add x_fuse/loaders.py tests/test_loaders.py
git commit -m "fix: correct recon_methods lookup and close h5py file in load_xct"
```

---

### Task 2: Fix `load_xrdct_phase` — `append` inside inner loop and no-op transpose

**Bug 1:** `xrdct_data.append(xrdct_i)` is inside the inner `for idx` loop, so it appends a reference to `xrdct_i` once per file rather than once per phase. It should run after the inner loop ends.

**Bug 2:** `np.transpose(np.array(f["data"][:]), (0, 1, 2))` is a no-op (identity permutation) and can be removed.

**Files:**
- Modify: `x_fuse/loaders.py`
- Test: `tests/test_loaders.py`

- [ ] **Step 1: Add a failing test for `load_xrdct_phase`**

Append to `tests/test_loaders.py`:

```python
from x_fuse.loaders import load_xrdct_phase


def _mock_listdir(phase: str, n_files: int):
    return [f"{phase}_file_{i:02d}.nxs" for i in range(n_files)]


@patch("x_fuse.loaders.os.listdir")
@patch("x_fuse.loaders.h5py.File")
def test_load_xrdct_phase_returns_one_array_per_phase(mock_h5, mock_ls):
    phases = ["ZnO", "Zn13X"]
    shape = (3, 4, 4)

    mock_ls.return_value = _mock_listdir("ZnO", 3) + _mock_listdir("Zn13X", 3)

    mock_ds = MagicMock()
    mock_ds.__getitem__ = MagicMock(return_value=np.zeros((1, 4, 4)))
    mock_file = MagicMock()
    mock_file.__enter__ = MagicMock(return_value=mock_file)
    mock_file.__exit__ = MagicMock(return_value=False)
    mock_file.__getitem__ = MagicMock(return_value=mock_ds)
    mock_h5.return_value = mock_file

    result = load_xrdct_phase("fake_folder", phases=phases, shape=shape)
    assert len(result) == len(phases), f"Expected {len(phases)} arrays, got {len(result)}"
```

- [ ] **Step 2: Run test to verify it fails**

```bash
pytest tests/test_loaders.py::test_load_xrdct_phase_returns_one_array_per_phase -v
```

Expected: FAIL — `AssertionError: Expected 2 arrays, got 6` (or similar, due to append inside inner loop)

- [ ] **Step 3: Fix `load_xrdct_phase` in `x_fuse/loaders.py`**

Move `xrdct_data.append(xrdct_i)` to after the inner loop, and remove the no-op transpose:

```python
def load_xrdct_phase(
    phase_folder: str, phases: list, shape: list, crop: slice = slice(None), **kwargs
) -> list[np.ndarray]:
    xrdct_data = []
    for i in range(len(phases)):
        files = [f for f in os.listdir(phase_folder) if phases[i] in f][crop]

        assert len(files) > 0, f"No files found for phase {phases[i]} in folder {phase_folder}."
        assert any(f.endswith(".nxs") for f in files), f"Expected .nxs files for phase {phases[i]}, but found none."
        assert len(files) == shape[0], f"Expected {shape[0]} files for phase {phases[i]}, but found {len(files)}."

        xrdct_i = np.zeros(shape)
        for idx in range(shape[0]):
            with h5py.File(os.path.join(phase_folder, files[idx]), "r") as f:
                xrdct_i[idx] += np.array(f["data"][:]).squeeze()
        xrdct_data.append(xrdct_i)

    return xrdct_data
```

- [ ] **Step 4: Run tests**

```bash
pytest tests/test_loaders.py -v
```

Expected: all tests PASS

- [ ] **Step 5: Commit**

```bash
git add x_fuse/loaders.py tests/test_loaders.py
git commit -m "fix: append per phase not per file in load_xrdct_phase, remove no-op transpose"
```

---

### Task 3: Fix `load_diad_xrdct` — list indexed by string

**Bug:** `load_xrdct_phase` returns a `list[np.ndarray]` indexed by integer, but `load_diad_xrdct` does `xrdct_phases[phase]` where `phase` is a string. This raises `TypeError` at runtime. Fix by iterating with `enumerate` and return a `dict[str, np.ndarray]` keyed by phase name for convenient downstream access.

**Files:**
- Modify: `x_fuse/loaders.py`
- Test: `tests/test_loaders.py`

- [ ] **Step 1: Add a failing test for `load_diad_xrdct`**

Append to `tests/test_loaders.py`:

```python
from x_fuse.loaders import load_diad_xrdct


@patch("x_fuse.loaders.load_xrdct_phase")
def test_load_diad_xrdct_returns_dict_keyed_by_phase(mock_load):
    phases = ["ZnO", "Zn13X"]
    mock_load.return_value = [np.ones((15, 20, 20)), np.ones((15, 20, 20)) * 2]

    result = load_diad_xrdct("fake_folder", phases=phases)

    assert isinstance(result, dict)
    assert set(result.keys()) == set(phases)
    assert result["ZnO"].shape == (15, 20, 20)
```

- [ ] **Step 2: Run test to verify it fails**

```bash
pytest tests/test_loaders.py::test_load_diad_xrdct_returns_dict_keyed_by_phase -v
```

Expected: FAIL — `TypeError: list indices must be integers or slices, not str`

- [ ] **Step 3: Fix `load_diad_xrdct` in `x_fuse/loaders.py`**

```python
def load_diad_xrdct(phase_folder: str, phases: list, **kwargs) -> dict[str, np.ndarray]:
    phase_arrays = load_xrdct_phase(
        phase_folder, shape=(21, 20, 20), phases=phases, crop=slice(5, -1), **kwargs
    )

    return {
        phase: np.flip(arr, axis=2).copy()
        for phase, arr in zip(phases, phase_arrays)
    }
```

- [ ] **Step 4: Run all tests**

```bash
pytest tests/test_loaders.py -v
```

Expected: all tests PASS

- [ ] **Step 5: Commit**

```bash
git add x_fuse/loaders.py tests/test_loaders.py
git commit -m "fix: index load_diad_xrdct result by position not phase name, return dict"
```

---

### Task 4: Final polish — type annotations and tidy `load_i13_xct`

Tighten return type annotations (`np.array` → `np.ndarray`) across all functions and ensure `load_i13_xct` has a consistent signature matching the other wrappers.

**Files:**
- Modify: `x_fuse/loaders.py`

- [ ] **Step 1: Update `load_i13_xct` signature and annotation**

Replace:

```python
def load_i13_xct(xct_path, **kwargs) -> np.array:

    return load_xct(xct_path, recon="Tomopy")
```

With:

```python
def load_i13_xct(xct_path: str, **kwargs) -> np.ndarray:
    return load_xct(xct_path, recon="Tomopy")
```

Also update `load_diad_xct_zn13x`:

```python
def load_diad_xct_zn13x(xct_path: str, **kwargs) -> np.ndarray:
    return load_xct(xct_path, recon="Astra")[150:2000, 350:2200, 350:2200]
```

- [ ] **Step 2: Run all tests to confirm nothing broke**

```bash
pytest tests/test_loaders.py -v
```

Expected: all PASS

- [ ] **Step 3: Commit**

```bash
git add x_fuse/loaders.py
git commit -m "chore: tighten type annotations in loaders.py"
```
