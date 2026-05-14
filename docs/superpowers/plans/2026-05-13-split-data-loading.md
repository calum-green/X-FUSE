# Split Raw Data Loading — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Expose `load_raw_data()` and `run_data_from_raw()` to allow users to load full XCT/XRDCT volumes once and iterate over slices without reloading.

**Architecture:** Extract the loading phase (`_set_cache_env` + `_validate_paths` + `_load_raw_data`) into public `load_raw_data()`, extract the processing phase (slice extraction through save) into public `run_data_from_raw()`, and refactor `run_data()` to call both. Backward compatibility maintained.

**Tech Stack:** NumPy, pytest, existing pipeline helpers.

---

## File Structure

**Modify:**
- `x_fuse/pipeline.py` — add `load_raw_data()`, add `run_data_from_raw()`, refactor `run_data()`
- `tests/test_pipeline.py` — add `test_load_raw_data_returns_correct_shapes()`, add `test_run_data_from_raw_matches_run_data()`

---

## Task 1: Test and Implement `load_raw_data()`

**Files:**
- Modify: `x_fuse/pipeline.py:25-56`
- Test: `tests/test_pipeline.py`

- [ ] **Step 1: Write failing test for `load_raw_data`**

Add to `tests/test_pipeline.py`:

```python
def test_load_raw_data_returns_correct_types_and_shapes(diad_config):
    """Verify load_raw_data returns (xct_raw, xrd_raw) with correct structure."""
    from x_fuse.pipeline import load_raw_data
    
    xct_raw, xrd_raw = load_raw_data(diad_config)
    
    # Type checks
    assert isinstance(xct_raw, np.ndarray), f"xct_raw should be ndarray, got {type(xct_raw)}"
    assert isinstance(xrd_raw, dict), f"xrd_raw should be dict, got {type(xrd_raw)}"
    
    # Shape checks — XCT should be 3D (slices, H, W)
    assert xct_raw.ndim == 3, f"xct_raw should be 3D, got {xct_raw.ndim}D with shape {xct_raw.shape}"
    
    # XRDCT phases should match config
    assert set(xrd_raw.keys()) == set(diad_config.phases), \
        f"xrd_raw phases {set(xrd_raw.keys())} don't match config phases {set(diad_config.phases)}"
    
    # Each XRDCT phase should be 3D
    for phase, arr in xrd_raw.items():
        assert isinstance(arr, np.ndarray), f"xrd_raw[{phase}] should be ndarray"
        assert arr.ndim == 3, f"xrd_raw[{phase}] should be 3D, got {arr.ndim}D with shape {arr.shape}"
```

- [ ] **Step 2: Run test to verify it fails**

```bash
cd /Users/trb56939/Documents/PhD\ Year\ 1/XRD_Fusion/X-FUSE
pytest tests/test_pipeline.py::test_load_raw_data_returns_correct_types_and_shapes -xvs
```

Expected: FAIL with `ImportError: cannot import name 'load_raw_data'` or `AttributeError: module has no attribute 'load_raw_data'`

- [ ] **Step 3: Implement `load_raw_data()` in pipeline.py**

Add this new function before the current `run_data()` (around line 25):

```python
def load_raw_data(config: XFuseConfig) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """Load full XCT and XRDCT volumes from disk (no slice extraction).
    
    This is the slow step (~10 min for large files). Load once, then use
    run_data_from_raw() multiple times with different slice indices.
    
    Args:
        config: XFuseConfig with paths and dataset settings
        
    Returns:
        Tuple of (xct_raw, xrd_raw) where:
        - xct_raw: 3D numpy array, shape determined by loader (typically N_slices, H, W)
        - xrd_raw: dict mapping phase names to 3D arrays of same structure
    """
    print(f"\n[X-FUSE] Loading raw data  ({config.name})")
    _set_cache_env(config)
    print("  validating paths...")
    _validate_paths(config)
    print("  loading full volumes...")
    xct_raw, xrd_raw = _load_raw_data(config)
    print(f"  XCT shape: {xct_raw.shape}, XRDCT shapes: {{{', '.join(f'{p}: {arr.shape}' for p, arr in xrd_raw.items())}}}")
    print("  done.")
    return xct_raw, xrd_raw
```

- [ ] **Step 4: Run test to verify it passes**

```bash
pytest tests/test_pipeline.py::test_load_raw_data_returns_correct_types_and_shapes -xvs
```

Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add x_fuse/pipeline.py tests/test_pipeline.py
git commit -m "feat: add load_raw_data() to expose full-volume loading"
```

---

## Task 2: Test and Implement `run_data_from_raw()`

**Files:**
- Modify: `x_fuse/pipeline.py`
- Test: `tests/test_pipeline.py`

- [ ] **Step 1: Write failing test for `run_data_from_raw`**

Add to `tests/test_pipeline.py`:

```python
def test_run_data_from_raw_produces_same_outputs_as_run_data(diad_config, tmp_path):
    """Verify run_data_from_raw() produces identical outputs to run_data()."""
    from x_fuse.pipeline import load_raw_data, run_data_from_raw, run_data
    
    # Load raw data once
    xct_raw, xrd_raw = load_raw_data(diad_config)
    
    # Run via refactored path (load_raw_data + run_data_from_raw)
    config1 = diad_config.replace(output_dir=str(tmp_path / "out1"))
    run_data_from_raw(xct_raw, xrd_raw, config1)
    
    # Run via direct path (full run_data)
    config2 = diad_config.replace(output_dir=str(tmp_path / "out2"))
    run_data(config2)
    
    # Compare outputs: xct.npy should be identical
    out1_xct = np.load(tmp_path / "out1" / diad_config.name / "data" / "xct.npy")
    out2_xct = np.load(tmp_path / "out2" / diad_config.name / "data" / "xct.npy")
    
    np.testing.assert_array_equal(out1_xct, out2_xct, 
        err_msg="xct.npy outputs differ between run_data_from_raw and run_data")
    
    # Compare XRDCT outputs
    for phase in diad_config.phases:
        fname = f"{phase}_xrd.npy"
        arr1 = np.load(tmp_path / "out1" / diad_config.name / "data" / fname)
        arr2 = np.load(tmp_path / "out2" / diad_config.name / "data" / fname)
        np.testing.assert_array_equal(arr1, arr2,
            err_msg=f"{fname} outputs differ between run_data_from_raw and run_data")
```

- [ ] **Step 2: Run test to verify it fails**

```bash
pytest tests/test_pipeline.py::test_run_data_from_raw_produces_same_outputs_as_run_data -xvs
```

Expected: FAIL with `ImportError: cannot import name 'run_data_from_raw'`

- [ ] **Step 3: Implement `run_data_from_raw()` in pipeline.py**

Add this new function after `load_raw_data()`:

```python
def run_data_from_raw(
    xct_raw: np.ndarray, 
    xrd_raw: dict[str, np.ndarray], 
    config: XFuseConfig
) -> None:
    """Process pre-loaded raw XCT/XRDCT volumes: extract slice, transform, save.
    
    This is the fast step (<1 min). Call after load_raw_data() with different
    slice indices via config.xct_sample_idx and config.xrdct_sample_idx.
    
    Args:
        xct_raw: Full 3D XCT volume (from load_raw_data)
        xrd_raw: Dict of full 3D XRDCT volumes (from load_raw_data)
        config: XFuseConfig with slice indices and output settings
    """
    print(f"\n[X-FUSE] Stage 0 - data  ({config.name})")
    print(
        f"  extracting sample (XCT idx={config.xct_sample_idx},"
        f" XRDCT idx={config.xrdct_sample_idx})..."
    )
    xct_sample, xrd_sample = _extract_sample(xct_raw, xrd_raw, config)
    if config.invert:
        print("  inverting images...")
        xct_sample = invert_image(xct_sample)
        xrd_sample = {p: invert_image(a) for p, a in xrd_sample.items()}
    _validate_arrays(xct_sample, xrd_sample)
    img_size = _snap_to_multiple_of_16(xct_sample.shape[0])
    print(f"  effective img_size: {img_size}")
    print("  transforming XCT image...")
    img_tr = tr.get_input_transform(img_size, img_size)
    xct_transformed = _apply_xct_transform(xct_sample, img_tr)
    out_dir = config.output_path / "data"
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"  saving to {out_dir}/")
    np.save(out_dir / "xct.npy", xct_transformed.astype(np.float32))
    for phase, arr in xrd_sample.items():
        np.save(out_dir / f"{phase}_xrd.npy", arr.astype(np.float32))
    _write_data_summary(out_dir, xct_transformed, xrd_sample, config, img_size)
    if config.vis_data:
        _save_data_overview(out_dir, xct_transformed, xrd_sample)
    print("  done.")
```

- [ ] **Step 4: Run test to verify it passes**

```bash
pytest tests/test_pipeline.py::test_run_data_from_raw_produces_same_outputs_as_run_data -xvs
```

Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add x_fuse/pipeline.py tests/test_pipeline.py
git commit -m "feat: add run_data_from_raw() to process pre-loaded volumes"
```

---

## Task 3: Refactor `run_data()` to use both new functions

**Files:**
- Modify: `x_fuse/pipeline.py:25-56` (current `run_data` function)

- [ ] **Step 1: Refactor `run_data()` to call `load_raw_data()` and `run_data_from_raw()`**

Replace the current `run_data()` function (lines 25–56) with:

```python
def run_data(config: XFuseConfig) -> None:
    """Validate, load, format, and save input data. CPU-only.
    
    High-level wrapper that combines load_raw_data() and run_data_from_raw().
    Use this when you don't need to iterate over multiple slices.
    
    For repeated iteration with different slice indices, call load_raw_data()
    once, then run_data_from_raw() multiple times.
    """
    xct_raw, xrd_raw = load_raw_data(config)
    run_data_from_raw(xct_raw, xrd_raw, config)
```

- [ ] **Step 2: Run full test suite to verify backward compatibility**

```bash
pytest tests/test_pipeline.py -v
```

Expected: All tests pass, including:
- Existing `test_run_data_*` tests
- New `test_load_raw_data_returns_correct_types_and_shapes`
- New `test_run_data_from_raw_produces_same_outputs_as_run_data`

- [ ] **Step 3: Commit**

```bash
git add x_fuse/pipeline.py
git commit -m "refactor: run_data() now calls load_raw_data() and run_data_from_raw()"
```

---

## Task 4: Verify all tests pass

**Files:**
- Test: `tests/test_pipeline.py`

- [ ] **Step 1: Run full test suite**

```bash
pytest tests/test_pipeline.py -v --tb=short
```

Expected: All tests pass (old + new)

- [ ] **Step 2: Run the notebook workflow manually (optional but recommended)**

In a Python notebook or interactive session:

```python
from x_fuse.config import XFuseConfig
from x_fuse.pipeline import load_raw_data, run_data_from_raw

# Load once
config = XFuseConfig.from_yaml("configs/example_diad.yaml")
xct_raw, xrd_raw = load_raw_data(config)

# Iterate with different slices
for idx in [0, -1]:
    cfg = config.replace(xct_sample_idx=idx)
    run_data_from_raw(xct_raw, xrd_raw, cfg)
    print(f"Processed slice {idx}")
```

Expected: Both slices process without reloading (~1 min each)

- [ ] **Step 3: Final commit (if any cleanup needed)**

```bash
git status
```

If any changes remain, commit them:

```bash
git add -A
git commit -m "test: verify split-loading workflow end-to-end"
```

Otherwise, you're done!

---

## Summary

This plan adds two public functions (`load_raw_data`, `run_data_from_raw`) and refactors one (`run_data`) to preserve backward compatibility while enabling the fast-iteration workflow. All changes are in `x_fuse/pipeline.py` and `tests/test_pipeline.py`. TDD workflow ensures correctness at each step.
