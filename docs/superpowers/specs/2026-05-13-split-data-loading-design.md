# Split Raw Data Loading from Pipeline — Design Spec

**Goal:** Allow users to load full XCT/XRDCT volumes once (10 min) and iterate over different slice indices without reloading.

**Architecture:** Expose `load_raw_data()` to return pre-loaded raw arrays, and `run_data_from_raw()` to process them. Refactor `run_data()` to use both, preserving backward compatibility.

**Tech Stack:** NumPy arrays, existing `_load_raw_data` and `_extract_sample` helpers.

---

## API Additions

### `load_raw_data(config: XFuseConfig) -> tuple[np.ndarray, dict[str, np.ndarray]]`

**Responsibility:** Load and return full XCT and XRDCT volumes from disk.

**Behavior:**
- Sets torch cache environment variables via `_set_cache_env(config)`
- Validates file paths via `_validate_paths(config)`
- Calls `_load_raw_data(config)` to load the full 3D volumes
- Returns tuple `(xct_raw, xrd_raw)` where:
  - `xct_raw`: numpy array, shape `(N_slices, H, W)` or similar (loader-dependent)
  - `xrd_raw`: dict mapping phase names (e.g. "Na", "Zn") to numpy arrays of the same structure
- Prints all loading progress (inherited from `_load_raw_data`)

**Usage:**
```python
xct_raw, xrd_raw = load_raw_data(config)  # ~10 min for large files
```

**Error handling:**
- Propagates `FileNotFoundError` from `_validate_paths` if paths don't exist
- Propagates `ValueError` from `_load_raw_data` if file format is invalid

---

### `run_data_from_raw(xct_raw: np.ndarray, xrd_raw: dict[str, np.ndarray], config: XFuseConfig) -> None`

**Responsibility:** Process pre-loaded raw data through slice extraction, transform, and save.

**Behavior:**
1. Extract single-slice samples via `_extract_sample(xct_raw, xrd_raw, config)`
   - Uses `config.xct_sample_idx` and `config.xrdct_sample_idx` to pick slices
   - Returns 2D arrays: `xct_sample, xrd_sample`
2. Invert images if `config.invert=True`
3. Validate array dimensions via `_validate_arrays()`
4. Compute effective `img_size` by snapping XCT shape[0] to nearest multiple of 16
5. Create output directory `{config.output_dir}/{config.name}/data/`
6. Apply transform and save:
   - `xct.npy` — transformed XCT slice
   - `{phase}_xrd.npy` — XRDCT slices for each phase
   - `data_summary.txt` — metadata (shapes, value ranges, indices)
7. If `config.vis_data=True`, save data overview visualization

**Usage:**
```python
config = config.replace(xct_sample_idx=5)  # change slice index
run_data_from_raw(xct_raw, xrd_raw, config)
run_features(config)  # continues with the new slice
```

**Error handling:**
- Propagates dimension errors from `_validate_arrays` if slice extraction fails
- Propagates file I/O errors from save operations

---

### `run_data(config: XFuseConfig)` — Refactored

**Behavior:** Now a thin wrapper that combines the two new functions.

```python
def run_data(config: XFuseConfig) -> None:
    xct_raw, xrd_raw = load_raw_data(config)
    run_data_from_raw(xct_raw, xrd_raw, config)
```

**Backward Compatibility:** ✓ Existing code calling `run_data(config)` continues to work identically.

---

## Data Flow

**Standard use (no pre-loading):**
```
run_data(config)
  ↓
load_raw_data(config)  [~10 min: loads 3D volumes]
  ↓
run_data_from_raw(xct_raw, xrd_raw, config)  [<1 min: extract slice + transform]
  ↓
Save outputs
```

**Fast iteration (pre-loaded):**
```
xct_raw, xrd_raw = load_raw_data(config)  [~10 min: once at top]
  ↓
for idx in [0, 5, 10]:
  config.replace(xct_sample_idx=idx)
    ↓
  run_data_from_raw(xct_raw, xrd_raw, config)  [<1 min each]
    ↓
  run_features(config)
    ↓
  ... (stages 2 & 3)
```

---

## Implementation Details

### New functions in `x_fuse/pipeline.py`

- `load_raw_data`: Extract lines 28–31 from current `run_data` (env + path setup) + call to `_load_raw_data`
- `run_data_from_raw`: Extract lines 32–56 from current `run_data` (everything after loading) into a new function
- `run_data`: Refactor to call both

### Existing private functions (unchanged)

- `_load_raw_data(config)` — continues to load full 3D volumes
- `_extract_sample(xct_raw, xrd_raw, config)` — continues to pick slices
- All other helpers remain unchanged

### No changes to

- `x_fuse/config.py`
- `x_fuse/loaders.py`
- `x_fuse/utils.py`
- Stages 1, 2, 3 (features, segment, refine)

---

## Testing

### New test: `test_load_raw_data_returns_correct_shapes`
- Call `load_raw_data()` on a test config
- Verify return types: tuple of (ndarray, dict)
- Verify shapes match expected data dimensions (3D volume layout)

### New test: `test_run_data_from_raw_matches_run_data`
- Load raw data once: `xct_raw, xrd_raw = load_raw_data(config)`
- Run both `run_data(config)` and `run_data_from_raw(xct_raw, xrd_raw, config)` on same config
- Verify output files (`xct.npy`, `{phase}_xrd.npy`, `data_summary.txt`) are identical

### Existing tests remain unchanged
- All current `test_run_data_*` tests should pass (they use `run_data` which is refactored, not replaced)

---

## Error Handling & Edge Cases

**Invalid data paths:** `_validate_paths` already handles; error is propagated by `load_raw_data`

**Slice index out of bounds:** `_extract_sample` may index out of range; behavior is NumPy indexing (wraps with negative indices, raises if out of bounds). Existing guards in tests catch this.

**File I/O failures:** `np.save` and text I/O operations can fail; errors propagate to caller (user sees traceback).

**Inversion flag changes:** User can change `config.invert` between calls to `run_data_from_raw` — each call behaves correctly based on its config.

---

## Rollback Plan

If the refactoring breaks something, `run_data` can be reverted to its original implementation (load + process in one function) without affecting the new public functions — they are pure additions and don't break the original contract.
