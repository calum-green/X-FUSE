# Data Loading, Slice Selection, and Notebook Visualisation Design

## Goal

Three related improvements to the Stage 0 (`run_data`) pipeline and supporting infrastructure:

1. **Granular loading prints** — per-step messages during data loading (XCT and per-phase XRDCT).
2. **Split slice indices** — replace the single `sample_idx` with `xct_sample_idx` and `xrdct_sample_idx` so XCT and XRDCT slices can be selected independently.
3. **Auto img_size** — remove `img_size` from config; detect it automatically from the native XCT slice shape, snapped down to the nearest multiple of 16 for ViT compatibility.
4. **Notebook visualisation fix** — figures are saved and displayed inline in Jupyter; currently `plt.close` fires before `plt.show` so nothing renders.

---

## Architecture

All changes are confined to `x_fuse/config.py`, `x_fuse/pipeline.py`, and the two example YAML files. No loader code changes. No changes to Stage 1–3 business logic, only the `img_size` derivation.

**Data flow for `img_size`:**
- `run_data` computes `img_size = _snap_to_multiple_of_16(xct_sample.shape[0])` after extraction.
- `xct.npy` is saved at that size.
- Stages 1–3 derive `img_size = xct.shape[0]` from the saved file — no config dependency.

**Data flow for slice indices:**
- `xct_sample_idx` indexes axis 0 of the XCT volume: `xct_raw[xct_sample_idx]` → `(H, W)`.
- `xrdct_sample_idx` indexes axis 0 of each XRDCT phase array: `arr[xrdct_sample_idx]` → `(H, W)`.
- Both DIAD `(1850,1850,1850)` and porespy `(N,H,W)` use axis-0 indexing — no shape branching needed.

---

## Changes

### `x_fuse/config.py`

**Remove:**
- `sample_idx: int = -1`
- `img_size: int = 224`

**Add:**
- `xct_sample_idx: int = -1`
- `xrdct_sample_idx: int = -1`

Update `from_yaml` to read `dataset.xct_sample_idx` and `dataset.xrdct_sample_idx`.
Update `to_yaml` to write the same keys. Remove all references to `sample_idx` and `img_size`.

---

### `x_fuse/pipeline.py`

#### New private helper: `_snap_to_multiple_of_16`

```python
def _snap_to_multiple_of_16(n: int) -> int:
    return (n // 16) * 16
```

Used in `run_data` to convert the native XCT slice dimension to a ViT-compatible size.

Examples:
| Native shape | Effective `img_size` |
|---|---|
| 1850 (DIAD) | 1840 |
| 224 | 224 |
| 300 | 288 |

#### `run_data`

Remove:
```python
print(f"  loading {config.dataset_type} data...")
```

Update extracting print:
```
  extracting sample (XCT idx=<xct_sample_idx>, XRDCT idx=<xrdct_sample_idx>)...
```

After `_extract_sample`, compute and print effective img_size:
```python
img_size = _snap_to_multiple_of_16(xct_sample.shape[0])
print(f"  effective img_size: {img_size}")
```

Pass `img_size` (not `config.img_size`) to `tr.get_input_transform` and `_write_data_summary`.

Remove the `plt.show()` calls after `_save_data_overview` in `run_data`, after `_save_mask_figure` in `run_segment`, and after `_save_sam2_figure` in `run_refine` — all moved into their respective helpers.

#### `_load_raw_data`

Add per-step prints. For the DIAD path with `entry_names` set (production case), load one phase at a time so each gets its own line:

```
  loading XCT...
  loading XRDCT [Na]...
  loading XRDCT [Zn]...
```

Implementation: call `load_xrdct_phase(phases=[phase], entry_names={phase: config.entry_names[phase]}, ...)` inside a `for phase in config.phases` loop instead of calling once for all phases.

For the DIAD fallback path (no `entry_names`), emit a single line:
```
  loading XRDCT (Na, Zn)...
```
and keep the existing single `load_diad_xrdct` call.

For porespy:
```
  loading porespy data...
```

#### `_extract_sample`

Remove the shape-based branching. Always index axis 0:

```python
def _extract_sample(xct_raw: np.ndarray, xrd_raw: dict, config: XFuseConfig) -> tuple:
    xct_sample = xct_raw[config.xct_sample_idx]
    xrd_sample = {p: arr[config.xrdct_sample_idx] for p, arr in xrd_raw.items()}
    return xct_sample, xrd_sample
```

#### `_write_data_summary`

Signature change: add `img_size: int` parameter (passed from `run_data`). Replace the `sample_idx` line in the output text with:
```
xct_sample_idx: <n>
xrdct_sample_idx: <n>
img_size (effective): <n>
```

#### `run_features`

Remove all `config.img_size` references. Derive locally:
```python
xct = np.load(data_dir / "xct.npy")
img_size = xct.shape[0]
```

Pass `img_size` to `_save_features_figures` (new parameter, replacing `config`).

#### `run_segment`

Remove all `config.img_size` references. Derive locally:
```python
xct = np.load(data_dir / "xct.npy")
img_size = xct.shape[0]
```

Use local `img_size` for `pca_map.reshape(img_size, img_size)`.

#### `_save_features_figures`

Signature change: replace `config: XFuseConfig` parameter with `img_size: int`. Use `img_size` for the PCA reshape. Remove all `config` references.

#### All `_save_*_figure` helpers: notebook visualisation fix

In `_save_data_overview`, `_save_mask_figure`, `_save_sam2_figure`, and inside `_save_features_figures`, add `plt.show()` before each `plt.close(fig)`:

```python
fig.savefig(out_dir / "...", dpi=150, bbox_inches="tight")
plt.show()      # renders inline in Jupyter; no-op in non-interactive backends
plt.close(fig)
```

Remove the now-redundant `plt.show()` calls from `run_data`, `run_segment`, and `run_refine` stage functions.

---

### `configs/example_diad.yaml`

Under `dataset`:
- Remove `sample_idx: -1`
- Add `xct_sample_idx: -1`
- Add `xrdct_sample_idx: -1`

Under `model`:
- Remove `img_size: 224`

### `configs/example_porespy.yaml`

Same changes as `example_diad.yaml`.

---

### `tests/test_pipeline.py`

- `test_config_defaults`: remove `img_size` assertion; add `xct_sample_idx == -1` and `xrdct_sample_idx == -1` assertions.
- `test_config_roundtrip`: add `xct_sample_idx` and `xrdct_sample_idx` to the constructed config and assert they roundtrip; remove `img_size`.
- Any `XFuseConfig(...)` calls that pass `img_size=...`: remove that argument.
- `_save_features_figures` call sites in tests: update signature (pass `img_size` int instead of `config`).

---

## Out of Scope

- No changes to `x_fuse/loaders.py`
- No changes to Stage 1–3 business logic (forward pass, PCA, thresholding, SAM2)
- No changes to `XFuse`, augmentation transforms, or shift/flip logic
- No new dataset types
- No averaging of XCT slice ranges (single-index selection only)
