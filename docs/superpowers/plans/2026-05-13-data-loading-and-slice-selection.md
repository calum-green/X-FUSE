# Data Loading, Slice Selection, and Notebook Visualisation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace `sample_idx`/`img_size` config fields with `xct_sample_idx`/`xrdct_sample_idx` and auto-detected img_size; add granular loading prints; fix notebook inline visualisation.

**Architecture:** Config changes land first (Task 1) since they define the contracts. Pipeline helpers (`_extract_sample`, `_load_raw_data`) are updated next, followed by stage functions (`run_data`, `run_features`, `run_segment`). `img_size` is computed once in `run_data` via `_snap_to_multiple_of_16(xct_sample.shape[0])`, saved into `xct.npy`, and then derived by downstream stages from `xct.shape[0]`. The `plt.show()` fix (Task 7) is independent and goes last.

**Tech Stack:** Python dataclasses, PyYAML, NumPy, Matplotlib, pytest

---

## File Map

| File | Change |
|------|--------|
| `x_fuse/config.py` | Remove `sample_idx`, `img_size`; add `xct_sample_idx`, `xrdct_sample_idx` |
| `x_fuse/pipeline.py` | Add `_snap_to_multiple_of_16`; rewrite `_extract_sample`, `_load_raw_data`; update all stage functions and figure helpers |
| `configs/example_diad.yaml` | Replace `sample_idx`/`img_size` with new fields |
| `configs/example_porespy.yaml` | Same |
| `tests/test_pipeline.py` | Update config tests, mock shapes, remove `img_size` from `XFuseConfig` calls |

---

### Task 1: Update XFuseConfig — remove sample_idx/img_size, add xct_sample_idx/xrdct_sample_idx

**Files:**
- Modify: `x_fuse/config.py`
- Modify: `configs/example_diad.yaml`
- Modify: `configs/example_porespy.yaml`
- Modify: `tests/test_pipeline.py`

- [ ] **Step 1: Update test_config_defaults and test_config_roundtrip**

In `tests/test_pipeline.py`, replace `test_config_defaults`:

```python
def test_config_defaults():
    cfg = minimal_config()
    assert cfg.dataset_type == "diad"
    assert cfg.fusion_method == "gating"
    assert cfg.stride == 4
    assert cfg.vis_data is True
    assert cfg.vis_dino_features is False
    assert cfg.device is None
    assert cfg.phases == ["Na", "Zn"]
    assert cfg.xct_sample_idx == -1
    assert cfg.xrdct_sample_idx == -1
```

Replace `test_config_roundtrip`:

```python
def test_config_roundtrip(tmp_path):
    cfg = XFuseConfig(
        name="roundtrip",
        dataset_type="porespy",
        phases=["alpha", "beta"],
        xct_sample_idx=5,
        xrdct_sample_idx=10,
        invert=True,
    )
    yaml_path = str(tmp_path / "cfg.yaml")
    cfg.to_yaml(yaml_path)
    loaded = XFuseConfig.from_yaml(yaml_path)
    assert loaded.name == "roundtrip"
    assert loaded.dataset_type == "porespy"
    assert loaded.phases == ["alpha", "beta"]
    assert loaded.xct_sample_idx == 5
    assert loaded.xrdct_sample_idx == 10
    assert loaded.invert is True
```

- [ ] **Step 2: Run config tests to confirm they fail**

```bash
pytest tests/test_pipeline.py::test_config_defaults tests/test_pipeline.py::test_config_roundtrip -v
```

Expected: FAIL — `XFuseConfig` has no `xct_sample_idx` attribute yet.

- [ ] **Step 3: Update XFuseConfig dataclass**

In `x_fuse/config.py`, in the `XFuseConfig` dataclass, replace:

```python
    sample_idx: int = -1
    n_samples: int = 10
```

with:

```python
    xct_sample_idx: int = -1
    xrdct_sample_idx: int = -1
    n_samples: int = 10
```

And remove `img_size: int = 224` from the `# model` section entirely.

- [ ] **Step 4: Update from_yaml**

In `XFuseConfig.from_yaml`, replace:

```python
            sample_idx=ds.get("sample_idx", -1),
```

with:

```python
            xct_sample_idx=ds.get("xct_sample_idx", -1),
            xrdct_sample_idx=ds.get("xrdct_sample_idx", -1),
```

And remove:

```python
            img_size=model.get("img_size", 224),
```

- [ ] **Step 5: Update to_yaml**

In `XFuseConfig.to_yaml`, in the `"dataset"` dict, replace:

```python
                "sample_idx": self.sample_idx,
```

with:

```python
                "xct_sample_idx": self.xct_sample_idx,
                "xrdct_sample_idx": self.xrdct_sample_idx,
```

And remove from the `"model"` dict:

```python
                "img_size": self.img_size,
```

- [ ] **Step 6: Update both example YAML configs**

In `configs/example_diad.yaml`, under `dataset:`, replace:

```yaml
  sample_idx: -1
```

with:

```yaml
  xct_sample_idx: -1
  xrdct_sample_idx: -1
```

And under `model:`, remove the line:

```yaml
  img_size: 224
```

Apply the same changes to `configs/example_porespy.yaml` (which has `sample_idx: -1` under `dataset:` and `img_size: 224` under `model:`).

- [ ] **Step 7: Remove img_size from all XFuseConfig constructions in tests**

In `tests/test_pipeline.py`, update `_make_diad_config` — remove `img_size=56`:

```python
def _make_diad_config(tmp_path, name="test") -> XFuseConfig:
    (tmp_path / "xct.h5").touch()
    (tmp_path / "phases").mkdir(exist_ok=True)
    return XFuseConfig(
        name=name,
        output_dir=str(tmp_path),
        dataset_type="diad",
        xct_path=str(tmp_path / "xct.h5"),
        phase_folder=str(tmp_path / "phases"),
        phases=["Na", "Zn"],
        vis_data=False,
    )
```

Also remove `img_size=img_size` (or `img_size=56`) from the `XFuseConfig(...)` calls in these tests:
- `test_run_data_custom_phases` (the `XFuseConfig` on line ~171)
- `test_run_data_invert_flag` (the `XFuseConfig` on line ~205)
- `test_run_features_saves_expected_files` (the `XFuseConfig` on line ~268)
- `test_run_segment_saves_masks_and_threshold` (the `XFuseConfig` on line ~326)
- `test_run_refine_saves_refined_masks_and_config` (the `XFuseConfig` on line ~371)

- [ ] **Step 8: Run config tests — expect pass**

```bash
pytest tests/test_pipeline.py::test_config_defaults tests/test_pipeline.py::test_config_roundtrip tests/test_pipeline.py::test_config_replace_does_not_mutate tests/test_pipeline.py::test_config_output_path -v
```

Expected: All PASS.

- [ ] **Step 9: Commit**

```bash
git add x_fuse/config.py configs/example_diad.yaml configs/example_porespy.yaml tests/test_pipeline.py
git commit -m "feat: replace sample_idx/img_size with xct_sample_idx/xrdct_sample_idx in config"
```

---

### Task 2: Add _snap_to_multiple_of_16 and rewrite _extract_sample

**Files:**
- Modify: `x_fuse/pipeline.py`
- Modify: `tests/test_pipeline.py`

- [ ] **Step 1: Write failing tests**

Add to `tests/test_pipeline.py` (after the existing config tests, before the `run_data` tests):

```python
def test_snap_to_multiple_of_16():
    from x_fuse.pipeline import _snap_to_multiple_of_16
    assert _snap_to_multiple_of_16(1850) == 1840
    assert _snap_to_multiple_of_16(224) == 224
    assert _snap_to_multiple_of_16(32) == 32
    assert _snap_to_multiple_of_16(17) == 16
    assert _snap_to_multiple_of_16(16) == 16


def test_extract_sample_uses_separate_indices():
    from x_fuse.pipeline import _extract_sample
    xct_raw = np.arange(3 * 8 * 8, dtype=np.float32).reshape(3, 8, 8)
    xrd_raw = {"Na": np.zeros((5, 4, 4), dtype=np.float32)}
    cfg = XFuseConfig(name="x", xct_sample_idx=1, xrdct_sample_idx=3)
    xct_s, xrd_s = _extract_sample(xct_raw, xrd_raw, cfg)
    assert xct_s.shape == (8, 8)
    np.testing.assert_array_equal(xct_s, xct_raw[1])
    assert xrd_s["Na"].shape == (4, 4)
    np.testing.assert_array_equal(xrd_s["Na"], xrd_raw["Na"][3])
```

- [ ] **Step 2: Run new tests to confirm they fail**

```bash
pytest tests/test_pipeline.py::test_snap_to_multiple_of_16 tests/test_pipeline.py::test_extract_sample_uses_separate_indices -v
```

Expected: FAIL — `_snap_to_multiple_of_16` not defined; `_extract_sample` uses old `sample_idx`.

- [ ] **Step 3: Add _snap_to_multiple_of_16 to pipeline.py**

In `x_fuse/pipeline.py`, add this function at the end of the private helpers section (after `_save_sam2_figure`):

```python
def _snap_to_multiple_of_16(n: int) -> int:
    return (n // 16) * 16
```

- [ ] **Step 4: Rewrite _extract_sample**

In `x_fuse/pipeline.py`, replace the entire `_extract_sample` function:

```python
def _extract_sample(xct_raw: np.ndarray, xrd_raw: dict, config: XFuseConfig) -> tuple:
    # (H, W, D) diad volume: slice along depth axis
    # (N, H, W) porespy stack: index first axis
    if xct_raw.ndim == 3 and xct_raw.shape[0] != xct_raw.shape[1]:
        xct_sample = xct_raw[config.sample_idx]
    else:
        xct_sample = xct_raw[:, :, config.sample_idx]
    xrd_sample = {p: arr[config.sample_idx] for p, arr in xrd_raw.items()}
    return xct_sample, xrd_sample
```

with:

```python
def _extract_sample(xct_raw: np.ndarray, xrd_raw: dict, config: XFuseConfig) -> tuple:
    xct_sample = xct_raw[config.xct_sample_idx]
    xrd_sample = {p: arr[config.xrdct_sample_idx] for p, arr in xrd_raw.items()}
    return xct_sample, xrd_sample
```

- [ ] **Step 5: Run new tests — expect pass**

```bash
pytest tests/test_pipeline.py::test_snap_to_multiple_of_16 tests/test_pipeline.py::test_extract_sample_uses_separate_indices -v
```

Expected: Both PASS.

- [ ] **Step 6: Commit**

```bash
git add x_fuse/pipeline.py tests/test_pipeline.py
git commit -m "feat: add _snap_to_multiple_of_16 helper and rewrite _extract_sample with axis-0 indexing"
```

---

### Task 3: Update run_data, _write_data_summary, and fix mock XCT shapes

**Files:**
- Modify: `x_fuse/pipeline.py`
- Modify: `tests/test_pipeline.py`

- [ ] **Step 1: Update mock XCT shapes in run_data tests**

The existing tests mock XCT as `(56, 56, 5)` shaped arrays, designed for the old `xct_raw[:, :, idx]` axis-2 indexing. With axis-0 indexing, `xct_raw[idx]` on `(56, 56, 5)` gives shape `(56, 5)` which fails validation. Change mock XCT to `(3, 32, 32)` — axis-0 index gives `(32, 32)` and `_snap_to_multiple_of_16(32) == 32`.

Note: The `img_size=` argument was already removed from `XFuseConfig(...)` calls in Task 1 Step 7. Only the mock array shapes change here.

In `test_run_data_saves_expected_files`, replace:

```python
    mock_xct.return_value = np.random.rand(56, 56, 5).astype(np.float32)
    mock_xrd.return_value = {
        "Na": np.random.rand(5, 8, 8).astype(np.float32),
        "Zn": np.random.rand(5, 8, 8).astype(np.float32),
    }
```

with:

```python
    mock_xct.return_value = np.random.rand(3, 32, 32).astype(np.float32)
    mock_xrd.return_value = {
        "Na": np.random.rand(3, 8, 8).astype(np.float32),
        "Zn": np.random.rand(3, 8, 8).astype(np.float32),
    }
```

In `test_run_data_custom_phases`, replace:

```python
    mock_xct.return_value = np.random.rand(56, 56, 3).astype(np.float32)
```

with:

```python
    mock_xct.return_value = np.random.rand(3, 32, 32).astype(np.float32)
```

In `test_run_data_invert_flag`, replace:

```python
    xct_arr = np.ones((56, 56, 3), dtype=np.float32) * 0.3
```

with:

```python
    xct_arr = np.ones((3, 32, 32), dtype=np.float32) * 0.3
```

- [ ] **Step 2: Run run_data tests to confirm they fail**

```bash
pytest tests/test_pipeline.py::test_run_data_saves_expected_files tests/test_pipeline.py::test_run_data_custom_phases tests/test_pipeline.py::test_run_data_invert_flag -v
```

Expected: FAIL — `run_data` still references `config.img_size` (AttributeError) and the extracting print still references `config.sample_idx`.

- [ ] **Step 3: Update _write_data_summary signature and content**

In `x_fuse/pipeline.py`, replace the entire `_write_data_summary` function:

```python
def _write_data_summary(
    out_dir: Path, xct: np.ndarray, xrd_dict: dict, config: XFuseConfig
) -> None:
    lines = [
        f"dataset_type: {config.dataset_type}",
        f"phases: {config.phases}",
        f"sample_idx: {config.sample_idx}",
        f"xct shape: {xct.shape}, min: {xct.min():.4f}, max: {xct.max():.4f}",
    ]
    for phase, arr in xrd_dict.items():
        lines.append(
            f"{phase}_xrd shape: {arr.shape}, "
            f"min: {arr.min():.4f}, max: {arr.max():.4f}"
        )
    (out_dir / "data_summary.txt").write_text("\n".join(lines))
```

with:

```python
def _write_data_summary(
    out_dir: Path, xct: np.ndarray, xrd_dict: dict, config: XFuseConfig, img_size: int
) -> None:
    lines = [
        f"dataset_type: {config.dataset_type}",
        f"phases: {config.phases}",
        f"xct_sample_idx: {config.xct_sample_idx}",
        f"xrdct_sample_idx: {config.xrdct_sample_idx}",
        f"img_size (effective): {img_size}",
        f"xct shape: {xct.shape}, min: {xct.min():.4f}, max: {xct.max():.4f}",
    ]
    for phase, arr in xrd_dict.items():
        lines.append(
            f"{phase}_xrd shape: {arr.shape}, "
            f"min: {arr.min():.4f}, max: {arr.max():.4f}"
        )
    (out_dir / "data_summary.txt").write_text("\n".join(lines))
```

- [ ] **Step 4: Update run_data**

In `x_fuse/pipeline.py`, replace the entire `run_data` function body:

```python
def run_data(config: XFuseConfig) -> None:
    """Validate, load, format, and save input data. CPU-only."""
    print(f"\n[X-FUSE] Stage 0 - data  ({config.name})")
    _set_cache_env(config)
    print("  validating paths...")
    _validate_paths(config)

    xct_raw, xrd_raw = _load_raw_data(config)
    print(f"  extracting sample (XCT idx={config.xct_sample_idx}, XRDCT idx={config.xrdct_sample_idx})...")
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

- [ ] **Step 5: Run run_data tests — expect pass**

```bash
pytest tests/test_pipeline.py::test_run_data_saves_expected_files tests/test_pipeline.py::test_run_data_custom_phases tests/test_pipeline.py::test_run_data_invert_flag tests/test_pipeline.py::test_run_data_missing_xct_path_raises -v
```

Expected: All PASS.

- [ ] **Step 6: Commit**

```bash
git add x_fuse/pipeline.py tests/test_pipeline.py
git commit -m "feat: update run_data to auto-detect img_size via _snap_to_multiple_of_16"
```

---

### Task 4: Add granular loading prints to _load_raw_data

**Files:**
- Modify: `x_fuse/pipeline.py`
- Modify: `tests/test_pipeline.py`

- [ ] **Step 1: Write failing test**

Add to `tests/test_pipeline.py` (in the run_data test section):

```python
@patch("x_fuse.pipeline.load_diad_xrdct")
@patch("x_fuse.pipeline.load_diad_xct_zn13x")
def test_load_raw_data_prints_loading_steps(mock_xct, mock_xrd, capsys):
    mock_xct.return_value = np.zeros((3, 32, 32), dtype=np.float32)
    mock_xrd.return_value = {"Na": np.zeros((3, 8, 8), dtype=np.float32)}
    cfg = XFuseConfig(name="x", dataset_type="diad", phases=["Na"])
    from x_fuse.pipeline import _load_raw_data
    _load_raw_data(cfg)
    out = capsys.readouterr().out
    assert "loading XCT" in out
    assert "loading XRDCT" in out
```

- [ ] **Step 2: Run test to confirm it fails**

```bash
pytest tests/test_pipeline.py::test_load_raw_data_prints_loading_steps -v
```

Expected: FAIL — no print output from `_load_raw_data`.

- [ ] **Step 3: Rewrite _load_raw_data with prints**

In `x_fuse/pipeline.py`, replace the entire `_load_raw_data` function:

```python
def _load_raw_data(config: XFuseConfig) -> tuple:
    if config.dataset_type == "diad":
        print("  loading XCT...")
        xct_raw = load_diad_xct_zn13x(config.xct_path)
        if config.entry_names:
            xrd_raw = {}
            for phase in config.phases:
                print(f"  loading XRDCT [{phase}]...")
                [arr] = load_xrdct_phase(
                    config.phase_folder,
                    phases=[phase],
                    shape=(21, 20, 20),
                    crop=slice(5, -1),
                    entry_names={phase: config.entry_names[phase]},
                )
                xrd_raw[phase] = np.flip(arr, axis=2).copy()
        else:
            print(f"  loading XRDCT ({', '.join(config.phases)})...")
            xrd_raw = load_diad_xrdct(config.phase_folder, phases=config.phases)
        return xct_raw, xrd_raw

    if config.dataset_type == "porespy":
        print("  loading porespy data...")
        if len(config.phases) != 2:
            raise ValueError(
                f"porespy dataset requires exactly 2 phases, got {config.phases}"
            )
        gray_imgs, low_A, low_B = get_ps_images()
        xrd_raw = {
            config.phases[0]: np.stack(low_A),
            config.phases[1]: np.stack(low_B),
        }
        return np.stack(gray_imgs), xrd_raw

    raise ValueError(f"Unknown dataset_type: {config.dataset_type!r}")
```

Note: `get_ps_images()` is called with no argument — it defaults to `IMG_SIZE=224`.

- [ ] **Step 4: Run test — expect pass**

```bash
pytest tests/test_pipeline.py::test_load_raw_data_prints_loading_steps -v
```

Expected: PASS.

- [ ] **Step 5: Run full run_data suite to check no regressions**

```bash
pytest tests/test_pipeline.py::test_run_data_saves_expected_files tests/test_pipeline.py::test_run_data_custom_phases tests/test_pipeline.py::test_run_data_invert_flag tests/test_pipeline.py::test_run_data_missing_xct_path_raises -v
```

Expected: All PASS.

- [ ] **Step 6: Commit**

```bash
git add x_fuse/pipeline.py tests/test_pipeline.py
git commit -m "feat: add granular per-step loading prints to _load_raw_data"
```

---

### Task 5: Update run_features and _save_features_figures — derive img_size from data

**Files:**
- Modify: `x_fuse/pipeline.py`
- Modify: `tests/test_pipeline.py`

- [ ] **Step 1: Update test to remove img_size from XFuseConfig**

In `test_run_features_saves_expected_files`, the `XFuseConfig` construction already had `img_size=img_size` removed in Task 1 Step 7. Verify it looks like:

```python
    cfg = XFuseConfig(
        name="run",
        output_dir=str(tmp_path),
        phases=phases,
        n_components=n_comp,
        vis_fused_maps=False,
        vis_pca_components=False,
        device="cpu",
    )
```

If `img_size=img_size` is still there, remove it now.

- [ ] **Step 2: Run features test to confirm it fails**

```bash
pytest tests/test_pipeline.py::test_run_features_saves_expected_files -v
```

Expected: FAIL — `run_features` references `config.img_size` (AttributeError).

- [ ] **Step 3: Update _save_features_figures signature**

In `x_fuse/pipeline.py`, replace the `_save_features_figures` function signature and body:

```python
def _save_features_figures(
    feat_dir: Path,
    phase: str,
    feats_np: np.ndarray,
    pcaed: np.ndarray,
    img_size: int,
    config: XFuseConfig,
) -> None:
    if config.vis_fused_maps:
        fig, ax = plt.subplots(figsize=(5, 5))
        ax.imshow(feats_np[0], cmap="viridis")
        ax.set_title(f"Fused features ch0 — {phase}")
        ax.axis("off")
        fig.savefig(feat_dir / f"{phase}_fused.png", dpi=150, bbox_inches="tight")
        plt.close(fig)

    if config.vis_pca_components:
        n = config.n_components
        fig, axs = plt.subplots(1, n, figsize=(3 * n, 3))
        h = w = img_size
        for i in range(n):
            axs[i].imshow(pcaed[:, i].reshape(h, w), cmap="viridis")
            axs[i].set_title(f"PCA {i + 1}")
            axs[i].axis("off")
        plt.suptitle(f"PCA components — {phase}")
        plt.tight_layout()
        fig.savefig(feat_dir / f"{phase}_pca_grid.png", dpi=150, bbox_inches="tight")
        plt.close(fig)
```

(The `plt.show()` calls are added to this function in Task 7.)

- [ ] **Step 4: Update run_features to derive img_size and pass it**

In `x_fuse/pipeline.py`, in `run_features`, replace:

```python
    img_tr = tr.get_input_transform(config.img_size, config.img_size)
    xct_tensor, _ = load_img(xct, img_tr)
```

with:

```python
    img_size = xct.shape[0]
    img_tr = tr.get_input_transform(img_size, img_size)
    xct_tensor, _ = load_img(xct, img_tr)
```

And replace the call to `_save_features_figures`:

```python
        _save_features_figures(feat_dir, phase, feats_np, pcaed, config)
```

with:

```python
        _save_features_figures(feat_dir, phase, feats_np, pcaed, img_size, config)
```

- [ ] **Step 5: Run features test — expect pass**

```bash
pytest tests/test_pipeline.py::test_run_features_saves_expected_files tests/test_pipeline.py::test_run_features_missing_data_raises -v
```

Expected: Both PASS.

- [ ] **Step 6: Commit**

```bash
git add x_fuse/pipeline.py tests/test_pipeline.py
git commit -m "feat: derive img_size from xct.npy in run_features; update _save_features_figures signature"
```

---

### Task 6: Update run_segment — derive img_size from data

**Files:**
- Modify: `x_fuse/pipeline.py`
- Modify: `tests/test_pipeline.py`

- [ ] **Step 1: Verify test has img_size removed from XFuseConfig**

In `test_run_segment_saves_masks_and_threshold`, confirm the `XFuseConfig` call no longer has `img_size=img_size` (removed in Task 1 Step 7). If still present, remove it now.

- [ ] **Step 2: Run segment test to confirm it fails**

```bash
pytest tests/test_pipeline.py::test_run_segment_saves_masks_and_threshold -v
```

Expected: FAIL — `run_segment` references `config.img_size`.

- [ ] **Step 3: Update run_segment to derive img_size**

In `x_fuse/pipeline.py`, in `run_segment`, after:

```python
    xct = np.load(data_dir / "xct.npy")
```

add:

```python
    img_size = xct.shape[0]
```

Then replace:

```python
        pca_map = pcaed[:, 0].reshape(config.img_size, config.img_size)
```

with:

```python
        pca_map = pcaed[:, 0].reshape(img_size, img_size)
```

Also verify `test_run_refine_saves_refined_masks_and_config` has `img_size=img_size` removed from its `XFuseConfig` call (Task 1 Step 7). `run_refine` does not use `config.img_size`, so no pipeline change needed there.

- [ ] **Step 4: Run segment and refine tests — expect pass**

```bash
pytest tests/test_pipeline.py::test_run_segment_saves_masks_and_threshold tests/test_pipeline.py::test_run_segment_missing_features_raises tests/test_pipeline.py::test_run_refine_saves_refined_masks_and_config tests/test_pipeline.py::test_run_refine_missing_segment_raises -v
```

Expected: All PASS.

- [ ] **Step 5: Commit**

```bash
git add x_fuse/pipeline.py tests/test_pipeline.py
git commit -m "feat: derive img_size from xct.npy in run_segment"
```

---

### Task 7: Fix plt.show() for inline notebook visualisation

**Files:**
- Modify: `x_fuse/pipeline.py`
- Modify: `tests/test_pipeline.py`

- [ ] **Step 1: Write test for vis_data showing and saving**

Add to `tests/test_pipeline.py`:

```python
@patch("x_fuse.pipeline.load_diad_xrdct")
@patch("x_fuse.pipeline.load_diad_xct_zn13x")
def test_run_data_vis_saves_overview_image(mock_xct, mock_xrd, tmp_path):
    mock_xct.return_value = np.random.rand(3, 32, 32).astype(np.float32)
    mock_xrd.return_value = {"Na": np.random.rand(3, 8, 8).astype(np.float32)}
    (tmp_path / "xct.h5").touch()
    (tmp_path / "phases").mkdir()
    cfg = XFuseConfig(
        name="vistest",
        output_dir=str(tmp_path),
        dataset_type="diad",
        xct_path=str(tmp_path / "xct.h5"),
        phase_folder=str(tmp_path / "phases"),
        phases=["Na"],
        vis_data=True,
    )
    run_data(cfg)
    assert (tmp_path / "vistest" / "data" / "data_overview.png").exists()
```

- [ ] **Step 2: Run test to confirm it passes already (save was already working)**

```bash
pytest tests/test_pipeline.py::test_run_data_vis_saves_overview_image -v
```

Expected: PASS (the file save already works; this test just confirms we don't break it).

- [ ] **Step 3: Add plt.show() inside _save_data_overview**

In `x_fuse/pipeline.py`, in `_save_data_overview`, replace:

```python
    fig.savefig(out_dir / "data_overview.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
```

with:

```python
    fig.savefig(out_dir / "data_overview.png", dpi=150, bbox_inches="tight")
    plt.show()
    plt.close(fig)
```

- [ ] **Step 4: Add plt.show() inside _save_mask_figure**

In `x_fuse/pipeline.py`, in `_save_mask_figure`, replace:

```python
    fig.savefig(seg_dir / f"{phase}_mask.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
```

with:

```python
    fig.savefig(seg_dir / f"{phase}_mask.png", dpi=150, bbox_inches="tight")
    plt.show()
    plt.close(fig)
```

- [ ] **Step 5: Add plt.show() inside _save_sam2_figure**

In `x_fuse/pipeline.py`, in `_save_sam2_figure`, replace:

```python
    fig.savefig(refine_dir / f"{phase}_sam2_overlay.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
```

with:

```python
    fig.savefig(refine_dir / f"{phase}_sam2_overlay.png", dpi=150, bbox_inches="tight")
    plt.show()
    plt.close(fig)
```

- [ ] **Step 6: Add plt.show() inside _save_features_figures**

In `x_fuse/pipeline.py`, in `_save_features_figures`, update both save+close pairs:

```python
    if config.vis_fused_maps:
        ...
        fig.savefig(feat_dir / f"{phase}_fused.png", dpi=150, bbox_inches="tight")
        plt.show()
        plt.close(fig)

    if config.vis_pca_components:
        ...
        fig.savefig(feat_dir / f"{phase}_pca_grid.png", dpi=150, bbox_inches="tight")
        plt.show()
        plt.close(fig)
```

- [ ] **Step 7: Remove redundant plt.show() from run_segment and run_refine**

In `run_segment`, inside the `if config.vis_masks:` block, remove:

```python
            plt.show()
```

In `run_refine`, inside the `if config.vis_sam2:` block, remove:

```python
            plt.show()
```

(The `plt.show()` in `run_data` was already removed in Task 3.)

- [ ] **Step 8: Run full test suite**

```bash
pytest tests/ -v
```

Expected: All tests PASS. This is the final integration check.

- [ ] **Step 9: Commit**

```bash
git add x_fuse/pipeline.py tests/test_pipeline.py
git commit -m "fix: move plt.show() inside figure helpers for inline notebook display"
```
