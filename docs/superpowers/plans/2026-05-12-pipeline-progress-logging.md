# Pipeline Progress Logging Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add stage banners, substep print statements, and per-phase tqdm progress bars to the four X-FUSE pipeline stage functions.

**Architecture:** All changes are confined to `x_fuse/pipeline.py` (plus one line in `pyproject.toml`). Print output is always on. `tqdm.auto` is used so the progress bar renders as a Jupyter widget in notebooks and a text bar in the CLI. `tqdm.write()` is used for any prints made inside a tqdm loop to avoid line clobbering. No changes to tests — prints don't affect return values or saved files; the existing test suite verifies correctness after each task.

**Tech Stack:** `tqdm>=4.0` (`tqdm.auto`)

---

### Task 1: Add tqdm dependency

**Files:**
- Modify: `pyproject.toml:13-29`

- [ ] **Step 1: Add `tqdm>=4.0` to the dependencies list in `pyproject.toml`**

Open `pyproject.toml`. Find this block:

```toml
    "pyyaml>=6.0",
    "sam2>=1.1.0",
```

Change it to:

```toml
    "pyyaml>=6.0",
    "tqdm>=4.0",
    "sam2>=1.1.0",
```

- [ ] **Step 2: Verify tqdm.auto is importable**

Run:
```bash
python -c "from tqdm.auto import tqdm; print('ok')"
```

Expected output: `ok`

If it fails with `ModuleNotFoundError`, run `pip install tqdm` and retry.

- [ ] **Step 3: Commit**

```bash
git add pyproject.toml
git commit -m "chore: add tqdm as explicit dependency"
```

---

### Task 2: Add tqdm import and update all four stage functions

No new tests are required — `print()` and `tqdm` calls don't affect the files saved or values returned by any stage. The existing test suite (run in Step 6) verifies that nothing is broken.

**Files:**
- Modify: `x_fuse/pipeline.py:1-17` (imports)
- Modify: `x_fuse/pipeline.py:24-52` (`run_data`)
- Modify: `x_fuse/pipeline.py:60-123` (`run_features`)
- Modify: `x_fuse/pipeline.py:130-152` (`run_segment`)
- Modify: `x_fuse/pipeline.py:160-205` (`run_refine`)

- [ ] **Step 1: Add `tqdm.auto` import to `pipeline.py`**

Find this line near the top of `x_fuse/pipeline.py`:

```python
import hr_dv2.transform as tr
```

Change it to:

```python
import hr_dv2.transform as tr
from tqdm.auto import tqdm
```

- [ ] **Step 2: Update `run_data` with stage banner and substep prints**

Find the full body of `run_data` in `x_fuse/pipeline.py`:

```python
def run_data(config: XFuseConfig) -> None:
    """Validate, load, format, and save input data. CPU-only."""
    _set_cache_env(config)
    _validate_paths(config)

    xct_raw, xrd_raw = _load_raw_data(config)
    xct_sample, xrd_sample = _extract_sample(xct_raw, xrd_raw, config)

    if config.invert:
        xct_sample = invert_image(xct_sample)
        xrd_sample = {p: invert_image(a) for p, a in xrd_sample.items()}

    _validate_arrays(xct_sample, xrd_sample)

    img_tr = tr.get_input_transform(config.img_size, config.img_size)
    xct_transformed = _apply_xct_transform(xct_sample, img_tr)

    out_dir = config.output_path / "data"
    out_dir.mkdir(parents=True, exist_ok=True)

    np.save(out_dir / "xct.npy", xct_transformed.astype(np.float32))
    for phase, arr in xrd_sample.items():
        np.save(out_dir / f"{phase}_xrd.npy", arr.astype(np.float32))

    _write_data_summary(out_dir, xct_transformed, xrd_sample, config)

    if config.vis_data:
        _save_data_overview(out_dir, xct_transformed, xrd_sample)
        plt.show()
```

Replace it with:

```python
def run_data(config: XFuseConfig) -> None:
    """Validate, load, format, and save input data. CPU-only."""
    print(f"\n[X-FUSE] Stage 0 - data  ({config.name})")
    _set_cache_env(config)
    print("  validating paths...")
    _validate_paths(config)

    print(f"  loading {config.dataset_type} data...")
    xct_raw, xrd_raw = _load_raw_data(config)
    print(f"  extracting sample (idx={config.sample_idx})...")
    xct_sample, xrd_sample = _extract_sample(xct_raw, xrd_raw, config)

    if config.invert:
        print("  inverting images...")
        xct_sample = invert_image(xct_sample)
        xrd_sample = {p: invert_image(a) for p, a in xrd_sample.items()}

    _validate_arrays(xct_sample, xrd_sample)

    print("  transforming XCT image...")
    img_tr = tr.get_input_transform(config.img_size, config.img_size)
    xct_transformed = _apply_xct_transform(xct_sample, img_tr)

    out_dir = config.output_path / "data"
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"  saving to {out_dir}/")

    np.save(out_dir / "xct.npy", xct_transformed.astype(np.float32))
    for phase, arr in xrd_sample.items():
        np.save(out_dir / f"{phase}_xrd.npy", arr.astype(np.float32))

    _write_data_summary(out_dir, xct_transformed, xrd_sample, config)

    if config.vis_data:
        _save_data_overview(out_dir, xct_transformed, xrd_sample)
        plt.show()
    print("  done.")
```

- [ ] **Step 3: Update `run_features` — replace existing device print, add stage banner, tqdm loop, and substep writes**

Find the full body of `run_features`:

```python
def run_features(config: XFuseConfig) -> None:
    """XFuse forward pass + PCA. Requires GPU."""
    from .fusion import XFuse

    device = config.resolve_device()
    print(f"[run_features] device: {device}")

    data_dir = config.output_path / "data"
    _check_stage_inputs(
        data_dir,
        ["xct.npy"] + [f"{p}_xrd.npy" for p in config.phases],
        prior_stage="data",
    )

    xct = np.load(data_dir / "xct.npy")
    xrd_dict = {p: np.load(data_dir / f"{p}_xrd.npy") for p in config.phases}

    shift_dists = list(config.shift_distances)
    fwd_shift, inv_shift = tr.get_shift_transforms(shift_dists, "Moore")
    fwd_flip, inv_flip = tr.get_flip_transforms()
    fwd, inv = tr.combine_transforms(fwd_shift, fwd_flip, inv_shift, inv_flip)

    img_tr = tr.get_input_transform(config.img_size, config.img_size)
    xct_tensor, _ = load_img(xct, img_tr)
    xct_tensor = xct_tensor.to(torch.float16).to(device)

    feat_dir = config.output_path / "features"
    feat_dir.mkdir(parents=True, exist_ok=True)

    for phase in config.phases:
        xrd_tensor = xrd_to_tensor(xrd_dict[phase], device)

        net = XFuse(
            config.dino_model,
            config.fusion_method,
            stride=config.stride,
            pca_dim=-1,
            track_grad=False,
            dtype=torch.float16,
            device=device,
            loss_fn=config.loss_fn,
            model_path=config.model_path,
            chk_path=config.chk_path,
            lib_path=config.lib_path,
        )
        net.set_xrd_transforms(xrd_tensor, fwd, inv)
        feats = net.forward_sequential(xct_tensor, top_k=config.top_k)

        feats_cpu = feats[0].cpu()
        feats_np = tr.to_numpy(feats_cpu)
        feats_flat = tr.flatten(
            feats_np, feats_np.shape[1], feats_np.shape[2], feats_np.shape[0]
        )
        np.save(feat_dir / f"{phase}_feats.npy", feats_np.astype(np.float32))

        pcaed = rescale_pca(
            do_single_pca(
                feats_flat, config.n_components, n_samples=config.n_samples_pca
            )
        )
        np.save(feat_dir / f"{phase}_pca.npy", pcaed.astype(np.float32))

        _save_features_figures(feat_dir, phase, feats_np, pcaed, config)
```

Replace it with:

```python
def run_features(config: XFuseConfig) -> None:
    """XFuse forward pass + PCA. Requires GPU."""
    print(f"\n[X-FUSE] Stage 1 - features  ({config.name})")
    from .fusion import XFuse

    device = config.resolve_device()
    print(f"  device: {device}")

    data_dir = config.output_path / "data"
    _check_stage_inputs(
        data_dir,
        ["xct.npy"] + [f"{p}_xrd.npy" for p in config.phases],
        prior_stage="data",
    )

    xct = np.load(data_dir / "xct.npy")
    xrd_dict = {p: np.load(data_dir / f"{p}_xrd.npy") for p in config.phases}

    shift_dists = list(config.shift_distances)
    fwd_shift, inv_shift = tr.get_shift_transforms(shift_dists, "Moore")
    fwd_flip, inv_flip = tr.get_flip_transforms()
    fwd, inv = tr.combine_transforms(fwd_shift, fwd_flip, inv_shift, inv_flip)

    img_tr = tr.get_input_transform(config.img_size, config.img_size)
    xct_tensor, _ = load_img(xct, img_tr)
    xct_tensor = xct_tensor.to(torch.float16).to(device)

    feat_dir = config.output_path / "features"
    feat_dir.mkdir(parents=True, exist_ok=True)

    for phase in tqdm(config.phases, desc="  phases", unit="phase"):
        tqdm.write(f"    [{phase}] building XFuse model...")
        xrd_tensor = xrd_to_tensor(xrd_dict[phase], device)

        net = XFuse(
            config.dino_model,
            config.fusion_method,
            stride=config.stride,
            pca_dim=-1,
            track_grad=False,
            dtype=torch.float16,
            device=device,
            loss_fn=config.loss_fn,
            model_path=config.model_path,
            chk_path=config.chk_path,
            lib_path=config.lib_path,
        )
        net.set_xrd_transforms(xrd_tensor, fwd, inv)
        tqdm.write(f"    [{phase}] running forward pass...")
        feats = net.forward_sequential(xct_tensor, top_k=config.top_k)

        feats_cpu = feats[0].cpu()
        feats_np = tr.to_numpy(feats_cpu)
        feats_flat = tr.flatten(
            feats_np, feats_np.shape[1], feats_np.shape[2], feats_np.shape[0]
        )
        np.save(feat_dir / f"{phase}_feats.npy", feats_np.astype(np.float32))

        tqdm.write(f"    [{phase}] PCA ({config.n_components} components)...")
        pcaed = rescale_pca(
            do_single_pca(
                feats_flat, config.n_components, n_samples=config.n_samples_pca
            )
        )
        np.save(feat_dir / f"{phase}_pca.npy", pcaed.astype(np.float32))

        _save_features_figures(feat_dir, phase, feats_np, pcaed, config)
    print(f"  features saved to {feat_dir}/")
```

- [ ] **Step 4: Update `run_segment` — add stage banner, threshold print, and tqdm loop**

Find the full body of `run_segment`:

```python
def run_segment(config: XFuseConfig, threshold: float) -> None:
    """Apply threshold to PCA component 0. CPU-only."""
    feat_dir = config.output_path / "features"
    data_dir = config.output_path / "data"
    _check_stage_inputs(
        feat_dir, [f"{p}_pca.npy" for p in config.phases], prior_stage="features"
    )
    _check_stage_inputs(data_dir, ["xct.npy"], prior_stage="data")

    xct = np.load(data_dir / "xct.npy")
    seg_dir = config.output_path / "segment"
    seg_dir.mkdir(parents=True, exist_ok=True)
    (seg_dir / "threshold.txt").write_text(str(threshold))

    for phase in config.phases:
        pcaed = np.load(feat_dir / f"{phase}_pca.npy")
        pca_map = pcaed[:, 0].reshape(config.img_size, config.img_size)
        mask = (pca_map > threshold).astype(np.uint8)
        np.save(seg_dir / f"{phase}_mask.npy", mask)

        if config.vis_masks:
            _save_mask_figure(seg_dir, phase, xct, mask, threshold)
            plt.show()
```

Replace it with:

```python
def run_segment(config: XFuseConfig, threshold: float) -> None:
    """Apply threshold to PCA component 0. CPU-only."""
    print(f"\n[X-FUSE] Stage 2 - segment  ({config.name})")
    print(f"  threshold: {threshold}")
    feat_dir = config.output_path / "features"
    data_dir = config.output_path / "data"
    _check_stage_inputs(
        feat_dir, [f"{p}_pca.npy" for p in config.phases], prior_stage="features"
    )
    _check_stage_inputs(data_dir, ["xct.npy"], prior_stage="data")

    xct = np.load(data_dir / "xct.npy")
    seg_dir = config.output_path / "segment"
    seg_dir.mkdir(parents=True, exist_ok=True)
    (seg_dir / "threshold.txt").write_text(str(threshold))

    for phase in tqdm(config.phases, desc="  phases", unit="phase"):
        pcaed = np.load(feat_dir / f"{phase}_pca.npy")
        pca_map = pcaed[:, 0].reshape(config.img_size, config.img_size)
        mask = (pca_map > threshold).astype(np.uint8)
        np.save(seg_dir / f"{phase}_mask.npy", mask)

        if config.vis_masks:
            _save_mask_figure(seg_dir, phase, xct, mask, threshold)
            plt.show()
    print(f"  masks saved to {seg_dir}/")
```

- [ ] **Step 5: Update `run_refine` — replace existing device print, add stage banner, SAM2 loading print, tqdm loop, and per-phase score write**

Find the full body of `run_refine`:

```python
def run_refine(config: XFuseConfig) -> None:
    """SAM2 refinement of binary masks. Requires GPU."""
    from sam2.build_sam import build_sam2
    from sam2.sam2_image_predictor import SAM2ImagePredictor

    device = config.resolve_device()
    print(f"[run_refine] device: {device}")

    seg_dir = config.output_path / "segment"
    data_dir = config.output_path / "data"
    _check_stage_inputs(
        seg_dir, [f"{p}_mask.npy" for p in config.phases], prior_stage="segment"
    )
    _check_stage_inputs(data_dir, ["xct.npy"], prior_stage="data")

    xct = np.load(data_dir / "xct.npy")

    sam2_cfg = f"{config.sam2_folder}{config.sam2_model}.yaml"
    sam2_ckpt = f"{config.sam2_folder}{config.sam2_model}.pt"
    sam2_model = build_sam2(sam2_cfg, sam2_ckpt, device=device)
    predictor = SAM2ImagePredictor(sam2_model)

    xct_pil = ImageOps.autocontrast(
        Image.fromarray((xct * 255).astype(np.uint8)).convert("RGB")
    )
    predictor.set_image(np.array(xct_pil))

    refine_dir = config.output_path / "refine"
    refine_dir.mkdir(parents=True, exist_ok=True)

    for phase in config.phases:
        mask = np.load(seg_dir / f"{phase}_mask.npy").astype(np.float32)
        mask_input = cv2.resize(mask, (256, 256))[None]
        mask_input = (mask_input * 2 - 1) * 10
        with torch.inference_mode():
            masks, scores, _ = predictor.predict(
                mask_input=mask_input, multimask_output=False
            )
        refined = masks[0].astype(bool)
        np.save(refine_dir / f"{phase}_refined_mask.npy", refined)

        if config.vis_sam2:
            _save_sam2_figure(refine_dir, phase, xct, refined, float(scores[0]))
            plt.show()

    config.to_yaml(str(refine_dir / "config.yaml"))
```

Replace it with:

```python
def run_refine(config: XFuseConfig) -> None:
    """SAM2 refinement of binary masks. Requires GPU."""
    print(f"\n[X-FUSE] Stage 3 - refine  ({config.name})")
    from sam2.build_sam import build_sam2
    from sam2.sam2_image_predictor import SAM2ImagePredictor

    device = config.resolve_device()
    print(f"  device: {device}")

    seg_dir = config.output_path / "segment"
    data_dir = config.output_path / "data"
    _check_stage_inputs(
        seg_dir, [f"{p}_mask.npy" for p in config.phases], prior_stage="segment"
    )
    _check_stage_inputs(data_dir, ["xct.npy"], prior_stage="data")

    xct = np.load(data_dir / "xct.npy")

    sam2_cfg = f"{config.sam2_folder}{config.sam2_model}.yaml"
    sam2_ckpt = f"{config.sam2_folder}{config.sam2_model}.pt"
    print("  loading SAM2 model...")
    sam2_model = build_sam2(sam2_cfg, sam2_ckpt, device=device)
    predictor = SAM2ImagePredictor(sam2_model)

    xct_pil = ImageOps.autocontrast(
        Image.fromarray((xct * 255).astype(np.uint8)).convert("RGB")
    )
    predictor.set_image(np.array(xct_pil))

    refine_dir = config.output_path / "refine"
    refine_dir.mkdir(parents=True, exist_ok=True)

    for phase in tqdm(config.phases, desc="  phases", unit="phase"):
        mask = np.load(seg_dir / f"{phase}_mask.npy").astype(np.float32)
        mask_input = cv2.resize(mask, (256, 256))[None]
        mask_input = (mask_input * 2 - 1) * 10
        with torch.inference_mode():
            masks, scores, _ = predictor.predict(
                mask_input=mask_input, multimask_output=False
            )
        refined = masks[0].astype(bool)
        np.save(refine_dir / f"{phase}_refined_mask.npy", refined)
        tqdm.write(f"    [{phase}] SAM2 score: {float(scores[0]):.3f}")

        if config.vis_sam2:
            _save_sam2_figure(refine_dir, phase, xct, refined, float(scores[0]))
            plt.show()

    config.to_yaml(str(refine_dir / "config.yaml"))
    print(f"  refined masks saved to {refine_dir}/")
```

- [ ] **Step 6: Run the existing test suite**

```bash
pytest tests/ -m "not gpu" -v
```

Expected: all non-GPU tests pass (same count as before — currently 257 passed, 2 deselected, 3 warnings). If any test fails, the change introduced a regression — fix before committing.

- [ ] **Step 7: Commit**

```bash
git add x_fuse/pipeline.py
git commit -m "feat: add stage banners, substep prints, and tqdm phase progress to pipeline"
```
