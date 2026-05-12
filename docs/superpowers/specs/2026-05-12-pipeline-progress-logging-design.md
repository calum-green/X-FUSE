# Pipeline Progress Logging Design

## Goal

Add informative print statements and tqdm progress bars to the four X-FUSE pipeline stages so the user can follow execution in both Jupyter notebooks and the CLI.

## Architecture

Print output is always on (no verbose flag). `tqdm.auto` is used so the progress bar renders as a Jupyter widget in notebooks and as a text bar in the terminal. Prints are added directly to the four public stage functions; private helpers are unchanged.

## Tech Stack

- `tqdm>=4.0` (new core dependency, `tqdm.auto` import)
- Python built-in `print()`

---

## Changes

### `pyproject.toml`

Add `"tqdm>=4.0"` to the `dependencies` list.

### `x_fuse/pipeline.py`

Add `from tqdm.auto import tqdm` to imports.

#### Stage 0 — `run_data` (no tqdm, fast CPU stage)

```
[X-FUSE] Stage 0 - data  (<run_name>)
  validating paths...
  loading <dataset_type> data...
  extracting sample (idx=<sample_idx>)...
  transforming XCT image...
  saving to <out_dir>/
  done.
```

Print sequence:
1. Stage banner before `_set_cache_env`
2. `"  validating paths..."` before `_validate_paths`
3. `"  loading <dataset_type> data..."` before `_load_raw_data`
4. `"  extracting sample (idx=<sample_idx>)..."` before `_extract_sample`
5. `"  inverting images..."` inside the `if config.invert` block
6. `"  transforming XCT image..."` before `_apply_xct_transform`
7. `"  saving to <out_dir>/"` after `out_dir.mkdir`
8. `"  done."` at end of function

#### Stage 1 — `run_features` (tqdm over phases, `tqdm.write()` for substeps)

```
[X-FUSE] Stage 1 - features  (<run_name>)
  device: <device>
  phases:  50%|█████     | 1/2 [00:23<00:23]
    [<phase>] building XFuse model...
    [<phase>] running forward pass...
    [<phase>] PCA (<n_components> components)...
  features saved to <feat_dir>/
```

Print sequence:
1. Stage banner before `from .fusion import XFuse`
2. `"  device: <device>"` after `resolve_device()`
3. Replace existing `for phase in config.phases:` with `for phase in tqdm(config.phases, desc="  phases", unit="phase"):`
4. `tqdm.write(f"    [{phase}] building XFuse model...")` before `XFuse(...)` construction
5. `tqdm.write(f"    [{phase}] running forward pass...")` before `net.forward_sequential`
6. `tqdm.write(f"    [{phase}] PCA ({config.n_components} components)...")` before `rescale_pca(...)`
7. `"  features saved to <feat_dir>/"` after the loop

Also remove the existing `print(f"[run_features] device: {device}")` line (replaced by item 2).

#### Stage 2 — `run_segment` (tqdm over phases)

```
[X-FUSE] Stage 2 - segment  (<run_name>)
  threshold: <threshold>
  phases: 100%|██████████| 2/2 [00:00<00:00]
  masks saved to <seg_dir>/
```

Print sequence:
1. Stage banner at start
2. `"  threshold: <threshold>"` on next line
3. Replace `for phase in config.phases:` with `for phase in tqdm(config.phases, desc="  phases", unit="phase"):`
4. `"  masks saved to <seg_dir>/"` after the loop

#### Stage 3 — `run_refine` (tqdm over phases, score printed per phase)

```
[X-FUSE] Stage 3 - refine  (<run_name>)
  device: <device>
  loading SAM2 model...
  phases: 100%|██████████| 2/2 [00:08<00:00]
    [<phase>] SAM2 score: 0.927
  refined masks saved to <refine_dir>/
```

Print sequence:
1. Stage banner before `from sam2...` imports
2. `"  device: <device>"` after `resolve_device()`
3. `"  loading SAM2 model..."` before `build_sam2(...)`
4. Replace `for phase in config.phases:` with `for phase in tqdm(config.phases, desc="  phases", unit="phase"):`
5. `tqdm.write(f"    [{phase}] SAM2 score: {float(scores[0]):.3f}")` after `predictor.predict`
6. `"  refined masks saved to <refine_dir>/"` after the loop and `config.to_yaml`

Also remove the existing `print(f"[run_refine] device: {device}")` line (replaced by item 2).

---

## Out of Scope

- No changes to private helper functions
- No changes to `XFuseConfig`
- No changes to tests (prints do not affect return values or saved files)
- No logging module, no rich library
