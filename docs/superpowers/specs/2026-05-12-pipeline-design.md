# X-FUSE Modular Pipeline Design

**Date:** 2026-05-12  
**Branch:** feature/modularise-notebooks  
**Status:** Approved

---

## Overview

Convert the notebook-based X-FUSE workflow (`notebooks/ps_dv3.ipynb`, `notebooks/dev.ipynb`) into a modular, config-driven pipeline. Users specify a YAML file containing all hyperparameters and dataset paths. The pipeline runs in four independent stages: data preparation, feature extraction, segmentation (thresholding), and SAM2 refinement. A user guide documents all additions.

---

## Goals

- Replace ad-hoc notebook hyperparams with a single versioned YAML config per run
- Enable headless HPC execution via a CLI entry point
- Keep a thin notebook driver for interactive use (same underlying functions)
- Allow configs to be created programmatically in Python and saved to YAML
- Produce per-run output directories so results from different configs are never mixed
- Document everything for use by the author and collaborators

## Non-Goals

- Changing any existing logic in `loaders.py`, `fusion.py`, `utils.py`, or `alibi.py`
- Supporting config sweeps or hyperparameter search (can be layered on later)
- A GUI or web interface

---

## Architecture

### New files

| File | Purpose |
|---|---|
| `x_fuse/config.py` | `XFuseConfig` dataclass with `from_yaml()`, `to_yaml()`, `replace()` |
| `x_fuse/pipeline.py` | `run_data()`, `run_features()`, `run_segment()`, `run_refine()` stage functions |
| `scripts/run_xfuse.py` | CLI entry point (`--config`, `--stage`, `--threshold`, `--vis-all`) |
| `configs/example_diad.yaml` | Example config for real DIAD XRD-CT data |
| `configs/example_porespy.yaml` | Example config for synthetic PoreSpy data |
| `docs/pipeline.md` | User guide covering config creation, CLI usage, and notebook usage |

### Unchanged files

`x_fuse/loaders.py`, `x_fuse/fusion.py`, `x_fuse/utils.py`, `x_fuse/alibi.py` — pipeline functions call these directly; no modifications needed.

### Thin notebook driver

`notebooks/run_pipeline.ipynb` — replaces both existing development notebooks. Contains ~10 lines of logic: load config, call the four stage functions. All visualisations are handled inside `pipeline.py`.

---

## Config Design (`x_fuse/config.py`)

### Structure

A stdlib `dataclass` populated from YAML via PyYAML. No new runtime dependencies.

```yaml
run:
  name: "zn13x_run1"        # used as output subdirectory name
  output_dir: "outputs/"

environment:
  torch_cache: "/dls/science/groups/imaging/infuse/XRDCT_Fusion/torch_cache/"

dataset:
  type: "diad"               # "diad" | "porespy"
  # diad-specific:
  xct_path: "/path/to/xct.h5"
  phase_folder: "/path/to/phases/"
  phases: ["Na", "Zn"]
  sample_idx: -1             # index of sample/slice to process (applies to both dataset types)
  # porespy-specific (ignored when type != "porespy"):
  # n_samples: 10
  # downsample_factor: 10

model:
  dino_model: "nope_dv3_vits16plus_1625"
  model_path: "/path/to/model.pth"
  lib_path: "/path/to/dinov3/"  # required for dv3/alibi models; null for dinov2
  img_size: 224              # target image size for both diad and porespy
  stride: 4
  fusion_method: "gating"    # "gating" | "learned_gating" | "attention"
  loss_fn: "bce"             # "bce" | "pearson"
  top_k: null                # channels to keep; null defaults to C // 4
  invert: false              # invert image intensities before processing

augmentation:
  shift_distances: [1, 2]
  use_flip: true

pca:
  n_components: 10
  n_samples: 5000            # pixels sampled for PCA fitting

sam2:
  folder: "//dls/science/groups/imaging/infuse/SAM2/"
  model: "sam2.1_hiera_small"

vis:
  data: true                 # loaded XCT and XRD maps per phase (inspect before running GPU)
  dino_features: false       # raw DINO PCA before fusion
  fused_maps: true           # XRD-fused feature maps per phase
  pca_components: true       # PCA component grid (n_components panels)
  masks: true                # binary threshold masks per phase
  sam2: true                 # SAM2-refined masks overlaid on XCT
```

### Defaults

All `model`, `augmentation`, `pca`, and `vis` fields have sensible defaults so only paths and identity fields are required at construction time. `vis.data` defaults to `true` since inspecting loaded data before GPU work is almost always desirable.

### API

```python
# Load from file
config = XFuseConfig.from_yaml("configs/my_run.yaml")

# Create programmatically (required fields only; rest use defaults)
config = XFuseConfig(
    name="zn13x_run1",
    dataset_type="diad",
    xct_path="/path/to/xct.h5",
    phase_folder="/path/to/phases/",
    phases=["Na", "Zn"],
    dino_model="nope_dv3_vits16plus_1625",
    model_path="/path/to/model.pth",
    lib_path="/path/to/dinov3/",
    torch_cache="/dls/.../torch_cache/",
    sam2_folder="//dls/.../SAM2/",
)

# Save to YAML
config.to_yaml("configs/zn13x_run1.yaml")

# Create a variant without mutating the original
cfg_attn = config.replace(name="zn13x_attention", fusion_method="attention")
cfg_attn.to_yaml("configs/zn13x_attention.yaml")
```

`replace()` returns a new `XFuseConfig` with the specified fields overridden — safe to use in loops for generating experiment variants.

---

## Pipeline Design (`x_fuse/pipeline.py`)

### Stage 0 — `run_data(config: XFuseConfig) -> None`

Validates, loads, and formats input data. Runs on CPU; no GPU required. Intended to be run first so data can be inspected before committing to the GPU-intensive feature extraction stage.

**Validation (fail fast before any loading):**
- All required file paths exist (`xct_path`, `phase_folder`, `model_path`, `lib_path` if set, `sam2_folder`)
- `dataset.type` is one of the registered types (`"diad"`, `"porespy"`)
- `dataset.phases` is non-empty
- `dataset.sample_idx` is within bounds after loading

**Loading and formatting:**
1. Set HuggingFace / Torch cache environment variables from `config.environment`
2. Load raw data via the loader registry:
   - `"diad"` → `load_diad_xct_zn13x` + `load_diad_xrdct` from `loaders.py`
   - `"porespy"` → `get_ps_images` from `utils.py`
3. Extract sample at `dataset.sample_idx`
4. Validate loaded arrays: assert shapes are consistent, XCT and XRD spatial dims are compatible after downsampling, and pixel values are in expected range (`[0, 1]` after normalisation)
5. Optionally invert XCT and XRD arrays (`model.invert`)
6. Apply image transforms (`tr.closest_crop`, `tr.get_input_transform`) to XCT
7. Save standardised arrays to `outputs/<name>/data/`:
   - `xct.npy` — transformed XCT image as `float32`
   - `<phase>_xrd.npy` — normalised XRD map per phase as `float32`
   - `data_summary.txt` — shapes, value ranges, dataset type, loader used
8. Save figure if `vis.data`: side-by-side panel of XCT + all XRD phase maps (`data_overview.png`)

### Stage 1 — `run_features(config: XFuseConfig) -> None`

Reads prepared arrays from Stage 0. No data loading or validation here.

1. Detect device (CUDA → MPS → CPU)
2. Load `xct.npy` and `<phase>_xrd.npy` from `outputs/<name>/data/` — raise `FileNotFoundError` with message `"Run --stage data first"` if absent
3. Build image tensor and XRD tensors per phase (tensorise, move to device)
4. Build augmentation transforms (`augmentation.shift_distances`, `augmentation.use_flip`)
5. Initialise one `XFuse` instance per phase, call `set_xrd_transforms`, then `forward_sequential`
6. Compute PCA on fused features (`pca.n_components`, `pca.n_samples`)
7. Save outputs to `outputs/<name>/features/`:
   - `<phase>_feats.npy` — fused feature array
   - `<phase>_pca.npy` — PCA-reduced feature array
8. Save figures (always to disk; shown inline if `vis.*` flag is true):
   - `dino_pca.png` if `vis.dino_features`
   - `<phase>_fused.png` if `vis.fused_maps`
   - `<phase>_pca_grid.png` if `vis.pca_components`

### Stage 2 — `run_segment(config: XFuseConfig, threshold: float) -> None`

1. Load `<phase>_pca.npy` from `outputs/<name>/features/` and `xct.npy` from `outputs/<name>/data/`
2. Apply threshold to PCA component 0 per phase → binary mask
3. Save outputs to `outputs/<name>/segment/`:
   - `<phase>_mask.npy`
   - `threshold.txt` — records the threshold value used
4. Save figures if `vis.masks`: `<phase>_mask.png` overlaid on XCT

### Stage 3 — `run_refine(config: XFuseConfig) -> None`

1. Load `<phase>_mask.npy` from `outputs/<name>/segment/` and `xct.npy` from `outputs/<name>/data/`
2. Initialise SAM2 predictor from `sam2.folder` + `sam2.model`
3. Set predictor image (XCT converted to uint8 RGB with autocontrast)
4. For each phase: resize binary mask → logit prompt → `predictor.predict`
5. Save outputs to `outputs/<name>/refine/`:
   - `<phase>_refined_mask.npy`
   - a copy of the YAML config used (`config.yaml`) for reproducibility
6. Save figures if `vis.sam2`: `<phase>_sam2_overlay.png`

### Inter-stage contract

Stages communicate only through files in `outputs/<name>/`. Each stage can be re-run independently. If expected input files are missing, the stage raises a clear `FileNotFoundError` with a message indicating which prior stage must be run first. `xct.npy` is written by Stage 0 and read by Stages 1, 2, and 3 — it is the single source of truth for the image used throughout a run.

---

## CLI Design (`scripts/run_xfuse.py`)

```
python scripts/run_xfuse.py --config <path> --stage <stage> [--threshold <float>] [--vis-all]
```

| Argument | Required | Description |
|---|---|---|
| `--config` | Yes | Path to YAML config file |
| `--stage` | Yes | `data` \| `features` \| `segment` \| `refine` |
| `--threshold` | Only for `segment` | Float threshold applied to PCA component 0 |
| `--vis-all` | No | Override all `vis.*` flags to true without editing YAML |

### Typical HPC workflow

```bash
# Stage 0 — validate and prepare data (CPU only, fast)
python scripts/run_xfuse.py --config configs/zn13x_run1.yaml --stage data

# Inspect outputs/zn13x_run1/data/data_overview.png to confirm data looks correct, then:

# Stage 1 — extract features (GPU required)
python scripts/run_xfuse.py --config configs/zn13x_run1.yaml --stage features

# Inspect outputs/zn13x_run1/features/ figures, then:

# Stage 2
python scripts/run_xfuse.py --config configs/zn13x_run1.yaml --stage segment --threshold 0.2

# Inspect outputs/zn13x_run1/segment/ figures, then:

# Stage 3
python scripts/run_xfuse.py --config configs/zn13x_run1.yaml --stage refine
```

---

## Thin Notebook Driver (`notebooks/run_pipeline.ipynb`)

```python
from x_fuse.config import XFuseConfig
from x_fuse.pipeline import run_data, run_features, run_segment, run_refine

# Option A: load an existing config
config = XFuseConfig.from_yaml("configs/zn13x_run1.yaml")

# Option B: create one in-notebook
config = XFuseConfig(
    name="zn13x_run1",
    dataset_type="diad",
    xct_path="...",
    ...
)
config.to_yaml("configs/zn13x_run1.yaml")

# Stage 0 — validate and inspect data before GPU run
run_data(config)

# Stage 1 — GPU-intensive; only run once data looks correct
run_features(config)

# Inspect PCA figures in outputs/zn13x_run1/features/, then set threshold:
threshold = 0.2
run_segment(config, threshold)

# Stage 3
run_refine(config)
```

---

## Output Directory Structure

```
outputs/
└── zn13x_run1/
    ├── data/                         # Stage 0 outputs
    │   ├── xct.npy                   # transformed XCT image (float32)
    │   ├── Na_xrd.npy                # normalised XRD map (float32)
    │   ├── Zn_xrd.npy
    │   ├── data_summary.txt          # shapes, value ranges, loader used
    │   └── data_overview.png         # if vis.data
    ├── features/                     # Stage 1 outputs
    │   ├── Na_feats.npy
    │   ├── Na_pca.npy
    │   ├── Zn_feats.npy
    │   ├── Zn_pca.npy
    │   ├── dino_pca.png              # if vis.dino_features
    │   ├── Na_fused.png              # if vis.fused_maps
    │   ├── Na_pca_grid.png           # if vis.pca_components
    │   └── ...
    ├── segment/                      # Stage 2 outputs
    │   ├── Na_mask.npy
    │   ├── Zn_mask.npy
    │   ├── threshold.txt
    │   ├── Na_mask.png               # if vis.masks
    │   └── ...
    └── refine/                       # Stage 3 outputs
        ├── Na_refined_mask.npy
        ├── Zn_refined_mask.npy
        ├── config.yaml               # copy of config for reproducibility
        ├── Na_sam2_overlay.png       # if vis.sam2
        └── ...
```

---

## User Guide (`docs/pipeline.md`)

The user guide covers:

1. **Installation** — prerequisites, environment setup, cache directory configuration
2. **Quick start** — end-to-end example from config creation to refined masks
3. **Config reference** — every YAML field documented with type, default, and description
4. **Creating configs in Python** — `XFuseConfig` constructor, `to_yaml()`, `replace()` with worked examples
5. **Running the CLI** — each stage with example commands and expected outputs
6. **Using the notebook driver** — annotated version of `run_pipeline.ipynb`
7. **Dataset types** — `diad` vs `porespy` config differences and which loader functions are called
8. **Data stage in detail** — what is validated, what the `data_summary.txt` contains, how to interpret `data_overview.png`, and common errors with remedies
9. **Visualisation flags** — what each `vis.*` flag controls and when to enable them for HPC vs interactive use
10. **Output files** — description of every file saved by each stage
11. **Extending the pipeline** — how to add a new dataset type (register a loader), a new stage, or new visualisations

---

## Testing

New unit tests in `tests/test_pipeline.py`:

**Config tests:**
- `test_config_roundtrip` — `from_yaml(to_yaml(config))` round-trips without mutation
- `test_config_replace` — `replace()` returns a new instance with only the specified field changed
- `test_config_defaults` — required-only construction sets all defaults correctly

**Stage 0 tests:**
- `test_data_stage_saves_expected_files` — mock loaders, assert `xct.npy`, `<phase>_xrd.npy`, `data_summary.txt` written
- `test_data_stage_missing_path_raises` — assert `FileNotFoundError` with descriptive message when `xct_path` does not exist
- `test_data_stage_shape_mismatch_raises` — assert `ValueError` when XCT and XRD spatial dims are incompatible
- `test_data_stage_vis_data_saves_figure` — assert `data_overview.png` written when `vis.data=True`

**Stage 1–3 tests:**
- `test_features_stage_saves_expected_files` — mock XFuse, assert feature and PCA arrays written
- `test_features_stage_missing_data_raises` — assert `FileNotFoundError` when `data/` absent
- `test_segment_stage_missing_features_raises` — assert `FileNotFoundError` when `features/` absent
- `test_refine_stage_missing_masks_raises` — assert `FileNotFoundError` when `segment/` absent

Integration tests (GPU required, marked `@pytest.mark.gpu`) are out of scope for this spec.
