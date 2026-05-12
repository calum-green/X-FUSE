# X-FUSE Pipeline User Guide

## Contents

1. [Installation](#installation)
2. [Quick start](#quick-start)
3. [Config reference](#config-reference)
4. [Model naming conventions](#model-naming-conventions)
5. [Creating configs in Python](#creating-configs-in-python)
6. [Device management](#device-management)
7. [Running the CLI](#running-the-cli)
8. [Using the notebook driver](#using-the-notebook-driver)
9. [Dataset types](#dataset-types)
10. [Phase naming](#phase-naming)
11. [entry_names in detail](#entry_names-in-detail)
12. [Data stage in detail](#data-stage-in-detail)
13. [Visualisation flags](#visualisation-flags)
14. [Output files](#output-files)
15. [Extending the pipeline](#extending-the-pipeline)

---

## Installation

Prerequisites: Python 3.10+, CUDA-capable GPU (for Stages 1 and 3), uv.

```bash
uv sync
```

Set HuggingFace / Torch cache directories in your config YAML under `environment.torch_cache`
(or leave blank to use the system default).

---

## Quick start

```python
from x_fuse.config import XFuseConfig
from x_fuse.pipeline import run_data, run_features, run_segment, run_refine

config = XFuseConfig(
    name="my_run",
    dataset_type="diad",
    xct_path="/path/to/xct.h5",
    phase_folder="/path/to/phases/",
    phases=["Na", "Zn"],
    entry_names={"Na": "entry/peak at q~1.651", "Zn": "entry/peak at q~1.656"},
    model_path="/path/to/model.pth",
    lib_path="/path/to/dinov3/",
    torch_cache="/path/to/torch_cache/",
    sam2_folder="//path/to/SAM2/",
)
config.to_yaml("configs/my_run.yaml")

run_data(config)        # validate + prepare data
run_features(config)    # GPU: extract features
run_segment(config, threshold=0.2)  # threshold PCA
run_refine(config)      # GPU: SAM2 refinement
```

Outputs are saved to `outputs/my_run/`.

---

## Config reference

| Field | Type | Default | Description |
|---|---|---|---|
| `name` | str | *required* | Run name; used as output subdirectory |
| `output_dir` | str | `"outputs/"` | Base output directory |
| `torch_cache` | str | `""` | Sets HF_HOME and TORCH_HOME env vars |
| `dataset_type` | str | `"diad"` | `"diad"` or `"porespy"` |
| `xct_path` | str\|null | `null` | Path to XCT `.h5` file (diad only) |
| `phase_folder` | str\|null | `null` | Folder containing `.nxs` phase files (diad only) |
| `phases` | list[str] | `["Na","Zn"]` | Phase names; drives all output file naming |
| `entry_names` | dict\|null | `null` | Maps phase → HDF5 entry path (see [entry_names](#entry_names-in-detail)) |
| `sample_idx` | int | `-1` | Index of slice/sample to process |
| `n_samples` | int | `10` | Number of synthetic images to generate (porespy only) |
| `downsample_factor` | int | `10` | XRD downsampling factor (porespy only) |
| `dino_model` | str | `"nope_dv3_vits16plus_1625"` | Model string (see [naming conventions](#model-naming-conventions)) |
| `model_path` | str\|null | `null` | Checkpoint for `nope_dv3_*` / `alibi_dv3_*` |
| `chk_path` | str\|null | `null` | Weights path for `dinov3_*` base models |
| `lib_path` | str\|null | `null` | Path to DINOv3 library for `torch.hub.load` |
| `img_size` | int | `224` | Target image size (applied to both XCT and porespy generation) |
| `stride` | int | `4` | Patch stride for DINOv3 |
| `fusion_method` | str | `"gating"` | `"gating"`, `"learned_gating"`, or `"attention"` |
| `loss_fn` | str | `"bce"` | `"bce"` or `"pearson"` |
| `top_k` | int\|null | `null` | Channels to retain; `null` defaults to `C // 4` |
| `invert` | bool | `false` | Invert XCT and XRD intensities before processing |
| `device` | str\|null | `null` | `null` = auto-detect; `"cuda"` or `"cpu"` to force |
| `shift_distances` | list[int] | `[1,2]` | Pixel shift distances for augmentation |
| `use_flip` | bool | `true` | Include flip augmentations |
| `n_components` | int | `10` | PCA components to compute |
| `n_samples_pca` | int | `5000` | Pixels sampled for PCA fitting |
| `sam2_folder` | str | `""` | Path to SAM2 model directory |
| `sam2_model` | str | `"sam2.1_hiera_small"` | SAM2 model name (without extension) |
| `vis_data` | bool | `true` | Save + show data overview figure |
| `vis_dino_features` | bool | `false` | Save + show raw DINO PCA figure |
| `vis_fused_maps` | bool | `true` | Save + show fused feature maps |
| `vis_pca_components` | bool | `true` | Save + show PCA component grid |
| `vis_masks` | bool | `true` | Save + show binary mask figures |
| `vis_sam2` | bool | `true` | Save + show SAM2 overlay figures |

---

## Model naming conventions

`dv3` is shorthand for DINOv3 throughout the codebase. The pipeline supports three model string formats:

| Format | Example | `model_path` | `chk_path` | `lib_path` |
|---|---|---|---|---|
| `nope_dv3_<arch>[_<id>]` | `nope_dv3_vits16plus_1625` | required | null | required |
| `alibi_dv3_<arch>[_<id>]` | `alibi_dv3_vits16` | required | null | required |
| `dinov3_<arch>` | `dinov3_vits16` | null | required | required |

**Terminology:**
- `nope` = No Positional Encoding variant
- `alibi` = ALiBi (Attention with Linear Biases) positional encoding variant
- `dv3` = DINOv3 base model
- `vits16` = ViT-Small, patch size 16
- `vits16plus` = ViT-Small, patch size 16, extended training

**Currently tested:** ViT-S (`vits16`, `vits16plus`) only. Larger architectures (`vitl16`, `vith16`) should work but are unverified.

**Not supported:** DINOv2 and other architectures (future work).

---

## Creating configs in Python

```python
from x_fuse.config import XFuseConfig

# Minimal construction — all other fields use defaults
config = XFuseConfig(
    name="zn13x_run1",
    dataset_type="diad",
    xct_path="/path/to/xct.h5",
    phase_folder="/path/to/phases/",
    phases=["Na", "Zn"],
    entry_names={"Na": "entry/peak at q~1.651", "Zn": "entry/peak at q~1.656"},
    model_path="/path/to/model.pth",
    lib_path="/path/to/dinov3/",
    torch_cache="/path/to/torch_cache/",
    sam2_folder="//path/to/SAM2/",
)

# Save to YAML
config.to_yaml("configs/zn13x_run1.yaml")

# Load from YAML
config = XFuseConfig.from_yaml("configs/zn13x_run1.yaml")

# Create experiment variants without mutating the original
for method in ["gating", "attention"]:
    cfg = config.replace(name=f"zn13x_{method}", fusion_method=method)
    cfg.to_yaml(f"configs/zn13x_{method}.yaml")
```

`replace()` returns a new `XFuseConfig`; the original is unchanged.

---

## Device management

The pipeline automatically selects the best available device:

- If `model.device` is set to `"cuda"` or `"cpu"` in the config, that device is used.
- If `model.device` is `null`, CUDA is used if `torch.cuda.is_available()`, otherwise CPU.

**Stages 0 and 2** (data prep and segmentation) run on CPU and do not perform device detection.

**Stages 1 and 3** (features and SAM2 refinement) resolve the device at startup and print it:
```
[run_features] device: cuda
```

This line is visible in HPC job logs and confirms which device was selected.

Feature tensors are moved back to CPU before being saved to `.npy` files. This frees GPU memory between phases and means the `outputs/` directory contains only CPU numpy arrays regardless of which device was used.

To force CPU (e.g. for debugging on a machine without a GPU):
```yaml
model:
  device: "cpu"
```

---

## Running the CLI

```bash
# Stage 0 — CPU only, fast
python scripts/run_xfuse.py --config configs/my_run.yaml --stage data

# Inspect outputs/my_run/data/data_overview.png

# Stage 1 — GPU required
python scripts/run_xfuse.py --config configs/my_run.yaml --stage features

# Inspect PCA grid in outputs/my_run/features/

# Stage 2 — CPU only
python scripts/run_xfuse.py --config configs/my_run.yaml --stage segment --threshold 0.2

# Inspect outputs/my_run/segment/ figures

# Stage 3 — GPU required
python scripts/run_xfuse.py --config configs/my_run.yaml --stage refine
```

Enable all visualisations without editing the YAML:
```bash
python scripts/run_xfuse.py --config configs/my_run.yaml --stage data --vis-all
```

---

## Using the notebook driver

Open `notebooks/run_pipeline.ipynb`. The notebook has one cell per stage:

```python
from x_fuse.config import XFuseConfig
from x_fuse.pipeline import run_data, run_features, run_segment, run_refine

# Load an existing config or create one inline (see cells in the notebook)
config = XFuseConfig.from_yaml("configs/example_diad.yaml")

run_data(config)
run_features(config)
run_segment(config, threshold=0.2)
run_refine(config)
```

Figures are always saved to `outputs/<name>/`. When run in Jupyter with `vis_*: true`, figures also display inline via `plt.show()`.

---

## Dataset types

| Field | `"diad"` | `"porespy"` |
|---|---|---|
| `xct_path` | required (.h5 file) | ignored |
| `phase_folder` | required | ignored |
| `entry_names` | optional (warns if absent) | ignored |
| `n_samples` | ignored | number of synthetic images |
| `downsample_factor` | ignored | XRD downsampling factor |

**diad** calls `load_diad_xct_zn13x(xct_path)` for the XCT volume and `load_diad_xrdct` or `load_xrdct_phase` for per-phase XRD data.

**porespy** calls `get_ps_images(img_size)` which generates synthetic blobs + random spheres. Exactly 2 phases must be specified; `phases[0]` maps to blobs, `phases[1]` to spheres.

---

## Phase naming

`config.dataset.phases` drives all output file naming. No phase names are hardcoded anywhere in `pipeline.py`.

| `phases` | Output files |
|---|---|
| `["Na", "Zn"]` | `Na_xrd.npy`, `Zn_xrd.npy`, `Na_pca.npy`, `Zn_mask.npy`, ... |
| `["alpha"]` | `alpha_xrd.npy`, `alpha_pca.npy`, `alpha_mask.npy`, ... |
| `["A", "B", "C"]` | `A_xrd.npy`, `B_xrd.npy`, `C_xrd.npy`, ... |

A single-phase run: `phases: ["Zn"]` processes only that one phase.

---

## entry_names in detail

For DIAD `.nxs` files, each phase string serves two purposes:

1. **Filename filter** — `load_xrdct_phase` lists files in `phase_folder` and keeps those whose name contains the phase string (case-insensitive). Phase `"Na"` matches files like `na_phase_001.nxs`.

2. **HDF5 entry key** — once a file is opened, `entry_names[phase]` is used to read the correct dataset inside the `.nxs` file. For Zn13X: `"Na" → "entry/peak at q~1.651"`.

**To find the correct entry path for a new dataset:**

```python
import h5py
with h5py.File("path/to/phase_file.nxs", "r") as f:
    f.visit(print)  # prints all group/dataset paths
```

Look for the path corresponding to the XRD peak of interest and add it to `entry_names`.

**If `entry_names` is omitted** from the config, the loader falls back to its hardcoded default (`{"default": "data"}`), which may not read the correct dataset. Stage 0 emits a `UserWarning` in this case:

```
UserWarning: entry_names not set in config — loader will use its hardcoded defaults.
Set dataset.entry_names in the YAML to silence this warning.
```

---

## Data stage in detail

Stage 0 (`run_data`) runs validation before loading anything:

| Check | Failure type | Remedy |
|---|---|---|
| `xct_path` exists | `FileNotFoundError: xct_path not found` | Check the path in your YAML |
| `phase_folder` exists | `FileNotFoundError: phase_folder not found` | Check the path in your YAML |
| `model_path` exists | `FileNotFoundError: model_path not found` | Check the path in your YAML |
| `entry_names` absent (diad) | `UserWarning` | Add `entry_names` to YAML |
| XCT/XRD sample is 2D after slice | `ValueError` | Check `sample_idx` is in range |

After validation, outputs are saved to `outputs/<name>/data/`:

- **`xct.npy`** — float32, shape `(img_size, img_size)`, values in `[0, 1]`
- **`<phase>_xrd.npy`** — float32, shape `(H, W)`, values in `[0, 1]`
- **`data_summary.txt`** — dataset type, phase names, array shapes and value ranges
- **`data_overview.png`** — side-by-side XCT + all XRD phase maps (if `vis_data: true`)

Check `data_summary.txt` to confirm loaded shapes look correct before submitting Stage 1 to a GPU queue.

---

## Visualisation flags

All figures are **always saved to disk** regardless of `vis_*` flags. The flags control only whether `plt.show()` is called — useful for inline display in notebooks, suppressed for headless HPC runs.

| Flag | What it controls | Default |
|---|---|---|
| `vis_data` | `data_overview.png` (XCT + XRD side by side) | `true` |
| `vis_dino_features` | `dino_pca.png` (raw DINO features before fusion) | `false` |
| `vis_fused_maps` | `<phase>_fused.png` (channel 0 of fused feature map) | `true` |
| `vis_pca_components` | `<phase>_pca_grid.png` (all n_components PCA panels) | `true` |
| `vis_masks` | `<phase>_mask.png` (binary mask overlaid on XCT) | `true` |
| `vis_sam2` | `<phase>_sam2_overlay.png` (SAM2 refined mask overlaid) | `true` |

On HPC, set all flags to `false` to suppress `plt.show()` calls (figures are still saved).
Use `--vis-all` on the CLI to override all flags to `true` without editing the YAML.

---

## Output files

```
outputs/<run.name>/
├── data/
│   ├── xct.npy                 float32 (H, W) — transformed XCT slice
│   ├── <phase>_xrd.npy         float32 (H, W) — normalised XRD map per phase
│   ├── data_summary.txt        shapes, value ranges, dataset type
│   └── data_overview.png       XCT + XRD side by side (if vis_data)
├── features/
│   ├── <phase>_feats.npy       float32 (C, H, W) — fused DINO features per phase
│   ├── <phase>_pca.npy         float32 (H*W, n_components) — PCA-reduced features
│   ├── dino_pca.png            (if vis_dino_features)
│   ├── <phase>_fused.png       (if vis_fused_maps)
│   └── <phase>_pca_grid.png    (if vis_pca_components)
├── segment/
│   ├── <phase>_mask.npy        uint8 (H, W) — binary mask (0/1)
│   ├── threshold.txt           the threshold value used
│   └── <phase>_mask.png        (if vis_masks)
└── refine/
    ├── <phase>_refined_mask.npy bool (H, W) — SAM2-refined mask
    ├── config.yaml             copy of config used for reproducibility
    └── <phase>_sam2_overlay.png (if vis_sam2)
```

---

## Extending the pipeline

### Adding a new dataset type

1. Write a loader function (or use an existing one from `loaders.py`) that returns `(xct_array, xrd_dict)`.
2. In `pipeline.py`, add a branch in `_load_raw_data`:

```python
if config.dataset_type == "my_new_type":
    xct_raw = my_loader(config.xct_path)
    xrd_raw = {p: my_xrd_loader(config.phase_folder, p) for p in config.phases}
    return xct_raw, xrd_raw
```

3. Add `"my_new_type"` to the validation logic in `_validate_paths`.
4. Add an example YAML to `configs/`.

### Adding a new stage

1. Add a function `run_<stage>(config: XFuseConfig, ...) -> None` to `pipeline.py`.
2. Add a new `elif args.stage == "<stage>":` branch in `scripts/run_xfuse.py`.
3. Add the stage name to the `choices` list in the argparse `--stage` argument.

### Adding new visualisations

1. Add a `vis_<name>: bool = True` field to `XFuseConfig` in `config.py`.
2. Add the new field to `from_yaml`, `to_yaml`, and the vis dict in `to_yaml`.
3. Call `fig.savefig(...)` unconditionally; wrap `plt.show()` with `if config.vis_<name>:`.
4. Add the flag to the `replace(...)` call in the `--vis-all` branch of the CLI.
