# Modular Pipeline Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Convert the notebook-based X-FUSE workflow into a four-stage, YAML-driven pipeline with a CLI entry point, thin notebook driver, and full user guide.

**Architecture:** `x_fuse/config.py` holds a flat `XFuseConfig` dataclass (PyYAML-backed); `x_fuse/pipeline.py` implements `run_data / run_features / run_segment / run_refine`; `scripts/run_xfuse.py` wraps these with argparse. Stages communicate only through files in `outputs/<name>/`. Utility functions are extracted from `notebooks/ps_dv3.ipynb` into `x_fuse/utils.py`.

**Tech Stack:** Python 3.10+, PyYAML, NumPy, PyTorch, hr_dv2, SAM2, porespy, pytest/unittest.mock

---

## File Map

| Action | Path | Responsibility |
|---|---|---|
| Modify | `x_fuse/utils.py` | Add 6 extracted utility functions |
| Create | `x_fuse/config.py` | `XFuseConfig` dataclass |
| Create | `x_fuse/pipeline.py` | Four stage functions + private helpers |
| Create | `scripts/run_xfuse.py` | argparse CLI |
| Create | `configs/example_diad.yaml` | Example config for DIAD data |
| Create | `configs/example_porespy.yaml` | Example config for PoreSpy data |
| Create | `notebooks/run_pipeline.ipynb` | Thin notebook driver |
| Create | `docs/pipeline.md` | User guide |
| Create | `tests/test_pipeline.py` | Unit tests for config + all stages |
| Modify | `pyproject.toml` | Add `pyyaml` dependency |

---

## Task 1: Add PyYAML dependency

**Files:**
- Modify: `pyproject.toml`

- [ ] **Step 1: Add pyyaml to project dependencies**

In `pyproject.toml`, add `"pyyaml>=6.0"` to the `dependencies` list:

```toml
dependencies = [
    "clip>=0.2.0",
    "featup",
    "flake8>=7.3.0",
    "h5py>=3.16.0",
    "hr_dv2",
    "ipykernel>=7.2.0",
    "jupyter>=1.1.1",
    "open-clip-torch>=3.3.0",
    "opencv-python>=4.13.0.92",
    "porespy>=3.0.2",
    "pytest>=9.0.3",
    "pyyaml>=6.0",
    "sam2>=1.1.0",
    "termcolor>=3.3.0",
    "torch<=2.6",
]
```

- [ ] **Step 2: Sync the lock file**

```bash
uv sync
```

Expected: lock file updated, no errors.

- [ ] **Step 3: Verify import**

```bash
python -c "import yaml; print(yaml.__version__)"
```

Expected: prints a version string like `6.0.2`.

- [ ] **Step 4: Commit**

```bash
git add pyproject.toml uv.lock
git commit -m "chore: add pyyaml dependency"
```

---

## Task 2: Extract utility functions into utils.py

**Files:**
- Modify: `x_fuse/utils.py`
- Create: `tests/test_pipeline.py` (initial stub with utility tests)

These six functions are defined inline in `notebooks/ps_dv3.ipynb` and must be importable from `x_fuse.utils`. The `auto_threshold` function has a bug in the notebook (uninitialized `best_score`/`best_t` if no sweep iteration improves `init_score`) — fixed below.

- [ ] **Step 1: Write failing tests for pure utility functions**

Create `tests/test_pipeline.py`:

```python
import numpy as np
import pytest
from x_fuse.utils import invert_image, xct_contrast, overlay_mask


def test_invert_image_float():
    img = np.array([[0.0, 0.5], [1.0, 0.25]], dtype=np.float32)
    result = invert_image(img)
    expected = np.array([[1.0, 0.5], [0.0, 0.75]], dtype=np.float32)
    np.testing.assert_allclose(result, expected)


def test_invert_image_preserves_shape():
    img = np.random.rand(64, 64).astype(np.float32)
    assert invert_image(img).shape == img.shape


def test_xct_contrast_empty_mask_returns_zero():
    pca = np.zeros((4, 4), dtype=np.float32)
    xct = np.random.rand(4, 4).astype(np.float32)
    assert xct_contrast(1.0, pca, xct) == 0.0


def test_xct_contrast_full_mask_returns_zero():
    pca = np.ones((4, 4), dtype=np.float32)
    xct = np.random.rand(4, 4).astype(np.float32)
    assert xct_contrast(0.0, pca, xct) == 0.0


def test_xct_contrast_returns_positive():
    pca = np.array([[0.0, 0.0, 1.0, 1.0]] * 4, dtype=np.float32)
    xct = np.array([[0.1, 0.1, 0.9, 0.9]] * 4, dtype=np.float32)
    score = xct_contrast(0.5, pca, xct)
    assert score > 0.0


def test_overlay_mask_shape():
    xct = np.random.rand(32, 32).astype(np.float32)
    mask = np.zeros((32, 32), dtype=bool)
    mask[10:20, 10:20] = True
    result = overlay_mask(xct, mask)
    assert result.shape == (32, 32, 3)


def test_overlay_mask_clipped():
    xct = np.ones((8, 8), dtype=np.float32)
    mask = np.ones((8, 8), dtype=bool)
    result = overlay_mask(xct, mask)
    assert result.max() <= 1.0
    assert result.min() >= 0.0
```

- [ ] **Step 2: Run tests — confirm they fail**

```bash
pytest tests/test_pipeline.py -v
```

Expected: `ImportError` or `AttributeError` — functions not yet defined.

- [ ] **Step 3: Add utility functions to utils.py**

At the top of `x_fuse/utils.py`, add `ImageOps` to the PIL import and `Any` to typing:

```python
from PIL import Image, ImageOps
from typing import Any
```

Then append the following six functions to the bottom of `x_fuse/utils.py`:

```python
def invert_image(image: np.ndarray) -> np.ndarray:
    return np.ones_like(image) - image


def load_img(img: np.ndarray, transform) -> tuple:
    unnormalize = transforms.Normalize(
        mean=(-0.485 / 0.229, -0.456 / 0.224, -0.406 / 0.225),
        std=(1 / 0.229, 1 / 0.224, 1 / 0.225),
    )
    img_uint8 = (img * 255).astype(np.uint8)
    image = ImageOps.autocontrast(Image.fromarray(img_uint8).convert("RGB"))
    tensor = transform(image)
    trans_img = transforms.ToPILImage()(unnormalize(tensor))
    return tensor, trans_img


def overlay_mask(
    xct: np.ndarray,
    mask: np.ndarray,
    color: tuple = (1, 0.2, 0.2),
    alpha: float = 0.4,
) -> np.ndarray:
    mask = mask.astype(bool)
    rgb = np.stack([xct] * 3, axis=-1)
    overlay = rgb.copy()
    overlay[mask] = (1 - alpha) * rgb[mask] + alpha * np.array(color)
    return np.clip(overlay, 0.0, 1.0)


def xct_contrast(
    threshold: float, pca_component: np.ndarray, xct_img: np.ndarray
) -> float:
    mask = pca_component > threshold
    if mask.sum() == 0 or mask.all():
        return 0.0
    inside = xct_img[mask].mean()
    outside = xct_img[~mask].mean()
    return float(abs(inside - outside))


def get_SAM2_score(
    pca_comp: np.ndarray,
    img_size: int,
    predictor: Any,
    threshold: float,
    get_mask: bool = False,
):
    binary_map = pca_comp[:, 0].reshape(img_size, img_size) > threshold
    mask_input = cv2.resize(binary_map.astype(np.float32), (256, 256))[None]
    mask_input = (mask_input * 2 - 1) * 10
    with torch.inference_mode():
        if not get_mask:
            _, scores, _ = predictor.predict(
                mask_input=mask_input, multimask_output=False
            )
        else:
            masks, scores, _ = predictor.predict(
                mask_input=mask_input, multimask_output=False
            )
    return float(scores[0]) if not get_mask else (masks, float(scores[0]))


def auto_threshold(
    pca_comp: np.ndarray,
    img_size: int,
    predictor: Any,
    n_iter: int = 10,
    lr: float = 0.1,
) -> tuple:
    eps = 0.01 * (pca_comp[:, 0].max() - pca_comp[:, 0].min())
    t_range = np.linspace(0, 1, 20)
    # Initialise best before sweep to avoid NameError if no iteration improves score
    best_t = float(t_range[0])
    best_score = get_SAM2_score(pca_comp, img_size, predictor, best_t)
    for t in t_range:
        score = get_SAM2_score(pca_comp, img_size, predictor, float(t))
        if score > best_score:
            best_score = score
            best_t = float(t)
    t = best_t
    history = [(t, best_score)]
    for _ in range(n_iter):
        grad = (
            get_SAM2_score(pca_comp, img_size, predictor, t + eps)
            - get_SAM2_score(pca_comp, img_size, predictor, t - eps)
        ) / (2 * eps)
        t = float(np.clip(t - lr * grad, 0.0, 1.0))
        score = get_SAM2_score(pca_comp, img_size, predictor, t)
        if score > best_score:
            best_score = score
            best_t = t
            history.append((t, score))
        if abs(score - best_score) < 1e-5:
            break
    best_mask = get_SAM2_score(
        pca_comp, img_size, predictor, best_t, get_mask=True
    )[0][0].astype(bool)
    return best_mask, best_t, best_score, history
```

- [ ] **Step 4: Run tests — confirm they pass**

```bash
pytest tests/test_pipeline.py -v
```

Expected: 8 tests PASS.

- [ ] **Step 5: Run full test suite to check for regressions**

```bash
pytest --ignore=tests/test_fusion.py -v
```

Expected: all existing tests pass (fusion tests may require GPU, skip if needed).

- [ ] **Step 6: Commit**

```bash
git add x_fuse/utils.py tests/test_pipeline.py
git commit -m "feat: extract utility functions from notebook into utils.py"
```

---

## Task 3: Implement XFuseConfig

**Files:**
- Create: `x_fuse/config.py`
- Modify: `tests/test_pipeline.py`

- [ ] **Step 1: Write failing config tests**

Append to `tests/test_pipeline.py`:

```python
import tempfile
import os
from x_fuse.config import XFuseConfig


def minimal_config(name="test_run") -> XFuseConfig:
    return XFuseConfig(name=name)


def test_config_defaults():
    cfg = minimal_config()
    assert cfg.dataset_type == "diad"
    assert cfg.fusion_method == "gating"
    assert cfg.img_size == 224
    assert cfg.stride == 4
    assert cfg.vis_data is True
    assert cfg.vis_dino_features is False
    assert cfg.device is None
    assert cfg.phases == ["Na", "Zn"]


def test_config_roundtrip(tmp_path):
    cfg = XFuseConfig(
        name="roundtrip",
        dataset_type="porespy",
        phases=["alpha", "beta"],
        img_size=128,
        invert=True,
    )
    yaml_path = str(tmp_path / "cfg.yaml")
    cfg.to_yaml(yaml_path)
    loaded = XFuseConfig.from_yaml(yaml_path)
    assert loaded.name == "roundtrip"
    assert loaded.dataset_type == "porespy"
    assert loaded.phases == ["alpha", "beta"]
    assert loaded.img_size == 128
    assert loaded.invert is True


def test_config_replace_does_not_mutate():
    cfg = minimal_config("original")
    cfg2 = cfg.replace(name="copy", fusion_method="attention")
    assert cfg.name == "original"
    assert cfg.fusion_method == "gating"
    assert cfg2.name == "copy"
    assert cfg2.fusion_method == "attention"


def test_config_output_path():
    cfg = XFuseConfig(name="myrun", output_dir="results/")
    assert str(cfg.output_path) == "results/myrun"


def test_config_resolve_device_override():
    cfg = XFuseConfig(name="x", device="cpu")
    assert cfg.resolve_device() == "cpu"


def test_config_resolve_device_auto():
    import torch
    cfg = XFuseConfig(name="x")
    device = cfg.resolve_device()
    expected = "cuda" if torch.cuda.is_available() else "cpu"
    assert device == expected
```

- [ ] **Step 2: Run tests — confirm they fail**

```bash
pytest tests/test_pipeline.py::test_config_defaults -v
```

Expected: `ImportError: cannot import name 'XFuseConfig'`.

- [ ] **Step 3: Create x_fuse/config.py**

```python
from dataclasses import dataclass, field
from dataclasses import replace as _replace
from pathlib import Path
from typing import Optional
import yaml


@dataclass
class XFuseConfig:
    # required
    name: str

    # run
    output_dir: str = "outputs/"

    # environment
    torch_cache: str = ""

    # dataset
    dataset_type: str = "diad"
    xct_path: Optional[str] = None
    phase_folder: Optional[str] = None
    phases: list = field(default_factory=lambda: ["Na", "Zn"])
    entry_names: Optional[dict] = None
    sample_idx: int = -1
    n_samples: int = 10          # porespy only
    downsample_factor: int = 10  # porespy only

    # model
    dino_model: str = "nope_dv3_vits16plus_1625"
    model_path: Optional[str] = None
    chk_path: Optional[str] = None
    lib_path: Optional[str] = None
    img_size: int = 224
    stride: int = 4
    fusion_method: str = "gating"
    loss_fn: str = "bce"
    top_k: Optional[int] = None
    invert: bool = False
    device: Optional[str] = None

    # augmentation
    shift_distances: list = field(default_factory=lambda: [1, 2])
    use_flip: bool = True

    # pca
    n_components: int = 10
    n_samples_pca: int = 5000

    # sam2
    sam2_folder: str = ""
    sam2_model: str = "sam2.1_hiera_small"

    # vis
    vis_data: bool = True
    vis_dino_features: bool = False
    vis_fused_maps: bool = True
    vis_pca_components: bool = True
    vis_masks: bool = True
    vis_sam2: bool = True

    @classmethod
    def from_yaml(cls, path: str) -> "XFuseConfig":
        with open(path) as f:
            d = yaml.safe_load(f)
        run = d.get("run", {})
        env = d.get("environment", {})
        ds = d.get("dataset", {})
        model = d.get("model", {})
        aug = d.get("augmentation", {})
        pca = d.get("pca", {})
        sam2 = d.get("sam2", {})
        vis = d.get("vis", {})
        return cls(
            name=run["name"],
            output_dir=run.get("output_dir", "outputs/"),
            torch_cache=env.get("torch_cache", ""),
            dataset_type=ds.get("type", "diad"),
            xct_path=ds.get("xct_path"),
            phase_folder=ds.get("phase_folder"),
            phases=ds.get("phases", ["Na", "Zn"]),
            entry_names=ds.get("entry_names"),
            sample_idx=ds.get("sample_idx", -1),
            n_samples=ds.get("n_samples", 10),
            downsample_factor=ds.get("downsample_factor", 10),
            dino_model=model.get("dino_model", "nope_dv3_vits16plus_1625"),
            model_path=model.get("model_path"),
            chk_path=model.get("chk_path"),
            lib_path=model.get("lib_path"),
            img_size=model.get("img_size", 224),
            stride=model.get("stride", 4),
            fusion_method=model.get("fusion_method", "gating"),
            loss_fn=model.get("loss_fn", "bce"),
            top_k=model.get("top_k"),
            invert=model.get("invert", False),
            device=model.get("device"),
            shift_distances=aug.get("shift_distances", [1, 2]),
            use_flip=aug.get("use_flip", True),
            n_components=pca.get("n_components", 10),
            n_samples_pca=pca.get("n_samples", 5000),
            sam2_folder=sam2.get("folder", ""),
            sam2_model=sam2.get("model", "sam2.1_hiera_small"),
            vis_data=vis.get("data", True),
            vis_dino_features=vis.get("dino_features", False),
            vis_fused_maps=vis.get("fused_maps", True),
            vis_pca_components=vis.get("pca_components", True),
            vis_masks=vis.get("masks", True),
            vis_sam2=vis.get("sam2", True),
        )

    def to_yaml(self, path: str) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        d = {
            "run": {"name": self.name, "output_dir": self.output_dir},
            "environment": {"torch_cache": self.torch_cache},
            "dataset": {
                "type": self.dataset_type,
                "xct_path": self.xct_path,
                "phase_folder": self.phase_folder,
                "phases": self.phases,
                "entry_names": self.entry_names,
                "sample_idx": self.sample_idx,
                "n_samples": self.n_samples,
                "downsample_factor": self.downsample_factor,
            },
            "model": {
                "dino_model": self.dino_model,
                "model_path": self.model_path,
                "chk_path": self.chk_path,
                "lib_path": self.lib_path,
                "img_size": self.img_size,
                "stride": self.stride,
                "fusion_method": self.fusion_method,
                "loss_fn": self.loss_fn,
                "top_k": self.top_k,
                "invert": self.invert,
                "device": self.device,
            },
            "augmentation": {
                "shift_distances": self.shift_distances,
                "use_flip": self.use_flip,
            },
            "pca": {"n_components": self.n_components, "n_samples": self.n_samples_pca},
            "sam2": {"folder": self.sam2_folder, "model": self.sam2_model},
            "vis": {
                "data": self.vis_data,
                "dino_features": self.vis_dino_features,
                "fused_maps": self.vis_fused_maps,
                "pca_components": self.vis_pca_components,
                "masks": self.vis_masks,
                "sam2": self.vis_sam2,
            },
        }
        with open(path, "w") as f:
            yaml.dump(d, f, default_flow_style=False, sort_keys=False)

    def replace(self, **kwargs) -> "XFuseConfig":
        return _replace(self, **kwargs)

    @property
    def output_path(self) -> Path:
        return Path(self.output_dir) / self.name

    def resolve_device(self) -> str:
        if self.device is not None:
            return self.device
        import torch
        return "cuda" if torch.cuda.is_available() else "cpu"
```

- [ ] **Step 4: Run tests — confirm they pass**

```bash
pytest tests/test_pipeline.py -k "config" -v
```

Expected: 6 config tests PASS.

- [ ] **Step 5: Commit**

```bash
git add x_fuse/config.py tests/test_pipeline.py
git commit -m "feat: add XFuseConfig dataclass with from_yaml, to_yaml, replace"
```

---

## Task 4: Implement run_data (Stage 0)

**Files:**
- Create: `x_fuse/pipeline.py` (initial skeleton + run_data)
- Modify: `tests/test_pipeline.py`

- [ ] **Step 1: Write failing tests for run_data**

Append to `tests/test_pipeline.py`:

```python
from unittest.mock import patch, MagicMock
from x_fuse.pipeline import run_data
from x_fuse.config import XFuseConfig


def _make_diad_config(tmp_path, name="test") -> XFuseConfig:
    return XFuseConfig(
        name=name,
        output_dir=str(tmp_path),
        dataset_type="diad",
        xct_path=str(tmp_path / "xct.h5"),
        phase_folder=str(tmp_path / "phases"),
        phases=["Na", "Zn"],
        img_size=56,
        vis_data=False,
    )


@patch("x_fuse.pipeline.load_diad_xrdct")
@patch("x_fuse.pipeline.load_diad_xct_zn13x")
def test_run_data_saves_expected_files(mock_xct, mock_xrd, tmp_path):
    mock_xct.return_value = np.random.rand(56, 56, 5).astype(np.float32)
    mock_xrd.return_value = {
        "Na": np.random.rand(5, 8, 8).astype(np.float32),
        "Zn": np.random.rand(5, 8, 8).astype(np.float32),
    }
    cfg = _make_diad_config(tmp_path)
    run_data(cfg)
    data_dir = tmp_path / "test" / "data"
    assert (data_dir / "xct.npy").exists()
    assert (data_dir / "Na_xrd.npy").exists()
    assert (data_dir / "Zn_xrd.npy").exists()
    assert (data_dir / "data_summary.txt").exists()


@patch("x_fuse.pipeline.load_diad_xrdct")
@patch("x_fuse.pipeline.load_diad_xct_zn13x")
def test_run_data_custom_phases(mock_xct, mock_xrd, tmp_path):
    mock_xct.return_value = np.random.rand(56, 56, 3).astype(np.float32)
    mock_xrd.return_value = {
        "alpha": np.random.rand(3, 8, 8).astype(np.float32),
    }
    cfg = XFuseConfig(
        name="custom",
        output_dir=str(tmp_path),
        dataset_type="diad",
        phases=["alpha"],
        img_size=56,
        vis_data=False,
    )
    run_data(cfg)
    data_dir = tmp_path / "custom" / "data"
    assert (data_dir / "alpha_xrd.npy").exists()
    assert not (data_dir / "Na_xrd.npy").exists()


def test_run_data_missing_xct_path_raises(tmp_path):
    cfg = XFuseConfig(
        name="bad",
        output_dir=str(tmp_path),
        dataset_type="diad",
        xct_path="/nonexistent/path.h5",
        phase_folder=str(tmp_path),
        phases=["Na"],
        vis_data=False,
    )
    with pytest.raises(FileNotFoundError, match="xct_path"):
        run_data(cfg)


@patch("x_fuse.pipeline.load_diad_xrdct")
@patch("x_fuse.pipeline.load_diad_xct_zn13x")
def test_run_data_invert_flag(mock_xct, mock_xrd, tmp_path):
    xct_arr = np.ones((56, 56, 3), dtype=np.float32) * 0.3
    mock_xct.return_value = xct_arr
    mock_xrd.return_value = {"Na": np.ones((3, 8, 8), dtype=np.float32) * 0.4}
    cfg = XFuseConfig(
        name="inv",
        output_dir=str(tmp_path),
        dataset_type="diad",
        phases=["Na"],
        img_size=56,
        invert=True,
        vis_data=False,
    )
    run_data(cfg)
    xct_saved = np.load(tmp_path / "inv" / "data" / "xct.npy")
    # After inversion, values near 0.3 should become near 0.7
    assert xct_saved.mean() > 0.5
```

- [ ] **Step 2: Run tests — confirm they fail**

```bash
pytest tests/test_pipeline.py -k "run_data" -v
```

Expected: `ImportError: cannot import name 'run_data'`.

- [ ] **Step 3: Create x_fuse/pipeline.py with run_data**

```python
import os
import warnings
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import torch
import matplotlib.pyplot as plt
from PIL import Image, ImageOps

import hr_dv2.transform as tr
from hr_dv2.utils import do_single_pca, rescale_pca

from .config import XFuseConfig
from .loaders import load_diad_xct_zn13x, load_diad_xrdct, load_xrdct_phase
from .utils import invert_image, load_img, overlay_mask, xrd_to_tensor, get_ps_images


# ---------------------------------------------------------------------------
# Stage 0
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Stage 1
# ---------------------------------------------------------------------------

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
            do_single_pca(feats_flat, config.n_components, n_samples=config.n_samples_pca)
        )
        np.save(feat_dir / f"{phase}_pca.npy", pcaed.astype(np.float32))

        _save_features_figures(feat_dir, phase, feats_np, pcaed, config)


# ---------------------------------------------------------------------------
# Stage 2
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Stage 3
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _set_cache_env(config: XFuseConfig) -> None:
    if config.torch_cache:
        for var in ("HUGGINGFACE_HUB_CACHE", "TORCH_HOME", "HF_HUB_CACHE", "HF_HOME"):
            os.environ[var] = config.torch_cache


def _validate_paths(config: XFuseConfig) -> None:
    if config.dataset_type == "diad":
        if config.xct_path and not Path(config.xct_path).exists():
            raise FileNotFoundError(
                f"xct_path not found: {config.xct_path}"
            )
        if config.phase_folder and not Path(config.phase_folder).exists():
            raise FileNotFoundError(
                f"phase_folder not found: {config.phase_folder}"
            )
        if config.entry_names is None:
            warnings.warn(
                "entry_names not set in config — loader will use its hardcoded defaults. "
                "Set dataset.entry_names in the YAML to silence this warning.",
                UserWarning,
                stacklevel=3,
            )
    if config.model_path and not Path(config.model_path).exists():
        raise FileNotFoundError(f"model_path not found: {config.model_path}")
    if config.lib_path and not Path(config.lib_path).exists():
        raise FileNotFoundError(f"lib_path not found: {config.lib_path}")


def _load_raw_data(config: XFuseConfig) -> tuple:
    if config.dataset_type == "diad":
        xct_raw = load_diad_xct_zn13x(config.xct_path)
        if config.entry_names:
            phase_arrays = load_xrdct_phase(
                config.phase_folder,
                phases=config.phases,
                shape=(21, 20, 20),
                crop=slice(5, -1),
                entry_names=config.entry_names,
            )
            xrd_raw = dict(zip(config.phases, phase_arrays))
            xrd_raw = {p: np.flip(arr, axis=2).copy() for p, arr in xrd_raw.items()}
        else:
            xrd_raw = load_diad_xrdct(config.phase_folder, phases=config.phases)
        return xct_raw, xrd_raw

    if config.dataset_type == "porespy":
        if len(config.phases) != 2:
            raise ValueError(
                f"porespy dataset requires exactly 2 phases, got {config.phases}"
            )
        gray_imgs, low_A, low_B = get_ps_images(config.img_size)
        xrd_raw = {config.phases[0]: np.stack(low_A), config.phases[1]: np.stack(low_B)}
        return np.stack(gray_imgs), xrd_raw

    raise ValueError(f"Unknown dataset_type: {config.dataset_type!r}")


def _extract_sample(xct_raw: np.ndarray, xrd_raw: dict, config: XFuseConfig) -> tuple:
    # XCT: (H, W, D) → select slice along last axis; or (N, H, W) → index first axis
    if xct_raw.ndim == 3 and xct_raw.shape[0] != xct_raw.shape[1]:
        # (N, H, W) — porespy stack
        xct_sample = xct_raw[config.sample_idx]
    else:
        # (H, W, D) — diad volume, slice along depth axis
        xct_sample = xct_raw[:, :, config.sample_idx]
    # XRD: (N, H, W) → select slice along first axis
    xrd_sample = {p: arr[config.sample_idx] for p, arr in xrd_raw.items()}
    return xct_sample, xrd_sample


def _validate_arrays(xct: np.ndarray, xrd_dict: dict) -> None:
    if xct.ndim != 2:
        raise ValueError(f"XCT sample must be 2D, got shape {xct.shape}")
    for phase, arr in xrd_dict.items():
        if arr.ndim != 2:
            raise ValueError(
                f"XRD sample for phase '{phase}' must be 2D, got shape {arr.shape}"
            )


def _apply_xct_transform(xct: np.ndarray, transform) -> np.ndarray:
    tensor, _ = load_img(xct, transform)
    # Return as HxW float32 numpy array (take first channel of RGB)
    return tensor.permute(1, 2, 0).numpy()[:, :, 0]


def _write_data_summary(out_dir: Path, xct: np.ndarray, xrd_dict: dict, config: XFuseConfig) -> None:
    lines = [
        f"dataset_type: {config.dataset_type}",
        f"phases: {config.phases}",
        f"sample_idx: {config.sample_idx}",
        f"xct shape: {xct.shape}, min: {xct.min():.4f}, max: {xct.max():.4f}",
    ]
    for phase, arr in xrd_dict.items():
        lines.append(
            f"{phase}_xrd shape: {arr.shape}, min: {arr.min():.4f}, max: {arr.max():.4f}"
        )
    (out_dir / "data_summary.txt").write_text("\n".join(lines))


def _save_data_overview(out_dir: Path, xct: np.ndarray, xrd_dict: dict) -> None:
    n_phases = len(xrd_dict)
    fig, axs = plt.subplots(1, 1 + n_phases, figsize=(6 * (1 + n_phases), 5))
    axs[0].imshow(xct, cmap="gray")
    axs[0].set_title("XCT")
    axs[0].axis("off")
    for i, (phase, arr) in enumerate(xrd_dict.items()):
        axs[i + 1].imshow(arr, cmap="gray", interpolation="lanczos")
        axs[i + 1].set_title(f"XRD — {phase}")
        axs[i + 1].axis("off")
    plt.tight_layout()
    fig.savefig(out_dir / "data_overview.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def _check_stage_inputs(directory: Path, filenames: list, prior_stage: str) -> None:
    for fname in filenames:
        p = directory / fname
        if not p.exists():
            raise FileNotFoundError(
                f"Expected '{p}' but it was not found. "
                f"Run --stage {prior_stage} first."
            )


def _save_features_figures(
    feat_dir: Path,
    phase: str,
    feats_np: np.ndarray,
    pcaed: np.ndarray,
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
        h = w = config.img_size
        for i in range(n):
            axs[i].imshow(pcaed[:, i].reshape(h, w), cmap="viridis")
            axs[i].set_title(f"PCA {i + 1}")
            axs[i].axis("off")
        plt.suptitle(f"PCA components — {phase}")
        plt.tight_layout()
        fig.savefig(feat_dir / f"{phase}_pca_grid.png", dpi=150, bbox_inches="tight")
        plt.close(fig)


def _save_mask_figure(
    seg_dir: Path, phase: str, xct: np.ndarray, mask: np.ndarray, threshold: float
) -> None:
    fig, axs = plt.subplots(1, 2, figsize=(10, 5))
    axs[0].imshow(xct, cmap="gray")
    axs[0].set_title("XCT")
    axs[0].axis("off")
    axs[1].imshow(overlay_mask(xct, mask.astype(bool)))
    axs[1].set_title(f"{phase} mask (threshold={threshold:.3f})")
    axs[1].axis("off")
    plt.tight_layout()
    fig.savefig(seg_dir / f"{phase}_mask.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def _save_sam2_figure(
    refine_dir: Path,
    phase: str,
    xct: np.ndarray,
    refined: np.ndarray,
    score: float,
) -> None:
    fig, axs = plt.subplots(1, 2, figsize=(10, 5))
    axs[0].imshow(xct, cmap="gray")
    axs[0].set_title("XCT")
    axs[0].axis("off")
    axs[1].imshow(overlay_mask(xct, refined))
    axs[1].set_title(f"{phase} SAM2 refined (score={score:.3f})")
    axs[1].axis("off")
    plt.tight_layout()
    fig.savefig(refine_dir / f"{phase}_sam2_overlay.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
```

- [ ] **Step 4: Run run_data tests — confirm they pass**

```bash
pytest tests/test_pipeline.py -k "run_data" -v
```

Expected: 4 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add x_fuse/pipeline.py tests/test_pipeline.py
git commit -m "feat: add run_data pipeline stage with validation and data saving"
```

---

## Task 5: Test run_features, run_segment, run_refine

**Files:**
- Modify: `tests/test_pipeline.py`

`pipeline.py` already contains all four stage functions (written in Task 4). This task adds and verifies the remaining tests.

- [ ] **Step 1: Write tests for run_features, run_segment, run_refine**

Append to `tests/test_pipeline.py`:

```python
from x_fuse.pipeline import run_features, run_segment, run_refine


def _write_fake_data(tmp_path, name, phases, img_size=56):
    """Write fake Stage 0 outputs so later stages can load them."""
    data_dir = tmp_path / name / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    xct = np.random.rand(img_size, img_size).astype(np.float32)
    np.save(data_dir / "xct.npy", xct)
    for phase in phases:
        xrd = np.random.rand(8, 8).astype(np.float32)
        np.save(data_dir / f"{phase}_xrd.npy", xrd)
    return xct


def _write_fake_features(tmp_path, name, phases, img_size=56, n_components=3):
    """Write fake Stage 1 outputs so segment stage can load them."""
    feat_dir = tmp_path / name / "features"
    feat_dir.mkdir(parents=True, exist_ok=True)
    for phase in phases:
        pca = np.random.rand(img_size * img_size, n_components).astype(np.float32)
        np.save(feat_dir / f"{phase}_pca.npy", pca)


def _write_fake_masks(tmp_path, name, phases, img_size=56):
    """Write fake Stage 2 outputs so refine stage can load them."""
    seg_dir = tmp_path / name / "segment"
    seg_dir.mkdir(parents=True, exist_ok=True)
    (seg_dir / "threshold.txt").write_text("0.5")
    for phase in phases:
        mask = np.random.randint(0, 2, (img_size, img_size), dtype=np.uint8)
        np.save(seg_dir / f"{phase}_mask.npy", mask)


@patch("x_fuse.pipeline.XFuse")
def test_run_features_saves_expected_files(mock_xfuse_cls, tmp_path):
    phases = ["Na", "Zn"]
    img_size = 56
    n_comp = 3
    _write_fake_data(tmp_path, "run", phases, img_size)

    # Make XFuse.forward_sequential return a plausible tensor
    fake_feats = torch.zeros(1, 384, img_size, img_size)
    mock_net = MagicMock()
    mock_net.forward_sequential.return_value = fake_feats
    mock_xfuse_cls.return_value = mock_net

    cfg = XFuseConfig(
        name="run",
        output_dir=str(tmp_path),
        phases=phases,
        img_size=img_size,
        n_components=n_comp,
        vis_fused_maps=False,
        vis_pca_components=False,
        device="cpu",
    )

    with patch("x_fuse.pipeline.tr.to_numpy", return_value=np.zeros((384, img_size, img_size))), \
         patch("x_fuse.pipeline.tr.flatten", return_value=np.zeros((img_size * img_size, 384))), \
         patch("x_fuse.pipeline.do_single_pca", return_value=np.zeros((img_size * img_size, n_comp))), \
         patch("x_fuse.pipeline.rescale_pca", return_value=np.zeros((img_size * img_size, n_comp))), \
         patch("x_fuse.pipeline.load_img", return_value=(torch.zeros(3, img_size, img_size), None)):
        run_features(cfg)

    feat_dir = tmp_path / "run" / "features"
    for phase in phases:
        assert (feat_dir / f"{phase}_feats.npy").exists()
        assert (feat_dir / f"{phase}_pca.npy").exists()


def test_run_features_missing_data_raises(tmp_path):
    cfg = XFuseConfig(
        name="missing",
        output_dir=str(tmp_path),
        phases=["Na"],
        device="cpu",
    )
    with pytest.raises(FileNotFoundError, match="data"):
        run_features(cfg)


def test_run_segment_saves_masks_and_threshold(tmp_path):
    phases = ["Na", "Zn"]
    img_size = 56
    _write_fake_data(tmp_path, "seg", phases, img_size)
    _write_fake_features(tmp_path, "seg", phases, img_size)

    cfg = XFuseConfig(
        name="seg",
        output_dir=str(tmp_path),
        phases=phases,
        img_size=img_size,
        vis_masks=False,
    )
    run_segment(cfg, threshold=0.3)

    seg_dir = tmp_path / "seg" / "segment"
    assert (seg_dir / "threshold.txt").read_text() == "0.3"
    for phase in phases:
        assert (seg_dir / f"{phase}_mask.npy").exists()
        mask = np.load(seg_dir / f"{phase}_mask.npy")
        assert mask.dtype == np.uint8
        assert set(np.unique(mask)).issubset({0, 1})


def test_run_segment_missing_features_raises(tmp_path):
    cfg = XFuseConfig(
        name="miss",
        output_dir=str(tmp_path),
        phases=["Na"],
    )
    with pytest.raises(FileNotFoundError, match="features"):
        run_segment(cfg, threshold=0.5)


@patch("x_fuse.pipeline.build_sam2")
@patch("x_fuse.pipeline.SAM2ImagePredictor")
def test_run_refine_saves_refined_masks_and_config(mock_predictor_cls, mock_build, tmp_path):
    phases = ["Na"]
    img_size = 56
    _write_fake_data(tmp_path, "ref", phases, img_size)
    _write_fake_masks(tmp_path, "ref", phases, img_size)

    fake_mask = np.ones((1, img_size, img_size), dtype=bool)
    fake_scores = np.array([0.95])
    mock_predictor = MagicMock()
    mock_predictor.predict.return_value = (fake_mask, fake_scores, None)
    mock_predictor_cls.return_value = mock_predictor
    mock_build.return_value = MagicMock()

    cfg = XFuseConfig(
        name="ref",
        output_dir=str(tmp_path),
        phases=phases,
        img_size=img_size,
        vis_sam2=False,
        device="cpu",
    )
    run_refine(cfg)

    refine_dir = tmp_path / "ref" / "refine"
    assert (refine_dir / "Na_refined_mask.npy").exists()
    assert (refine_dir / "config.yaml").exists()


def test_run_refine_missing_segment_raises(tmp_path):
    cfg = XFuseConfig(
        name="miss",
        output_dir=str(tmp_path),
        phases=["Na"],
    )
    with pytest.raises(FileNotFoundError, match="segment"):
        run_refine(cfg)
```

- [ ] **Step 2: Run all pipeline tests**

```bash
pytest tests/test_pipeline.py -v
```

Expected: all tests PASS (some may need `pytest-mock` or standard `unittest.mock` — both are stdlib/installed).

- [ ] **Step 3: Run full test suite**

```bash
pytest -v -m "not gpu"
```

Expected: all non-GPU tests pass.

- [ ] **Step 4: Commit**

```bash
git add tests/test_pipeline.py
git commit -m "test: add full pipeline stage tests (run_features, run_segment, run_refine)"
```

---

## Task 6: Implement CLI

**Files:**
- Create: `scripts/run_xfuse.py`

- [ ] **Step 1: Create scripts/ directory and CLI**

```bash
mkdir -p scripts
```

Create `scripts/run_xfuse.py`:

```python
#!/usr/bin/env python3
"""X-FUSE pipeline CLI.

Usage:
    python scripts/run_xfuse.py --config configs/my_run.yaml --stage data
    python scripts/run_xfuse.py --config configs/my_run.yaml --stage features
    python scripts/run_xfuse.py --config configs/my_run.yaml --stage segment --threshold 0.2
    python scripts/run_xfuse.py --config configs/my_run.yaml --stage refine
    python scripts/run_xfuse.py --config configs/my_run.yaml --stage features --vis-all
"""
import argparse
import sys
from x_fuse.config import XFuseConfig
from x_fuse.pipeline import run_data, run_features, run_segment, run_refine


def main():
    parser = argparse.ArgumentParser(description="X-FUSE pipeline runner")
    parser.add_argument("--config", required=True, help="Path to YAML config file")
    parser.add_argument(
        "--stage",
        required=True,
        choices=["data", "features", "segment", "refine"],
        help="Pipeline stage to run",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=None,
        help="Threshold for --stage segment (required for that stage)",
    )
    parser.add_argument(
        "--vis-all",
        action="store_true",
        help="Override all vis.* flags to True without editing the YAML",
    )
    args = parser.parse_args()

    config = XFuseConfig.from_yaml(args.config)

    if args.vis_all:
        config = config.replace(
            vis_data=True,
            vis_dino_features=True,
            vis_fused_maps=True,
            vis_pca_components=True,
            vis_masks=True,
            vis_sam2=True,
        )

    if args.stage == "data":
        run_data(config)
    elif args.stage == "features":
        run_features(config)
    elif args.stage == "segment":
        if args.threshold is None:
            parser.error("--threshold is required for --stage segment")
        run_segment(config, args.threshold)
    elif args.stage == "refine":
        run_refine(config)


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Verify the CLI is importable without errors**

```bash
python scripts/run_xfuse.py --help
```

Expected output includes:
```
usage: run_xfuse.py [-h] --config CONFIG --stage {data,features,segment,refine}
                    [--threshold THRESHOLD] [--vis-all]
```

- [ ] **Step 3: Commit**

```bash
git add scripts/run_xfuse.py
git commit -m "feat: add CLI entry point scripts/run_xfuse.py"
```

---

## Task 7: Create example YAML configs

**Files:**
- Create: `configs/example_diad.yaml`
- Create: `configs/example_porespy.yaml`

- [ ] **Step 1: Create configs/ directory**

```bash
mkdir -p configs
```

- [ ] **Step 2: Create configs/example_diad.yaml**

```yaml
run:
  name: "zn13x_run1"
  output_dir: "outputs/"

environment:
  torch_cache: "/dls/science/groups/imaging/infuse/XRDCT_Fusion/torch_cache/"

dataset:
  type: "diad"
  xct_path: "/dls/science/groups/imaging/infuse/XRDCT_Fusion/data/Zn13X/xct.h5"
  phase_folder: "/dls/science/groups/imaging/infuse/XRDCT_Fusion/data/Zn13X/phases/"
  phases: ["Na", "Zn"]
  entry_names:
    Na: "entry/peak at q~1.651"
    Zn: "entry/peak at q~1.656"
  sample_idx: -1

model:
  # Supported: "nope_dv3_<arch>", "alibi_dv3_<arch>", "dinov3_<arch>"
  # Only ViT-S architecture is currently tested.
  dino_model: "nope_dv3_vits16plus_1625"
  model_path: "/dls/science/groups/imaging/infuse/XRDCT_Fusion/dinosaw-env/models/dinov3_SMALL_REG_1625_plus_alibi/best_model.pth"
  chk_path: null
  lib_path: "/dls/science/groups/imaging/infuse/DINOv3/dinov3/"
  img_size: 224
  stride: 4
  fusion_method: "gating"
  loss_fn: "bce"
  top_k: null
  invert: false
  device: null

augmentation:
  shift_distances: [1, 2]
  use_flip: true

pca:
  n_components: 10
  n_samples: 5000

sam2:
  folder: "//dls/science/groups/imaging/infuse/SAM2/"
  model: "sam2.1_hiera_small"

vis:
  data: true
  dino_features: false
  fused_maps: true
  pca_components: true
  masks: true
  sam2: true
```

- [ ] **Step 3: Create configs/example_porespy.yaml**

```yaml
run:
  name: "porespy_blobs_run1"
  output_dir: "outputs/"

environment:
  torch_cache: "/dls/science/groups/imaging/infuse/XRDCT_Fusion/torch_cache/"

dataset:
  type: "porespy"
  # porespy generates synthetic data — no file paths needed
  phases: ["blobs", "spheres"]
  sample_idx: -1
  n_samples: 10
  downsample_factor: 10

model:
  dino_model: "nope_dv3_vits16plus_1625"
  model_path: "/dls/science/groups/imaging/infuse/XRDCT_Fusion/dinosaw-env/models/dinov3_SMALL_REG_1625_plus_alibi/best_model.pth"
  chk_path: null
  lib_path: "/dls/science/groups/imaging/infuse/DINOv3/dinov3/"
  img_size: 224
  stride: 4
  fusion_method: "gating"
  loss_fn: "bce"
  top_k: null
  invert: false
  device: null

augmentation:
  shift_distances: [1, 2]
  use_flip: true

pca:
  n_components: 10
  n_samples: 5000

sam2:
  folder: "//dls/science/groups/imaging/infuse/SAM2/"
  model: "sam2.1_hiera_small"

vis:
  data: true
  dino_features: false
  fused_maps: true
  pca_components: true
  masks: true
  sam2: true
```

- [ ] **Step 4: Verify both configs load without error**

```bash
python -c "
from x_fuse.config import XFuseConfig
d = XFuseConfig.from_yaml('configs/example_diad.yaml')
p = XFuseConfig.from_yaml('configs/example_porespy.yaml')
print('diad:', d.name, d.phases)
print('porespy:', p.name, p.phases)
"
```

Expected:
```
diad: zn13x_run1 ['Na', 'Zn']
porespy: porespy_blobs_run1 ['blobs', 'spheres']
```

- [ ] **Step 5: Commit**

```bash
git add configs/
git commit -m "feat: add example YAML configs for diad and porespy datasets"
```

---

## Task 8: Create thin notebook driver

**Files:**
- Create: `notebooks/run_pipeline.ipynb`

- [ ] **Step 1: Create the notebook**

Create `notebooks/run_pipeline.ipynb` with the following cells. Use `jupyter nbformat` or create the JSON directly:

```json
{
 "cells": [
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": ["# X-FUSE Pipeline\n\nRun the four-stage pipeline from a YAML config."]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "from x_fuse.config import XFuseConfig\n",
    "from x_fuse.pipeline import run_data, run_features, run_segment, run_refine"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": ["## Option A: load an existing config"]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "config = XFuseConfig.from_yaml('configs/example_diad.yaml')"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": ["## Option B: create a config inline (overrides Option A)"]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# config = XFuseConfig(\n",
    "#     name='my_run',\n",
    "#     dataset_type='diad',\n",
    "#     xct_path='/path/to/xct.h5',\n",
    "#     phase_folder='/path/to/phases/',\n",
    "#     phases=['Na', 'Zn'],\n",
    "#     entry_names={'Na': 'entry/peak at q~1.651', 'Zn': 'entry/peak at q~1.656'},\n",
    "#     model_path='/path/to/model.pth',\n",
    "#     lib_path='/path/to/dinov3/',\n",
    "#     torch_cache='/path/to/torch_cache/',\n",
    "#     sam2_folder='//path/to/sam2/',\n",
    "# )\n",
    "# config.to_yaml('configs/my_run.yaml')"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": ["## Stage 0 — validate and prepare data (CPU, fast)\n\nInspect `outputs/<name>/data/data_overview.png` before proceeding."]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": ["run_data(config)"]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": ["## Stage 1 — extract features (GPU required)\n\nInspect PCA grid in `outputs/<name>/features/` before setting threshold."]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": ["run_features(config)"]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": ["## Stage 2 — threshold to binary masks\n\nSet `threshold` after inspecting the PCA component 0 range."]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": ["threshold = 0.2\nrun_segment(config, threshold)"]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": ["## Stage 3 — SAM2 refinement"]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": ["run_refine(config)"]
  }
 ],
 "metadata": {
  "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
  "language_info": {"name": "python", "version": "3.12.0"}
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
```

- [ ] **Step 2: Verify the notebook is valid JSON**

```bash
python -c "import json; json.load(open('notebooks/run_pipeline.ipynb')); print('valid')"
```

Expected: `valid`

- [ ] **Step 3: Commit**

```bash
git add notebooks/run_pipeline.ipynb
git commit -m "feat: add thin notebook driver notebooks/run_pipeline.ipynb"
```

---

## Task 9: Write user guide

**Files:**
- Create: `docs/pipeline.md`

- [ ] **Step 1: Create docs/pipeline.md**

Write `docs/pipeline.md` with the following sections. Each section must contain complete, accurate content — no placeholders:

```markdown
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

3. Add `"my_new_type"` to the `choices` list in `_validate_paths`.
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
```

- [ ] **Step 2: Verify the markdown renders (no broken headings or tables)**

```bash
python -c "
text = open('docs/pipeline.md').read()
headings = [l for l in text.splitlines() if l.startswith('#')]
print(f'{len(headings)} headings found')
assert len(headings) >= 15, 'Missing sections'
print('OK')
"
```

Expected: `16 headings found` (or similar, ≥15), then `OK`.

- [ ] **Step 3: Commit**

```bash
git add docs/pipeline.md
git commit -m "docs: add full pipeline user guide docs/pipeline.md"
```

---

## Task 10: Final verification and spec commit

**Files:**
- None new

- [ ] **Step 1: Run the full test suite**

```bash
pytest -v -m "not gpu"
```

Expected: all tests PASS, no failures.

- [ ] **Step 2: Verify CLI help still works**

```bash
python scripts/run_xfuse.py --help
```

Expected: usage printed without error.

- [ ] **Step 3: Verify config roundtrip from both example YAMLs**

```bash
python -c "
from x_fuse.config import XFuseConfig
for path in ['configs/example_diad.yaml', 'configs/example_porespy.yaml']:
    cfg = XFuseConfig.from_yaml(path)
    cfg2 = XFuseConfig.from_yaml(path)
    assert cfg.name == cfg2.name
    assert cfg.phases == cfg2.phases
    print(f'{path}: OK — name={cfg.name}, phases={cfg.phases}')
"
```

Expected:
```
configs/example_diad.yaml: OK — name=zn13x_run1, phases=['Na', 'Zn']
configs/example_porespy.yaml: OK — name=porespy_blobs_run1, phases=['blobs', 'spheres']
```

- [ ] **Step 4: Commit spec update to mark implementation complete**

```bash
git add docs/superpowers/specs/2026-05-12-pipeline-design.md
git commit -m "docs: mark pipeline spec as implemented"
```

---

## Self-Review

**Spec coverage check:**

| Spec requirement | Covered by |
|---|---|
| XFuseConfig with from_yaml, to_yaml, replace | Task 3 |
| run_data with validation + loading + saving | Task 4 |
| run_features with device detection + XFuse + PCA | Task 4 (pipeline.py) |
| run_segment with threshold → binary masks | Task 4 (pipeline.py) |
| run_refine with SAM2 | Task 4 (pipeline.py) |
| CLI with --stage, --threshold, --vis-all | Task 6 |
| Example YAML configs (diad + porespy) | Task 7 |
| Thin notebook driver | Task 8 |
| User guide (all 14 sections) | Task 9 |
| Utility extraction to utils.py | Task 2 |
| Phase naming from config (no hardcoded Na/Zn) | Task 4 (_load_raw_data, _extract_sample) |
| entry_names optional, warning if absent | Task 4 (_validate_paths) |
| Device auto-detect + override | Task 3 (resolve_device) + Task 4 |
| Figures always saved; vis flags control plt.show() | Task 4 (_save_* helpers) |
| Config copy saved in refine/ | Task 4 (run_refine) |
| threshold.txt saved in segment/ | Task 4 (run_segment) |
| data_summary.txt saved in data/ | Task 4 (_write_data_summary) |
| Tests: config roundtrip, replace, defaults | Task 3 tests |
| Tests: stage missing-input FileNotFoundError | Tasks 4+5 tests |
| Tests: run_data saves expected files | Task 4 tests |
| Tests: invert flag applied | Task 4 tests |
| PyYAML dependency | Task 1 |
