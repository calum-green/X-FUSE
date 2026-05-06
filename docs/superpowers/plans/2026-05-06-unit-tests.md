# Unit Test Suite — X-FUSE Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a pytest-based unit test suite for `x_fuse/fusion.py` (XRDCrossAttention, LearnedChannelGating, XRDFusionMethod._direct_correlation_gating) and `x_fuse/utils.py` (xrd_to_tensor, get_multiphase, downsample_xrdct).

**Architecture:** Three files under `tests/`: a `conftest.py` defining 9 parametrized DINO flavour fixtures and shared synthetic tensors, `test_fusion.py` covering the three fusion module targets, and `test_utils.py` covering the utility functions. No model weights are loaded — all fusion tests use `torch.randn`/`torch.rand` synthetic tensors sized `(1, feat_dim, 8, 8)` for speed on CPU.

**Tech Stack:** pytest, torch, numpy, porespy, opencv-python (all in project dev dependencies via `pyproject.toml`)

---

## File Map

| File | Action | Responsibility |
|---|---|---|
| `tests/conftest.py` | Create | 9 DINO flavour profiles; `dino_flavour`, `features`, `xrd_map_4d` fixtures |
| `tests/test_fusion.py` | Create | All fusion module tests |
| `tests/test_utils.py` | Create | All utility function tests |

---

## Task 1: Create `tests/conftest.py`

**Files:**
- Create: `tests/conftest.py`

- [ ] **Step 1: Write conftest.py**

```python
import pytest
import torch

DINO_FLAVOURS = [
    {"id": "dinov2_s_regs",    "feat_dim": 384,  "num_heads": 6,  "patch_size": 14, "n_register_tokens": 4},
    {"id": "dinov2_b_regs",    "feat_dim": 768,  "num_heads": 12, "patch_size": 14, "n_register_tokens": 4},
    {"id": "dinov2_l_regs",    "feat_dim": 1024, "num_heads": 16, "patch_size": 14, "n_register_tokens": 4},
    {"id": "dinov2_g_regs",    "feat_dim": 1536, "num_heads": 16, "patch_size": 14, "n_register_tokens": 4},
    {"id": "dinov2_s_no_regs", "feat_dim": 384,  "num_heads": 6,  "patch_size": 16, "n_register_tokens": 0},
    {"id": "dinov3_s_regs",    "feat_dim": 384,  "num_heads": 6,  "patch_size": 16, "n_register_tokens": 4},
    {"id": "dinov3_s_no_regs", "feat_dim": 384,  "num_heads": 6,  "patch_size": 16, "n_register_tokens": 0},
    {"id": "dinov3_l_regs",    "feat_dim": 1024, "num_heads": 16, "patch_size": 16, "n_register_tokens": 4},
    {"id": "dinov3_7b_regs",   "feat_dim": 4096, "num_heads": 32, "patch_size": 16, "n_register_tokens": 4},
]


@pytest.fixture(params=DINO_FLAVOURS, ids=[f["id"] for f in DINO_FLAVOURS])
def dino_flavour(request):
    return request.param


@pytest.fixture
def features(dino_flavour):
    torch.manual_seed(0)
    return torch.randn(1, dino_flavour["feat_dim"], 8, 8)


@pytest.fixture
def xrd_map_4d():
    torch.manual_seed(42)
    return torch.rand(1, 1, 8, 8)
```

- [ ] **Step 2: Verify conftest loads cleanly**

Run: `pytest tests/ --collect-only 2>&1 | head -30`

Expected: pytest discovers tests (even if there are none yet) with no import errors.

- [ ] **Step 3: Commit**

```bash
git add tests/conftest.py
git commit -m "Add test fixtures: 9 DINO flavour profiles and synthetic tensor fixtures"
```

---

## Task 2: XRDCrossAttention tests

**Files:**
- Create: `tests/test_fusion.py`

- [ ] **Step 1: Write XRDCrossAttention tests**

Create `tests/test_fusion.py` with the following content:

```python
import pytest
import torch
from x_fuse.fusion import XRDCrossAttention, LearnedChannelGating, XRDFusionMethod


# ── XRDCrossAttention ──────────────────────────────────────────────────────────

def test_cross_attn_output_shape(dino_flavour, features, xrd_map_4d):
    model = XRDCrossAttention(
        feat_dim=dino_flavour["feat_dim"], num_heads=dino_flavour["num_heads"]
    )
    out = model(features, xrd_map_4d)
    assert out.shape == features.shape


def test_cross_attn_output_dtype(dino_flavour, features, xrd_map_4d):
    model = XRDCrossAttention(
        feat_dim=dino_flavour["feat_dim"], num_heads=dino_flavour["num_heads"]
    )
    out = model(features, xrd_map_4d)
    assert out.dtype == features.dtype


def test_cross_attn_output_is_tensor(dino_flavour, features, xrd_map_4d):
    model = XRDCrossAttention(
        feat_dim=dino_flavour["feat_dim"], num_heads=dino_flavour["num_heads"]
    )
    out = model(features, xrd_map_4d)
    assert isinstance(out, torch.Tensor)


def test_cross_attn_xrd_2d_input(dino_flavour, features):
    model = XRDCrossAttention(
        feat_dim=dino_flavour["feat_dim"], num_heads=dino_flavour["num_heads"]
    )
    xrd_2d = torch.rand(8, 8)
    out = model(features, xrd_2d)
    assert out.shape == features.shape


def test_cross_attn_xrd_3d_input(dino_flavour, features):
    model = XRDCrossAttention(
        feat_dim=dino_flavour["feat_dim"], num_heads=dino_flavour["num_heads"]
    )
    xrd_3d = torch.rand(1, 8, 8)
    out = model(features, xrd_3d)
    assert out.shape == features.shape
```

- [ ] **Step 2: Run and verify all pass**

Run: `.venv/bin/pytest tests/test_fusion.py -v -k "cross_attn"`

Expected: 45 passed (5 tests × 9 flavours). If any fail, the failure message will name the flavour and assert that failed — investigate before proceeding.

- [ ] **Step 3: Commit**

```bash
git add tests/test_fusion.py
git commit -m "Add XRDCrossAttention unit tests (shape, dtype, tensor type, 2D/3D xrd input)"
```

---

## Task 3: LearnedChannelGating tests

**Files:**
- Modify: `tests/test_fusion.py`

- [ ] **Step 1: Append LearnedChannelGating tests to `tests/test_fusion.py`**

Add after the last XRDCrossAttention test:

```python
# ── LearnedChannelGating ───────────────────────────────────────────────────────

def test_gating_bce_loss_scalar(dino_flavour, features, xrd_map_4d):
    gating = LearnedChannelGating(feat_dim=dino_flavour["feat_dim"], loss_fn="bce")
    loss = gating.compute_loss(features, xrd_map_4d)
    assert loss.ndim == 0
    assert loss.item() > 0


def test_gating_pearson_loss_scalar(dino_flavour, features, xrd_map_4d):
    gating = LearnedChannelGating(feat_dim=dino_flavour["feat_dim"], loss_fn="pearson")
    loss = gating.compute_loss(features, xrd_map_4d)
    assert loss.ndim == 0
    assert 0.0 <= loss.item() <= 2.0


def test_gating_forward_no_mask_passthrough(dino_flavour, features, xrd_map_4d):
    gating = LearnedChannelGating(feat_dim=dino_flavour["feat_dim"])
    result = gating(features, xrd_map_4d)
    assert result is features


def test_gating_select_top_k_channel_count(dino_flavour, features, xrd_map_4d):
    k = 10
    gating = LearnedChannelGating(feat_dim=dino_flavour["feat_dim"])
    gating.select_top_k(k)
    result = gating(features, xrd_map_4d)
    n_nonzero = (result.abs().sum(dim=(0, 2, 3)) > 0).sum().item()
    assert n_nonzero == k


def test_gating_forward_shape(dino_flavour, features, xrd_map_4d):
    gating = LearnedChannelGating(feat_dim=dino_flavour["feat_dim"])
    result = gating(features, xrd_map_4d)
    assert result.shape == features.shape


def test_gating_forward_dtype(dino_flavour, features, xrd_map_4d):
    gating = LearnedChannelGating(feat_dim=dino_flavour["feat_dim"])
    gating.select_top_k(10)
    result = gating(features, xrd_map_4d)
    assert result.dtype == features.dtype
```

- [ ] **Step 2: Run and verify all pass**

Run: `.venv/bin/pytest tests/test_fusion.py -v -k "gating"`

Expected: 54 passed (6 tests × 9 flavours). If `test_gating_pearson_loss_scalar` fails with a value outside `[0, 2]`, check whether `compute_loss` produced a NaN — `torch.randn` features are non-constant so this should not occur.

- [ ] **Step 3: Commit**

```bash
git add tests/test_fusion.py
git commit -m "Add LearnedChannelGating unit tests (loss scalars, passthrough, top-k, shape, dtype)"
```

---

## Task 4: XRDFusionMethod._direct_correlation_gating tests

**Files:**
- Modify: `tests/test_fusion.py`

- [ ] **Step 1: Append _direct_correlation_gating tests to `tests/test_fusion.py`**

Add after the last LearnedChannelGating test:

```python
# ── XRDFusionMethod._direct_correlation_gating ────────────────────────────────

@pytest.mark.parametrize("loss_fn", ["pearson", "bce"])
def test_dcg_output_shape(dino_flavour, features, xrd_map_4d, loss_fn):
    fusion = XRDFusionMethod(
        xrd_img=xrd_map_4d, transform=[], require_grad=False,
        learned_gating=None, spatial_attention=None, loss_fn=loss_fn,
    )
    result = fusion._direct_correlation_gating(features, xrd_map_4d)
    assert result.shape == features.shape


@pytest.mark.parametrize("loss_fn", ["pearson", "bce"])
def test_dcg_output_dtype(dino_flavour, features, xrd_map_4d, loss_fn):
    fusion = XRDFusionMethod(
        xrd_img=xrd_map_4d, transform=[], require_grad=False,
        learned_gating=None, spatial_attention=None, loss_fn=loss_fn,
    )
    result = fusion._direct_correlation_gating(features, xrd_map_4d)
    assert result.dtype == features.dtype


@pytest.mark.parametrize("loss_fn", ["pearson", "bce"])
def test_dcg_output_is_tensor(dino_flavour, features, xrd_map_4d, loss_fn):
    fusion = XRDFusionMethod(
        xrd_img=xrd_map_4d, transform=[], require_grad=False,
        learned_gating=None, spatial_attention=None, loss_fn=loss_fn,
    )
    result = fusion._direct_correlation_gating(features, xrd_map_4d)
    assert isinstance(result, torch.Tensor)


@pytest.mark.parametrize("loss_fn", ["pearson", "bce"])
def test_dcg_gate_values_in_range(dino_flavour, features, xrd_map_4d, loss_fn):
    # Gates are in [0, 1], so |output[c]| <= |features[c]| elementwise.
    fusion = XRDFusionMethod(
        xrd_img=xrd_map_4d, transform=[], require_grad=False,
        learned_gating=None, spatial_attention=None, loss_fn=loss_fn,
    )
    result = fusion._direct_correlation_gating(features, xrd_map_4d)
    assert torch.all(result.abs() <= features.abs() + 1e-6)


@pytest.mark.parametrize("loss_fn", ["pearson", "bce"])
def test_dcg_top_k_zeroing(dino_flavour, features, xrd_map_4d, loss_fn):
    k = 10
    fusion = XRDFusionMethod(
        xrd_img=xrd_map_4d, transform=[], require_grad=False,
        learned_gating=None, spatial_attention=None, loss_fn=loss_fn,
    )
    result = fusion._direct_correlation_gating(features, xrd_map_4d, top_k=k)
    n_nonzero = (result.abs().sum(dim=(0, 2, 3)) > 0).sum().item()
    assert n_nonzero == k


@pytest.mark.parametrize("loss_fn", ["pearson", "bce"])
def test_dcg_default_top_k(dino_flavour, features, xrd_map_4d, loss_fn):
    fusion = XRDFusionMethod(
        xrd_img=xrd_map_4d, transform=[], require_grad=False,
        learned_gating=None, spatial_attention=None, loss_fn=loss_fn,
    )
    result = fusion._direct_correlation_gating(features, xrd_map_4d)
    expected_k = features.shape[1] // 4
    n_nonzero = (result.abs().sum(dim=(0, 2, 3)) > 0).sum().item()
    assert n_nonzero == expected_k


@pytest.mark.parametrize("loss_fn", ["pearson", "bce"])
def test_dcg_xrd_spatial_mismatch(dino_flavour, features, xrd_map_4d, loss_fn):
    fusion = XRDFusionMethod(
        xrd_img=xrd_map_4d, transform=[], require_grad=False,
        learned_gating=None, spatial_attention=None, loss_fn=loss_fn,
    )
    xrd_mismatched = torch.rand(1, 1, 4, 4)
    result = fusion._direct_correlation_gating(features, xrd_mismatched)
    assert result.shape == features.shape
```

- [ ] **Step 2: Run and verify all pass**

Run: `.venv/bin/pytest tests/test_fusion.py -v -k "dcg"`

Expected: 126 passed (7 tests × 9 flavours × 2 loss functions). If `test_dcg_top_k_zeroing` fails with `n_nonzero != 10`, check whether two channels share the same gate value at the threshold boundary — this is theoretically possible but statistically negligible with `torch.randn` features.

- [ ] **Step 3: Run the full fusion test file**

Run: `.venv/bin/pytest tests/test_fusion.py -v`

Expected: 225 passed total (45 + 54 + 126).

- [ ] **Step 4: Commit**

```bash
git add tests/test_fusion.py
git commit -m "Add XRDFusionMethod._direct_correlation_gating unit tests (shape, dtype, gate range, top-k, spatial mismatch)"
```

---

## Task 5: Utility function tests

**Files:**
- Create: `tests/test_utils.py`

- [ ] **Step 1: Write `tests/test_utils.py`**

```python
import pytest
import torch
import numpy as np
import porespy as ps
from x_fuse.utils import xrd_to_tensor, get_multiphase, downsample_xrdct


# ── xrd_to_tensor ──────────────────────────────────────────────────────────────

@pytest.fixture
def xrd_np():
    rng = np.random.default_rng(0)
    return rng.integers(0, 256, (64, 64), dtype=np.uint8)


def test_xrd_to_tensor_is_tensor(xrd_np):
    result = xrd_to_tensor(xrd_np, device="cpu")
    assert isinstance(result, torch.Tensor)


def test_xrd_to_tensor_shape(xrd_np):
    result = xrd_to_tensor(xrd_np, device="cpu")
    assert result.shape == (1, 1, 64, 64)


def test_xrd_to_tensor_dtype(xrd_np):
    result = xrd_to_tensor(xrd_np, device="cpu")
    assert result.dtype == torch.float32


def test_xrd_to_tensor_range(xrd_np):
    result = xrd_to_tensor(xrd_np, device="cpu")
    assert result.min().item() >= 0.0
    assert result.max().item() <= 1.0


def test_xrd_to_tensor_device(xrd_np):
    result = xrd_to_tensor(xrd_np, device="cpu")
    assert result.device == torch.device("cpu")


# ── get_multiphase ─────────────────────────────────────────────────────────────

@pytest.fixture
def multiphase_args():
    im1_kwargs = {"shape": [64, 64], "blobiness": 1.0, "porosity": 0.5}
    im2_kwargs = {"r": 5, "clearance": 1}
    return ps.generators.blobs, im1_kwargs, ps.generators.random_spheres, im2_kwargs


def test_get_multiphase_return_types(multiphase_args):
    im1, im1_kw, im2, im2_kw = multiphase_args
    gray, phase_A, phase_B = get_multiphase(im1, im1_kw, im2, im2_kw, extract_spheres=True, dataset_size=2)
    assert all(isinstance(x, np.ndarray) for x in gray)
    assert all(isinstance(x, np.ndarray) for x in phase_A)
    assert all(isinstance(x, np.ndarray) for x in phase_B)


def test_get_multiphase_dataset_size(multiphase_args):
    im1, im1_kw, im2, im2_kw = multiphase_args
    gray, phase_A, phase_B = get_multiphase(im1, im1_kw, im2, im2_kw, extract_spheres=True, dataset_size=3)
    assert len(gray) == 3
    assert len(phase_A) == 3
    assert len(phase_B) == 3


def test_get_multiphase_mask_binary(multiphase_args):
    im1, im1_kw, im2, im2_kw = multiphase_args
    _, phase_A, phase_B = get_multiphase(im1, im1_kw, im2, im2_kw, extract_spheres=True, dataset_size=2)
    for mask in phase_A + phase_B:
        assert set(np.unique(mask)).issubset({0.0, 1.0})


def test_get_multiphase_gray_range(multiphase_args):
    im1, im1_kw, im2, im2_kw = multiphase_args
    gray, _, _ = get_multiphase(im1, im1_kw, im2, im2_kw, extract_spheres=True, dataset_size=2)
    for img in gray:
        assert img.min() >= 0.0
        assert img.max() <= 1.0


def test_get_multiphase_shape_consistency(multiphase_args):
    im1, im1_kw, im2, im2_kw = multiphase_args
    gray, phase_A, phase_B = get_multiphase(im1, im1_kw, im2, im2_kw, extract_spheres=True, dataset_size=3)
    shapes = [x.shape for x in gray + phase_A + phase_B]
    assert len(set(shapes)) == 1


# ── downsample_xrdct ───────────────────────────────────────────────────────────

@pytest.fixture
def downsample_inputs():
    rng = np.random.default_rng(1)
    gray   = [rng.random((64, 64)).astype(np.float32) for _ in range(3)]
    phase_A = [rng.random((64, 64)).astype(np.float32) for _ in range(3)]
    phase_B = [rng.random((64, 64)).astype(np.float32) for _ in range(3)]
    return gray, phase_A, phase_B


def test_downsample_output_shape(downsample_inputs):
    gray, phase_A, phase_B = downsample_inputs
    _, ds_A, ds_B = downsample_xrdct(gray, phase_A, phase_B, factor=4)
    assert ds_A[0].shape == (16, 16)
    assert ds_B[0].shape == (16, 16)


def test_downsample_output_dtype(downsample_inputs):
    gray, phase_A, phase_B = downsample_inputs
    _, ds_A, ds_B = downsample_xrdct(gray, phase_A, phase_B, factor=4)
    assert ds_A[0].dtype == np.float32
    assert ds_B[0].dtype == np.float32


def test_downsample_list_lengths(downsample_inputs):
    gray, phase_A, phase_B = downsample_inputs
    _, ds_A, ds_B = downsample_xrdct(gray, phase_A, phase_B, factor=2)
    assert len(ds_A) == 3
    assert len(ds_B) == 3
```

- [ ] **Step 2: Run and verify all pass**

Run: `.venv/bin/pytest tests/test_utils.py -v`

Expected: 13 passed. The `get_multiphase` tests invoke porespy; they are slower (~2–5 s total) but require no GPU.

- [ ] **Step 3: Run the full suite**

Run: `.venv/bin/pytest tests/ -v`

Expected: 238 passed total (225 fusion + 13 utils). Note the test count does not include the `dino_flavour` fixture parametrization for `test_utils.py` since utils tests do not depend on that fixture.

- [ ] **Step 4: Commit**

```bash
git add tests/test_utils.py
git commit -m "Add utility function unit tests (xrd_to_tensor, get_multiphase, downsample_xrdct)"
```

---

## Self-Review

**Spec coverage check:**

| Spec requirement | Task covering it |
|---|---|
| `conftest.py` with 9 DINO flavour profiles | Task 1 |
| `features` fixture `(1, feat_dim, 8, 8)` | Task 1 |
| `xrd_map_4d` fixture `(1, 1, 8, 8)` | Task 1 |
| XRDCrossAttention: shape, dtype, tensor, 2D/3D xrd input | Task 2 |
| LearnedChannelGating: loss scalars, passthrough, top-k count, shape, dtype | Task 3 |
| `_direct_correlation_gating`: shape, dtype, tensor, gate range, top-k, default top-k, spatial mismatch | Task 4 |
| `xrd_to_tensor`: tensor, shape, dtype, range, device | Task 5 |
| `get_multiphase`: types, dataset_size, mask binary, gray range, shape consistency | Task 5 |
| `downsample_xrdct`: shape, dtype, list lengths | Task 5 |

All spec requirements covered.

**Placeholder scan:** No TBDs, TODOs, or "similar to above" references. All test code is fully written out in each task.

**Type consistency:** `XRDCrossAttention`, `LearnedChannelGating`, and `XRDFusionMethod` are imported from `x_fuse.fusion` consistently across Tasks 2–4. `xrd_to_tensor`, `get_multiphase`, `downsample_xrdct` imported from `x_fuse.utils` in Task 5. Fixture names (`dino_flavour`, `features`, `xrd_map_4d`) match `conftest.py` definitions exactly.
