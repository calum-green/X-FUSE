# Unit Test Suite Design — X-FUSE

**Date:** 2026-05-06  
**Branch:** `feature/unit-tests`  
**Scope:** `x_fuse/fusion.py` (selected modules) and `x_fuse/utils.py`

---

## Overview

Add a pytest-based unit test suite for the X-FUSE codebase. Tests use synthetic random tensors rather than real model weights, making them fast, self-contained, and runnable on CPU without checkpoints. The suite is parametrized over 9 DINO model flavour profiles.

Code style is enforced with **black** (formatter) and **flake8** (linter), both configured to a maximum line length of **88 characters**.

---

## Directory Structure

```
tests/
├── conftest.py       # shared fixtures: DINO flavour profiles + synthetic tensors
├── test_fusion.py    # XRDCrossAttention, LearnedChannelGating, _direct_correlation_gating
└── test_utils.py     # xrd_to_tensor, get_multiphase, downsample_xrdct
```

---

## DINO Flavour Profiles

Defined in `conftest.py` as a parametrized `dino_flavour` fixture. Each profile is a dict of `{id, feat_dim, num_heads, patch_size, n_register_tokens}`. Profiles are derived from `MODEL_LIST` in `x_fuse/alibi.py`; new models can be added later by appending to this list.

| Fixture ID | Model | feat_dim | num_heads | patch | regs |
|---|---|---|---|---|---|
| `dinov2_s_regs` | DINOv2 ViT-S | 384 | 6 | 14 | 4 |
| `dinov2_b_regs` | DINOv2 ViT-B | 768 | 12 | 14 | 4 |
| `dinov2_l_regs` | DINOv2 ViT-L | 1024 | 16 | 14 | 4 |
| `dinov2_g_regs` | DINOv2 ViT-G | 1536 | 16 | 14 | 4 |
| `dinov2_s_no_regs` | DINOv2 ViT-S NoPE/ALiBi | 384 | 6 | 16 | 0 |
| `dinov3_s_regs` | DINOv3 ViT-S | 384 | 6 | 16 | 4 |
| `dinov3_s_no_regs` | DINOv3 ViT-S NoPE/ALiBi | 384 | 6 | 16 | 0 |
| `dinov3_l_regs` | DINOv3 ViT-L | 1024 | 16 | 16 | 4 |
| `dinov3_7b_regs` | DINOv3 7B | 4096 | 32 | 16 | 4 |

**Synthetic tensors** created per flavour:
- `features`: `torch.randn(1, feat_dim, 8, 8)` — small spatial dims for CPU speed
- `xrd_map`: `torch.rand(1, 1, 8, 8)` — normalised `[0, 1]` to mimic real XRD input

---

## Modules in Scope

### Out of scope
- `XFuse.__init__` and model-loading paths (`get_alibi_model`, `get_dv3_model`) — these require real checkpoints and `torch.hub.load`; integration-level concerns.
- `XRDFusionMethod._learned_channel_gating` and `_spatial_attention` — not currently used in production pipeline.
- `XFuse.forward_sequential` and `XFuse.train_fusion_step` — depend on a live DINO backbone.

---

## Test Coverage

### `conftest.py`

- `dino_flavour` fixture: parametrized list of 9 profile dicts, each with `id`, `feat_dim`, `num_heads`, `patch_size`, `n_register_tokens`.
- `features` fixture: derived from `dino_flavour`, returns `torch.randn(1, feat_dim, 8, 8)`.
- `xrd_map_4d` fixture: returns `torch.rand(1, 1, 8, 8)`.

---

### `test_fusion.py`

#### `XRDCrossAttention`

Instantiated with `feat_dim` and `num_heads` from each flavour profile.

| Test | Assertion |
|---|---|
| `test_output_shape` | `output.shape == features.shape` |
| `test_output_dtype` | `output.dtype == features.dtype` |
| `test_output_is_tensor` | `isinstance(output, torch.Tensor)` |
| `test_xrd_2d_input` | xrd_map as `(H, W)` → output shape unchanged |
| `test_xrd_3d_input` | xrd_map as `(1, H, W)` → output shape unchanged |

#### `LearnedChannelGating`

Instantiated with `feat_dim` from each flavour profile.

| Test | Assertion |
|---|---|
| `test_compute_loss_bce_scalar` | loss is 0-dim tensor and `> 0` |
| `test_compute_loss_pearson_scalar` | loss is 0-dim tensor and `∈ [0, 2]` |
| `test_forward_no_mask_passthrough` | without `select_top_k`, `forward` returns input unchanged |
| `test_select_top_k_channel_count` | after `select_top_k(k)`, exactly `k` channels non-zero in output |
| `test_forward_shape` | `output.shape == input.shape` |
| `test_forward_dtype` | `output.dtype == input.dtype` |

#### `XRDFusionMethod._direct_correlation_gating`

Instantiated with `loss_fn ∈ {"pearson", "bce"}` × 9 flavour profiles (18 combinations).  
`XRDFusionMethod` is constructed with `transform=[]`, `require_grad=False`, `learned_gating=None`, `spatial_attention=None`.

| Test | Assertion |
|---|---|
| `test_output_shape` | `output.shape == (B, C, H, W)` |
| `test_output_dtype` | `output.dtype == features.dtype` |
| `test_output_is_tensor` | `isinstance(output, torch.Tensor)` |
| `test_gate_values_in_range` | gate weights `∈ [0, 1]`; for each channel `c`, `output[:, c].abs() ≤ features[:, c].abs()` elementwise |
| `test_top_k_zeroing` | with explicit `top_k=k`, exactly `k` channels non-zero across spatial dims |
| `test_default_top_k` | without explicit `top_k`, `C // 4` channels non-zero |
| `test_xrd_spatial_mismatch` | xrd_map at different `(H, W)` → output shape still `(B, C, H, W)` |

---

### `test_utils.py`

#### `xrd_to_tensor`

Input: a random `uint8` numpy array of shape `(H, W)`.

| Test | Assertion |
|---|---|
| `test_output_is_tensor` | `isinstance(result, torch.Tensor)` |
| `test_output_shape` | `result.shape == (1, 1, H, W)` |
| `test_output_dtype` | `result.dtype == torch.float32` |
| `test_output_range` | `result.min() >= 0` and `result.max() <= 1` |
| `test_output_device` | `result.device == torch.device("cpu")` |

#### `get_multiphase`

| Test | Assertion |
|---|---|
| `test_return_types` | returns 3 lists of `np.ndarray` |
| `test_dataset_size` | all 3 lists have `len == dataset_size` |
| `test_mask_binary` | phase mask values `∈ {0.0, 1.0}` |
| `test_gray_range` | grayscale arrays `∈ [0, 1]` |
| `test_shape_consistency` | all arrays in each list share the same shape |

#### `downsample_xrdct`

| Test | Assertion |
|---|---|
| `test_output_shape` | downsampled spatial dims == `original // factor` |
| `test_output_dtype` | `np.float32` |
| `test_list_lengths` | output lists same length as input |

---

## Validation Criteria Summary

Every test checks at least one of:
1. **Shape** — output tensor/array has correct dimensions
2. **Dtype** — `torch.float32` or matching input dtype for pass-through ops
3. **Value range** — gates `∈ [0, 1]`, losses are finite positive scalars, arrays `∈ [0, 1]`

---

## Running the Suite

```bash
# From repo root, with dev dependencies installed
pytest tests/ -v

# Single module
pytest tests/test_fusion.py -v

# Single flavour
pytest tests/test_fusion.py -v -k "dinov3_7b"
```

## Formatting

All test files must pass black and flake8 with a max line length of 88:

```bash
black --line-length 88 tests/
flake8 --max-line-length 88 tests/
```
