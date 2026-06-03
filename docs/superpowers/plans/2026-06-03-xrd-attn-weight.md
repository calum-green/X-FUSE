# xrd_attn_weight Fusion Method — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add `xrd_attn_weight` as a new `FusionOptions` value that injects a log-scale XRD bias into DINO's last-block `memory_efficient_attention`, producing full C-channel XRD-biased DINO feature maps.

**Architecture:** Before each DINO forward call, `forward_sequential` sets `blocks[-1].attn._xrd_bias = log(w + 1e-8)` padded to `N_total` tokens. The patched `_fix_mem_eff_attn` reads and immediately clears this attribute, passing it as `attn_bias` to xformers. `forward_xrd` returns features unchanged (like `vanilla`) since the bias was applied upstream. Active-channel skipping in `forward_sequential` is guarded to sparse methods only.

**Tech Stack:** PyTorch, xformers (`memory_efficient_attention`), HR-Dv2 `MethodType` patching

**Spec:** `docs/superpowers/specs/2026-06-03-xrd-attn-weight-design.md`

---

## File Map

| File | Change |
|---|---|
| `x_fuse/fusion.py` | `FusionOptions`, `forward_xrd` branch, `patch_last_block` vanilla_dv3, `forward_sequential` bias injection + active-channel guard |
| `HR-Dv2/hr_dv2/patch.py` | `_fix_mem_eff_attn`: read/clear `_xrd_bias`, pass as `attn_bias` |
| `tests/test_fusion.py` | `forward_xrd` dispatch tests for `xrd_attn_weight` |
| `tests/test_patch.py` | NEW — `_fix_mem_eff_attn` clears `_xrd_bias` regardless of xformers availability |

---

### Task 1: `FusionOptions` + `forward_xrd` dispatch

**Files:**
- Modify: `x_fuse/fusion.py:21-28` (`FusionOptions`)
- Modify: `x_fuse/fusion.py:190-210` (`forward_xrd`)
- Test: `tests/test_fusion.py`

- [ ] **Step 1: Write the failing tests**

Add to the bottom of `tests/test_fusion.py`:

```python
# ── XRDFusionMethod.forward_xrd xrd_attn_weight dispatch ─────────────────────


def test_forward_xrd_attn_weight_returns_x(dino_flavour, features, xrd_map_4d):
    fusion = _make_fusion(xrd_map_4d)
    result = fusion.forward_xrd(features, idx=0, xrd_fusion_method="xrd_attn_weight")
    assert result is features


def test_forward_xrd_attn_weight_shape(dino_flavour, features, xrd_map_4d):
    fusion = _make_fusion(xrd_map_4d)
    result = fusion.forward_xrd(features, idx=0, xrd_fusion_method="xrd_attn_weight")
    assert result.shape == features.shape


def test_forward_xrd_attn_weight_dtype(dino_flavour, features, xrd_map_4d):
    fusion = _make_fusion(xrd_map_4d)
    result = fusion.forward_xrd(features, idx=0, xrd_fusion_method="xrd_attn_weight")
    assert result.dtype == features.dtype
```

- [ ] **Step 2: Run tests to confirm they fail**

```bash
pytest tests/test_fusion.py::test_forward_xrd_attn_weight_returns_x \
       tests/test_fusion.py::test_forward_xrd_attn_weight_shape \
       tests/test_fusion.py::test_forward_xrd_attn_weight_dtype -v
```

Expected: FAIL — `forward_xrd` returns `None` for unknown method

- [ ] **Step 3: Update `FusionOptions` type alias**

In `x_fuse/fusion.py`, replace:

```python
FusionOptions: TypeAlias = Literal[
    "gating",
    "learned_gating",
    "attention",
    "vanilla",
    "weighted_pca",
    "cosine_similarity",
]
```

With:

```python
FusionOptions: TypeAlias = Literal[
    "gating",
    "learned_gating",
    "attention",
    "vanilla",
    "weighted_pca",
    "cosine_similarity",
    "xrd_attn_weight",
]
```

- [ ] **Step 4: Add `forward_xrd` dispatch branch**

In `XRDFusionMethod.forward_xrd`, add the new branch immediately after the `"vanilla"` check:

```python
        if xrd_fusion_method == "vanilla":
            return x

        if xrd_fusion_method == "xrd_attn_weight":
            return x  # features already biased before this call

        if xrd_fusion_method == "weighted_pca":
```

- [ ] **Step 5: Run tests to confirm they pass**

```bash
pytest tests/test_fusion.py::test_forward_xrd_attn_weight_returns_x \
       tests/test_fusion.py::test_forward_xrd_attn_weight_shape \
       tests/test_fusion.py::test_forward_xrd_attn_weight_dtype -v
```

Expected: PASS

- [ ] **Step 6: Run full fusion test suite for regressions**

```bash
pytest tests/test_fusion.py -v
```

Expected: all existing tests pass

- [ ] **Step 7: Commit**

```bash
git add x_fuse/fusion.py tests/test_fusion.py
git commit -m "feat: add xrd_attn_weight to FusionOptions and forward_xrd dispatch"
```

---

### Task 2: Modify `_fix_mem_eff_attn` to consume `_xrd_bias`

**Files:**
- Modify: `HR-Dv2/hr_dv2/patch.py:184-215`
- Test: `tests/test_patch.py` (new file)

`_xrd_bias` must be cleared **before** the xformers availability check so it is always cleared even when xformers is absent (CPU-only environments, test runners).

- [ ] **Step 1: Write the failing tests**

Create `tests/test_patch.py`:

```python
import pytest
import torch
from types import MethodType
from hr_dv2.patch import Patch


class MockAttn(torch.nn.Module):
    """Minimal mock of a DINOv2 attention module with the attributes
    that _fix_mem_eff_attn expects: qkv, proj, proj_drop, attn_drop, num_heads."""

    def __init__(self, feat_dim: int = 64, num_heads: int = 4):
        super().__init__()
        self.num_heads = num_heads
        self.qkv = torch.nn.Linear(feat_dim, feat_dim * 3, bias=False)
        self.proj = torch.nn.Linear(feat_dim, feat_dim, bias=False)
        self.proj_drop = torch.nn.Dropout(0.0)
        self.attn_drop = torch.nn.Dropout(0.0)

    def forward(self, x, attn_bias=None, attn_choice="none"):
        return x  # replaced entirely by the patch


@pytest.fixture
def patched_attn():
    attn = MockAttn(feat_dim=64, num_heads=4)
    attn.forward = MethodType(Patch._fix_mem_eff_attn(), attn)
    return attn


def test_xrd_bias_cleared_after_forward(patched_attn):
    """_xrd_bias must be cleared on the module after a forward call,
    regardless of whether xformers is available."""
    B, N, C = 1, 10, 64
    x = torch.randn(B, N, C)
    xrd_bias = torch.zeros(1, 1, 1, N)
    xrd_bias[0, 0, 0, 2:] = torch.log(torch.rand(N - 2) + 1e-8)
    patched_attn._xrd_bias = xrd_bias
    try:
        patched_attn(x)
    except Exception:
        pass  # xformers may not be available in test env; we only care about clearing
    assert getattr(patched_attn, '_xrd_bias', None) is None


def test_no_xrd_bias_does_not_error(patched_attn):
    """Calling forward without _xrd_bias set must not raise AttributeError."""
    B, N, C = 1, 10, 64
    x = torch.randn(B, N, C)
    assert not hasattr(patched_attn, '_xrd_bias')
    try:
        patched_attn(x)
    except AttributeError:
        pytest.fail("forward raised AttributeError when _xrd_bias was not set")
    except Exception:
        pass  # other errors (e.g. no xformers) are acceptable
```

- [ ] **Step 2: Run tests to confirm they fail**

```bash
pytest tests/test_patch.py -v
```

Expected: `test_xrd_bias_cleared_after_forward` FAIL — `_xrd_bias` is not cleared by current code

- [ ] **Step 3: Modify `_fix_mem_eff_attn` in `patch.py`**

Inside `_fix_mem_eff_attn`, replace the `forward` function body:

```python
        def forward(
            self,
            x: torch.Tensor,
            attn_bias=None,
            attn_choice: AttentionOptions = "none",
        ) -> torch.Tensor:
            # Read and immediately clear _xrd_bias so it is always consumed,
            # even when xformers is unavailable (prevents stale state).
            xrd_bias = getattr(self, '_xrd_bias', None)
            if xrd_bias is not None:
                self._xrd_bias = None

            if not XFORMERS_AVAILABLE:
                if attn_bias is not None:
                    raise AssertionError(
                        "xFormers is required for using nested tensors"
                    )
                return super(type(self), self).forward(x)  # type: ignore
            B, N, C = x.shape
            qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads)

            q, k, v = unbind(qkv, 2)
            effective_bias = xrd_bias if xrd_bias is not None else attn_bias
            x = memory_efficient_attention(q, k, v, attn_bias=effective_bias)
            to_append: torch.Tensor
            if attn_choice != "none":
                to_append = get_qkvo_per_head(q, k, v, x, attn_choice, self.attn_drop)

            x = x.reshape([B, N, C])

            x = self.proj(x)
            x = self.proj_drop(x)

            if attn_choice != "none":
                x = torch.concat((x, to_append), dim=-1)
            return x
```

- [ ] **Step 4: Run tests to confirm they pass**

```bash
pytest tests/test_patch.py -v
```

Expected: both tests PASS

- [ ] **Step 5: Commit**

```bash
git add HR-Dv2/hr_dv2/patch.py tests/test_patch.py
git commit -m "feat: consume _xrd_bias in _fix_mem_eff_attn for XRD attention weighting"
```

---

### Task 3: `forward_sequential` — active-channel guard + XRD bias injection

**Files:**
- Modify: `x_fuse/fusion.py:689-725`

`forward_sequential` requires a live DINO model and cannot be unit-tested in isolation. Correctness is verified by running the integration notebook.

- [ ] **Step 1: Fix the active-channel guard**

In `forward_sequential`, replace the block starting at line 708 (`# Compact to active channels...`) through line 723:

```python
            # Compact to active channels on GPU before upsampling, then
            # scatter-accumulate into the CPU accumulator by index.
            active_idx = (
                fused_img.abs().sum(dim=(0, 2, 3)).nonzero(as_tuple=True)[0].cpu()
            )
            if active_idx.numel() == 0:
                continue

            full_size = F.interpolate(
                fused_img[:, active_idx],
                (img_h, img_w),
                mode=self.interpolation_mode,
            )
            inv_transform = self.inverse_transforms[i]
            inverted: torch.Tensor = inv_transform(full_size)
            out_feature_img[:, active_idx] += inverted.cpu()
```

With:

```python
            if self.xrd_fuse_method in ("gating", "learned_gating", "attention"):
                # Skip fully-zeroed channels — only valid for sparse gating methods.
                active_idx = (
                    fused_img.abs().sum(dim=(0, 2, 3)).nonzero(as_tuple=True)[0].cpu()
                )
                if active_idx.numel() == 0:
                    continue
                full_size = F.interpolate(
                    fused_img[:, active_idx],
                    (img_h, img_w),
                    mode=self.interpolation_mode,
                )
                inv_transform = self.inverse_transforms[i]
                inverted: torch.Tensor = inv_transform(full_size)
                out_feature_img[:, active_idx] += inverted.cpu()
            else:
                full_size = F.interpolate(
                    fused_img,
                    (img_h, img_w),
                    mode=self.interpolation_mode,
                )
                inv_transform = self.inverse_transforms[i]
                inverted: torch.Tensor = inv_transform(full_size)
                out_feature_img += inverted.cpu()
```

- [ ] **Step 2: Add XRD bias injection before the `forward_feats_attn` call**

Inside the `for i in range(N_transforms):` loop, insert the following block **before** the `out_dict = self.dinov2.forward_feats_attn(...)` line:

```python
            if self.xrd_fuse_method == "xrd_attn_weight":
                tr_xrd = self.xrd_fusion_module.get_tr()[i]
                xrd_patch = F.interpolate(
                    tr_xrd.float(),
                    (n_patch_h, n_patch_w),
                    mode="bilinear",
                    align_corners=False,
                ).reshape(n_patch_h * n_patch_w)
                w = xrd_patch / (xrd_patch.sum() + 1e-8)
                n_prefix = 1 + self.n_register_tokens
                N_total = n_prefix + n_patch_h * n_patch_w
                xrd_bias = torch.zeros(
                    1, 1, 1, N_total, dtype=self.dtype, device=xrd_patch.device
                )
                xrd_bias[0, 0, 0, n_prefix:] = torch.log(w + 1e-8)
                self.dinov2.blocks[-1].attn._xrd_bias = xrd_bias
```

- [ ] **Step 3: Commit**

```bash
git add x_fuse/fusion.py
git commit -m "feat: add XRD bias injection and fix active-channel guard in forward_sequential"
```

---

### Task 4: `patch_last_block` — apply `_fix_mem_eff_attn` for `vanilla_dv3`

**Files:**
- Modify: `x_fuse/fusion.py:622-636`

`patch_last_block` requires a live DINOv3 model to verify. Correctness is confirmed by running the integration notebook with `vanilla_dv3` and `xrd_attn_weight`.

- [ ] **Step 1: Add vanilla_dv3 attention patch**

In `XFuse.patch_last_block`, add the following block immediately after the `dino_model.forward_feats_attn = MethodType(...)` assignment:

```python
        if "vanilla_dv3" in dino_name:
            attn_block = dino_model.blocks[-1].attn
            attn_block.forward = MethodType(Patch._fix_mem_eff_attn(), attn_block)
```

The complete method after the change:

```python
    def patch_last_block(self, dino_model: nn.Module, dino_name: str) -> None:
        if (
            "alibi" not in dino_name
            and "nope" not in dino_name
            and "dv3" not in dino_name
        ):
            super().patch_last_block(dino_model, dino_name)
            return

        def forward_feats_attn(self_model, x, masks=None, attn_choice="none"):
            feats = self_model.forward_features(x)  # (B, C, N_patches)
            feats = feats.permute(0, 2, 1)  # (B, N_patches, C)
            return {"x_norm_patchtokens": feats, "masks": masks}

        dino_model.forward_feats_attn = MethodType(forward_feats_attn, dino_model)

        if "vanilla_dv3" in dino_name:
            attn_block = dino_model.blocks[-1].attn
            attn_block.forward = MethodType(Patch._fix_mem_eff_attn(), attn_block)
```

- [ ] **Step 2: Run full unit test suite**

```bash
pytest tests/ -v
```

Expected: all tests pass

- [ ] **Step 3: Commit**

```bash
git add x_fuse/fusion.py
git commit -m "feat: patch blocks[-1].attn with _fix_mem_eff_attn for vanilla_dv3 xrd_attn_weight support"
```
