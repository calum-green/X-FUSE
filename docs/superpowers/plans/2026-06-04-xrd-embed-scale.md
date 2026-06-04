# xrd_embed_scale Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add `xrd_embed_scale` as a new `FusionOptions` value that scales DINOv3 patch token embeddings by min-max normalised XRD weights before the transformer blocks run, propagating phase information through the full network depth.

**Architecture:** A PyTorch forward hook registered on `inner.patch_embed` reads `_xrd_scale` (set per forward call in `forward_sequential`), multiplies the patch embedding output by `(1 + w).reshape(1, N_patches, 1)` where `w ∈ [0, 1]`, then clears the attribute. `forward_xrd` returns `x` unchanged. The hook is a no-op for all other fusion methods. DINOv3 only — a `ValueError` is raised for non-DINOv3 models.

**Tech Stack:** PyTorch (`register_forward_hook`), existing X-FUSE fusion patterns (`_xrd_bias` attribute injection)

**Spec:** `docs/superpowers/specs/2026-06-04-xrd-embed-scale-design.md`

---

## File Map

| File | Change |
|---|---|
| `x_fuse/fusion.py` | `FusionOptions`, `forward_xrd` dispatch, `patch_last_block` hook + assertion, `forward_sequential` scale injection |
| `tests/test_fusion.py` | 3 new dispatch tests |

---

### Task 1: `FusionOptions` + `forward_xrd` dispatch

**Files:**
- Modify: `x_fuse/fusion.py:22-30` (`FusionOptions`)
- Modify: `x_fuse/fusion.py:192-196` (`forward_xrd`)
- Test: `tests/test_fusion.py`

- [ ] **Step 1: Write the failing tests**

Add to the bottom of `tests/test_fusion.py`:

```python
# ── XRDFusionMethod.forward_xrd xrd_embed_scale dispatch ─────────────────────


def test_forward_xrd_embed_scale_returns_x(dino_flavour, features, xrd_map_4d):
    fusion = _make_fusion(xrd_map_4d)
    result = fusion.forward_xrd(features, idx=0, xrd_fusion_method="xrd_embed_scale")
    assert result is features


def test_forward_xrd_embed_scale_shape(dino_flavour, features, xrd_map_4d):
    fusion = _make_fusion(xrd_map_4d)
    result = fusion.forward_xrd(features, idx=0, xrd_fusion_method="xrd_embed_scale")
    assert result.shape == features.shape


def test_forward_xrd_embed_scale_dtype(dino_flavour, features, xrd_map_4d):
    fusion = _make_fusion(xrd_map_4d)
    result = fusion.forward_xrd(features, idx=0, xrd_fusion_method="xrd_embed_scale")
    assert result.dtype == features.dtype
```

- [ ] **Step 2: Run tests to confirm they fail**

```bash
pytest tests/test_fusion.py::test_forward_xrd_embed_scale_returns_x \
       tests/test_fusion.py::test_forward_xrd_embed_scale_shape \
       tests/test_fusion.py::test_forward_xrd_embed_scale_dtype -v
```

Expected: FAIL — `"xrd_embed_scale"` is not a valid `FusionOptions` literal

- [ ] **Step 3: Add `"xrd_embed_scale"` to `FusionOptions`**

In `x_fuse/fusion.py`, replace:

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
    "xrd_embed_scale",
]
```

- [ ] **Step 4: Add `forward_xrd` dispatch branch**

In `XRDFusionMethod.forward_xrd`, add immediately after the `xrd_attn_weight` branch:

```python
        if xrd_fusion_method == "xrd_attn_weight":
            return x  # features already biased before this call

        if xrd_fusion_method == "xrd_embed_scale":
            return x  # features already biased at patch_embed before this call
```

- [ ] **Step 5: Run tests to confirm they pass**

```bash
pytest tests/test_fusion.py::test_forward_xrd_embed_scale_returns_x \
       tests/test_fusion.py::test_forward_xrd_embed_scale_shape \
       tests/test_fusion.py::test_forward_xrd_embed_scale_dtype -v
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
git commit -m "feat: add xrd_embed_scale to FusionOptions and forward_xrd dispatch"
```

---

### Task 2: `patch_last_block` — hook registration + DINOv3 assertion

**Files:**
- Modify: `x_fuse/fusion.py:627-646` (`XFuse.patch_last_block`)

`patch_last_block` is called once in `XFuse.__init__`. The hook registered here fires on
every subsequent `patch_embed` forward call. `inner = getattr(dino_model, "model", dino_model)`
unwraps `_DV3Wrapper` to reach the real transformer (same pattern as `xrd_attn_weight`).

This task cannot be unit-tested in isolation (requires a live DINOv3 model). Correctness
is verified by running the integration notebook. The full unit test suite must still pass.

- [ ] **Step 1: Add assertion + hook registration to `patch_last_block`**

In `XFuse.patch_last_block`, the method currently ends with:

```python
        if "vanilla_dv3" in dino_name:
            inner = getattr(dino_model, "model", dino_model)
            attn_block = inner.blocks[-1].attn
            attn_block.forward = MethodType(Patch._fix_dv3_attn(), attn_block)
```

Replace with:

```python
        if "vanilla_dv3" in dino_name:
            inner = getattr(dino_model, "model", dino_model)
            attn_block = inner.blocks[-1].attn
            attn_block.forward = MethodType(Patch._fix_dv3_attn(), attn_block)

        if self.xrd_fuse_method == "xrd_embed_scale":
            if "dv3" not in dino_name:
                raise ValueError(
                    f"xrd_embed_scale requires a DINOv3 model, got '{dino_name}'"
                )
            inner = getattr(dino_model, "model", dino_model)

            def _embed_hook(module, input, output):
                scale = getattr(module, "_xrd_scale", None)
                if scale is not None:
                    module._xrd_scale = None
                    return output * scale.to(
                        dtype=output.dtype, device=output.device
                    )

            inner.patch_embed.register_forward_hook(_embed_hook)
```

- [ ] **Step 2: Run full unit test suite**

```bash
pytest tests/ -v
```

Expected: all existing tests pass (4 pre-existing failures: 2 CUDA-only, 2 segmentation)

- [ ] **Step 3: Commit**

```bash
git add x_fuse/fusion.py
git commit -m "feat: register patch_embed hook and DINOv3 assertion for xrd_embed_scale"
```

---

### Task 3: `forward_sequential` — scale injection

**Files:**
- Modify: `x_fuse/fusion.py:699-721` (`XFuse.forward_sequential`)

Scale is injected before the DINO forward call, mirroring the `xrd_attn_weight` bias
injection pattern. `_xrd_scale` is consumed and cleared inside the hook registered in
Task 2.

This task requires a live DINOv3 model. Correctness is verified in the integration notebook.

- [ ] **Step 1: Add scale injection before the DINO forward call**

In `forward_sequential`, the loop currently starts:

```python
        for i in range(N_transforms):
            if self.xrd_fuse_method == "xrd_attn_weight":
                tr_xrd = self.xrd_fusion_module.get_tr()[i]
                xrd_patch = F.interpolate(
                    tr_xrd.float(),
                    (n_patch_h, n_patch_w),
                    mode="bilinear",
                    align_corners=False,
                ).reshape(n_patch_h * n_patch_w)
                ...
                inner.blocks[-1].attn._xrd_bias = xrd_bias

            transformed_img = img_batch[i].unsqueeze(0)
```

Add the `xrd_embed_scale` block immediately after the `xrd_attn_weight` block:

```python
            if self.xrd_fuse_method == "xrd_embed_scale":
                tr_xrd = self.xrd_fusion_module.get_tr()[i]
                xrd_patch = F.interpolate(
                    tr_xrd.float(),
                    (n_patch_h, n_patch_w),
                    mode="bilinear",
                    align_corners=False,
                ).reshape(n_patch_h * n_patch_w)
                xrd_min = xrd_patch.min()
                xrd_max = xrd_patch.max()
                w = (xrd_patch - xrd_min) / (xrd_max - xrd_min + 1e-8)
                scale = (1.0 + w).reshape(1, -1, 1).to(
                    dtype=self.dtype, device=xrd_patch.device
                )
                inner = getattr(self.dinov2, "model", self.dinov2)
                inner.patch_embed._xrd_scale = scale
```

- [ ] **Step 2: Run full unit test suite**

```bash
pytest tests/ -v
```

Expected: all existing tests pass

- [ ] **Step 3: Commit**

```bash
git add x_fuse/fusion.py
git commit -m "feat: inject min-max XRD scale at patch_embed in forward_sequential"
```
