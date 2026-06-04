# xrd_embed_scale Fusion Method — Design Spec

**Date:** 2026-06-04
**Branch:** feature/weighted_pca
**Status:** Approved for implementation

---

## Problem

`xrd_attn_weight` injects XRD as an additive log-bias into the last transformer block's
attention. By that point the representation is fully formed across 46 earlier blocks that
had no XRD knowledge. The effect is shallow: a single block's attention routing is nudged,
but the value vectors it mixes are unmodified DINO features.

To make DINO genuinely encode XRD phase information, XRD must enter before the first block
so the signal propagates through the full network depth via DINO's own attention.

---

## Solution

`xrd_embed_scale` intercepts the output of `patch_embed` — the earliest token
representation — and scales each patch token by `(1 + w_i)` where `w_i ∈ [0, 1]` is the
min-max normalised XRD weight for patch i.

High-XRD (phase A) patches enter the transformer with larger token magnitudes. In
self-attention, larger-magnitude tokens produce larger `q·k` dot products, so they
attract more attention from other tokens AND attend more strongly to each other — across
all 47 blocks, not just the last. The XRD signal compounds through depth via DINO's own
learned attention mechanism.

---

## Data Flow

```
XCT image  (B, 3, H, W)
    │
patch_embed                     → (B, N_patches, embed_dim)
    │   forward hook fires here
    │   output *= (1 + w).reshape(1, N_patches, 1)
    │
prepare_tokens_with_masks       prepend CLS, register tokens  ← unaffected
    │                           add positional encoding
    ↓
block 0 … block N-1             all see XRD-scaled patch tokens
    │
x_norm_patchtokens              (B, N_patches, embed_dim)  ← XRD-biased output
```

---

## Normalisation

XRD is interpolated to patch resolution then min-max normalised:

```
xrd_patch  = interpolate(tr_xrd, (n_patch_h, n_patch_w))  # (N_patches,)
w          = (xrd_patch - xrd_patch.min()) / (xrd_patch.max() - xrd_patch.min() + 1e-8)
scale      = (1.0 + w).reshape(1, -1, 1)                  # (1, N_patches, 1)
```

`w ∈ [0, 1]` so scale ∈ [1.0, 2.0]:
- No XRD patch → scale = 1.0 (token unchanged)
- Peak XRD patch → scale = 2.0 (token doubled in magnitude)

Scale is cast to `output.dtype` and `output.device` inside the hook to avoid dtype/device
mismatches (lesson from `xrd_attn_weight`).

**Why min-max and not sum-to-1?** The probability normalisation used in `xrd_attn_weight`
gives `w_avg ≈ 1/N_patches` (e.g. 0.00073 for 1369 patches), making `scale ≈ 1.00073` —
essentially invisible. Min-max gives a meaningful [1, 2] range. The log transform in
`xrd_attn_weight` made sum-to-1 acceptable there; direct multiplication requires min-max.

---

## Injection Mechanism

A PyTorch forward hook is registered once on `inner.patch_embed` during `patch_last_block`.
Before each DINO forward call in `forward_sequential`, `_xrd_scale` is set on the module.
The hook reads, clears, and applies it:

```python
def _embed_hook(module, input, output):
    scale = getattr(module, '_xrd_scale', None)
    if scale is not None:
        module._xrd_scale = None
        return output * scale.to(dtype=output.dtype, device=output.device)

inner.patch_embed.register_forward_hook(_embed_hook)
```

The hook is a no-op when `_xrd_scale` is not set (other fusion methods unaffected).

---

## Model Support

**DINOv3 only.** `patch_last_block` raises `ValueError` if `xrd_fuse_method ==
"xrd_embed_scale"` and `"dv3"` is not in `dino_name`. DINOv2 uses a different code path
(HR-DV2 base class) and is not supported in this version.

---

## File Changes

| File | Change |
|---|---|
| `x_fuse/fusion.py` | Add `"xrd_embed_scale"` to `FusionOptions`; dispatch in `forward_xrd`; hook registration + assertion in `patch_last_block`; scale injection in `forward_sequential` |
| `tests/test_fusion.py` | 3 dispatch tests: `returns_x`, `shape`, `dtype` |

`HR-Dv2/hr_dv2/patch.py` and `tests/test_patch.py` are unchanged — the hook is a plain
closure in `fusion.py`, not a `Patch` static method.

---

## Upsampling

`forward_xrd` returns `x` unchanged (same as `vanilla`, `xrd_attn_weight`). Features flow
into the full-channel `else` upsampling branch in `forward_sequential`.

---

## Comparison with xrd_attn_weight

| | `xrd_attn_weight` | `xrd_embed_scale` |
|---|---|---|
| Injection point | `blocks[-1].attn` (last block) | `patch_embed` (before block 0) |
| Blocks influenced | 1 of N | All N |
| Mechanism | Additive log-bias on attention logits | Multiplicative scale on token magnitude |
| Normalisation | Sum-to-1 probability → log-prob | Min-max → [0, 1] |
| Bias shape | `(1, 1, N_total, N_total)` | `(1, N_patches, 1)` |
| Memory overhead | O(N²) | O(N) |

---

## Future Work

- Support DINOv2 (register hook via HR-DV2 base class `patch_last_block`)
- Combine `xrd_embed_scale` + `xrd_attn_weight` as a dual-injection option
- Make α configurable (currently fixed at 1.0)
