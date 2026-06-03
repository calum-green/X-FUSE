# xrd_attn_weight Fusion Method — Design Spec

**Date:** 2026-06-03
**Branch:** feature/weighted_pca

---

## Background

`weighted_pca` and `cosine_similarity` both operate on DINO features *after* they are extracted, using the XRD map to weight or select features in post-processing. A complementary approach is to bias the attention computation *inside* DINO's last transformer block so that the features themselves are shaped by the XRD signal before they leave the model.

The mechanism: in the last block's multi-head self-attention, high-XRD patch tokens are weighted as more relevant keys. Every query token (including CLS and register tokens) attends more strongly to Phase A patches. The resulting patch features encode more Phase A-relevant context than a vanilla forward pass.

This is achieved by passing `log(w + 1e-8)` — where `w` is the XRD map normalised to sum=1 — as an additive pre-softmax bias to `memory_efficient_attention`. Adding a log bias before softmax is mathematically equivalent to post-softmax multiplicative weighting with renormalisation:

```
softmax(q @ k.T + log(w)) = softmax(q @ k.T) * w / Z
```

The XRD bias is `(1, 1, 1, N_total)` — broadcasting across batch and head dimensions — so every head receives the same per-key XRD weight. CLS and register token key positions receive a bias of `0.0` (i.e. `log(1)`), leaving them unaffected.

---

## Goal

Add `xrd_attn_weight` as a new `FusionOptions` value. For each HR-Dv2 augmentation pass, the XRD map is used to construct a log-scale attention bias that is injected into `blocks[-1].attn` before the DINO forward pass. The resulting features are XRD-biased full-channel DINO features, upsampled by HR-Dv2 as normal.

Output shape: `(1, C, img_h, img_w)` — full feature channels, one per call, averaged across augmentation passes.

---

## Algorithm

**Per transform in `forward_sequential`** when `xrd_fuse_method == "xrd_attn_weight"`:

1. Interpolate `tr_xrd` to patch grid `(H_p, W_p)` → flatten to `(N_patches,)` → normalise to sum=1 → `w`
2. Compute `log_w = log(w + 1e-8)` → `(N_patches,)`
3. Build `xrd_bias (1, 1, 1, N_total)` where `N_total = 1 + n_register_tokens + N_patches`:
   - positions `0..n_prefix-1`: `0.0` (CLS + register tokens — unbiased)
   - positions `n_prefix..N_total-1`: `log_w` (patch tokens)
4. Set `self.dinov2.blocks[-1].attn._xrd_bias = xrd_bias`
5. Call `forward_feats_attn` → inside `_fix_mem_eff_attn`, `_xrd_bias` is passed as `attn_bias` to `memory_efficient_attention` and cleared immediately after
6. Features exit DINO with XRD-biased last-block attention
7. `forward_xrd` returns features unchanged (behaves like `vanilla`)
8. Full-channel upsample and accumulate — no active-channel skipping

When `xrd_fuse_method` is anything other than `xrd_attn_weight`, steps 1–4 are skipped and `_xrd_bias` is never set. The attention module behaves identically to before.

---

## Integration Points

### 1. `FusionOptions` — `fusion.py:21`

```python
FusionOptions: TypeAlias = Literal[
    "gating", "learned_gating", "attention", "vanilla",
    "weighted_pca", "cosine_similarity", "xrd_attn_weight",
]
```

### 2. `XRDFusionMethod.forward_xrd` — new branch

```python
if xrd_fusion_method == "xrd_attn_weight":
    return x  # features already biased before this call
```

### 3. `XFuse.patch_last_block` — `vanilla_dv3` extension

After the existing `forward_feats_attn` override for the `dv3` branch, additionally patch the last block's attention module so `_fix_mem_eff_attn` is active:

```python
if "vanilla_dv3" in dino_name:
    attn_block = dino_model.blocks[-1].attn
    attn_block.forward = MethodType(Patch._fix_mem_eff_attn(), attn_block)
```

Standard DINOv2 already has `_fix_mem_eff_attn` applied via `super().patch_last_block()`.

### 4. `XFuse.forward_sequential` — bias injection + active-channel fix

**Active-channel optimisation** currently applied unconditionally must be guarded to methods where channels can be zeroed out:

```python
# Only skip zero channels for sparse gating methods
if self.xrd_fuse_method in ("gating", "learned_gating", "attention"):
    active_idx = fused_img.abs().sum(dim=(0, 2, 3)).nonzero(as_tuple=True)[0].cpu()
    if active_idx.numel() == 0:
        continue
    fused_img = fused_img[:, active_idx]
    # ... upsample active_idx slice, accumulate by index
else:
    # all channels active — upsample directly
    full_size = F.interpolate(fused_img, (img_h, img_w), mode=self.interpolation_mode)
    ...
```

**XRD bias injection** — before the `forward_feats_attn` call:

```python
if self.xrd_fuse_method == "xrd_attn_weight":
    tr_xrd = self.xrd_fusion_module.get_tr()[i]
    xrd_patch = F.interpolate(
        tr_xrd.float(), (n_patch_h, n_patch_w),
        mode="bilinear", align_corners=False,
    ).reshape(n_patch_h * n_patch_w)
    w = xrd_patch / (xrd_patch.sum() + 1e-8)
    n_prefix = 1 + self.n_register_tokens
    N_total = n_prefix + n_patch_h * n_patch_w
    xrd_bias = torch.zeros(1, 1, 1, N_total, dtype=self.dtype, device=xrd_patch.device)
    xrd_bias[0, 0, 0, n_prefix:] = torch.log(w + 1e-8)
    self.dinov2.blocks[-1].attn._xrd_bias = xrd_bias
```

### 5. `Patch._fix_mem_eff_attn` — `patch.py`

After building `q`, `k`, `v` and before calling `memory_efficient_attention`, read and consume `_xrd_bias`:

```python
xrd_bias = getattr(self, '_xrd_bias', None)
effective_bias = xrd_bias if xrd_bias is not None else attn_bias
x = memory_efficient_attention(q, k, v, attn_bias=effective_bias)
if xrd_bias is not None:
    self._xrd_bias = None
```

Existing `attn_bias` behaviour is unchanged when `_xrd_bias` is absent.

---

## Model Support

| Model | Supported | Notes |
|---|---|---|
| DINOv2 (all sizes) | ✓ | `_fix_mem_eff_attn` already applied via `super().patch_last_block()` |
| vanilla DINOv3 | ✓ | Additional `_fix_mem_eff_attn` patch added in `patch_last_block` |
| NoPE | Future | See below |
| ALiBi | Future | See below |

---

## Future Work: NoPE and ALiBi Support

### NoPE

NoPE removes positional encoding but keeps standard self-attention with the same module interface as DINOv2. Support is expected to be trivial: add the same `_fix_mem_eff_attn` patch to `blocks[-1].attn` in the NoPE branch of `patch_last_block`, identical to `vanilla_dv3`. Requires confirming `qkv`/`proj`/`proj_drop` attribute names match — likely yes given shared ViT architecture.

### ALiBi

ALiBi adds a distance-based linear bias to attention logits pre-softmax — the same additive position as the XRD bias. The two biases can simply be summed: `effective_bias = alibi_bias + xrd_bias`. The implementation effort depends on where ALiBi constructs its bias tensor:
- If ALiBi already passes its bias via the `attn_bias` parameter of `memory_efficient_attention`, the change is `effective_bias = existing_attn_bias + xrd_bias`
- If ALiBi embeds its bias inside a custom `forward`, the XRD bias injection point needs to be located within that method

Either way, no custom CUDA kernel is required.

---

## Output

`forward_sequential` returns `(1, C, img_h, img_w)` when `xrd_fuse_method == "xrd_attn_weight"`, where `C = feat_dim` (or `feat_dim + n_heads` if `attn_choice != "none"`).

Notebook usage:
```python
net.set_xrd_transforms(xrd_A_tensor, fwd, inv)
feats_A = net.forward_sequential(xct_tensor)   # (1, C, img_h, img_w)

net.set_xrd_transforms(xrd_B_tensor, fwd, inv)
feats_B = net.forward_sequential(xct_tensor)   # (1, C, img_h, img_w)
```

Downstream: apply PCA, cosine similarity, or direct visualisation to `feats_A` / `feats_B`.

---

## Files Changed

| File | Change |
|---|---|
| `x_fuse/fusion.py` | Add `"xrd_attn_weight"` to `FusionOptions`; branch in `forward_xrd`; `vanilla_dv3` attention patch in `patch_last_block`; bias injection + active-channel guard in `forward_sequential` |
| `HR-Dv2/hr_dv2/patch.py` | `_fix_mem_eff_attn`: read and consume `_xrd_bias` as `attn_bias` |

---

## Out of Scope

- ALiBi and NoPE support (noted above as future work)
- Changes to `train_fusion_step` (no learnable parameters)
- Changes to `XFuse.__init__` (no new modules)
- Combining `xrd_attn_weight` with `cosine_similarity`/`weighted_pca` in a single pass
