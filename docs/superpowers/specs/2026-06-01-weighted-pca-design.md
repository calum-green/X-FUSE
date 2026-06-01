# weighted_pca Fusion Method — Design Spec

**Date:** 2026-06-01  
**Branch:** feature/weighted_pca

---

## Background

Current `gating` fusion scores each DINO feature channel independently against the XRD map (BCE or Pearson), then keeps the top-K. This greedy per-channel selection only works when the phase signal is concentrated in individually XRD-correlated channels — a property that fine-tuned ViT-S models exhibit but vanilla or less-converged 7B models do not.

On real DIAD XCT data at 1840px, phase signal is weak relative to texture and intensity variation. Unweighted PCA and per-channel gating both fail to surface it. `weighted_pca` addresses this by using the XRD map as a continuous per-patch weight in a joint covariance computation, finding the linear combination of all channels that is maximally common in high-XRD regions — without requiring the signal to be axis-aligned with individual channels.

---

## Goal

Add `weighted_pca` as a new `FusionOptions` value. For each HR-Dv2 augmentation pass, the XRD map weights the patch-level DINO feature covariance, yielding an eigenvector that captures the phase direction in channel space. All patches are projected onto this direction to produce a continuous score map (high score = high XRD intensity). HR-Dv2 aggregates score maps across augmentation passes to produce a full-resolution output.

Output shape: `(1, 1, img_h, img_w)` — one continuous score map per call, one call per phase.

---

## Algorithm: `_weighted_pca`

**Inputs:**
- `x`: `(1, C, H_patch, W_patch)` — DINO features at patch level (on GPU, dtype = model dtype)
- `xrd_map`: `(1, 1, H_xrd, W_xrd)` — pre-transformed XRD map (same transform as augmentation pass `i`)

**Steps:**

1. Interpolate XRD to patch grid → `(1, 1, H_patch, W_patch)` (bilinear, same as gating)
2. Flatten XRD → weights `(N,)` where `N = H_patch * W_patch`; normalise to sum=1
3. Flatten features → `(N, C)`, cast to **float32** for numerical stability
4. Weighted mean: `mu = (w[:, None] * feats).sum(0)` → `(C,)`
5. Centre: `centered = feats - mu` → `(N, C)`
6. Apply sqrt weights: `wc = w.sqrt()[:, None] * centered` → `(N, C)`
7. `_, _, V = torch.pca_lowrank(wc, q=1, center=False, niter=4)` → `v = V[:, 0]` → `(C,)`
8. Sign correction: if `(feats @ v * w).sum() < 0`: `v = -v`  (ensures high score = high XRD intensity)
9. Project: `scores = feats @ v` → reshape `(1, 1, H_patch, W_patch)`
10. Cast back to input dtype, return `(1, 1, H_patch, W_patch)`

**Why step 6 works:** The eigenvectors of the weighted covariance `X_c^T diag(w) X_c` are the right singular vectors of `diag(w)^{1/2} X_c`. This avoids explicitly forming the `(C, C)` covariance matrix (64 MB in float32 for C=4096).

**XRD upsampling:** XRD goes from its native resolution (e.g. 20×20) to patch grid only (e.g. 115×115 for 1840px / stride 16). Never upsampled to full XCT resolution.

---

## Integration Points

### 1. `FusionOptions` type alias — `fusion.py:21`

```python
FusionOptions: TypeAlias = Literal["gating", "learned_gating", "attention", "vanilla", "weighted_pca"]
```

### 2. `XRDFusionMethod.forward_xrd` — new branch before existing `tr_xrd = self.get_tr()[idx]`

```python
if xrd_fusion_method == "weighted_pca":
    tr_xrd = self.get_tr()[idx]
    return self._weighted_pca(x, tr_xrd)
```

### 3. `XFuse.forward_sequential` — dynamic accumulator (2 lines replacing 1)

```python
# replace:
out_feature_img = torch.zeros(1, c, img_h, img_w, dtype=self.dtype)

# with:
c_out = 1 if self.xrd_fuse_method == "weighted_pca" else c
out_feature_img = torch.zeros(1, c_out, img_h, img_w, dtype=self.dtype)
```

The existing `active_idx` scatter-accumulation in `forward_sequential` handles `(1, 1, H, W)` output unchanged — `active_idx = [0]`, accumulates into `out_feature_img[:, [0]]`.

---

## Output

`forward_sequential` returns `(1, 1, img_h, img_w)` when `xrd_fuse_method == "weighted_pca"`.

Notebook usage:
```python
net.set_xrd_transforms(xrd_A_tensor, fwd, inv)
score_A = net.forward_sequential(xct_tensor)   # (1, 1, img_h, img_w)
score_map_A = score_A[0, 0]                    # (img_h, img_w) — ready for SAM2

net.set_xrd_transforms(xrd_B_tensor, fwd, inv)
score_B = net.forward_sequential(xct_tensor)
score_map_B = score_B[0, 0]
```

No downstream PCA step needed — the score map is directly phase-meaningful.

---

## Files Changed

| File | Change |
|---|---|
| `x_fuse/fusion.py` | Add `_weighted_pca` to `XRDFusionMethod`; branch in `forward_xrd`; 2-line change in `forward_sequential`; update `FusionOptions` |

---

## Out of Scope

- Changes to `train_fusion_step` (weighted_pca requires no training)
- Changes to `XFuse.__init__` (no new modules needed)
- L2 normalisation of features before weighted PCA (can be added in notebook if needed)
