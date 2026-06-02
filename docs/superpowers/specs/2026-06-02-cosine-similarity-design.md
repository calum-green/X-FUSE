# cosine_similarity Fusion Method — Design Spec

**Date:** 2026-06-02
**Branch:** feature/modularise-notebooks

---

## Background

`weighted_pca` finds the direction of maximum variance *within* Phase A patches (the first eigenvector of the XRD-weighted covariance). For real DIAD XCT data, this direction is partially driven by contextual effects from DINO's global attention — patches near the sample boundary accumulate background context in their feature vectors, and this boundary variation dominates the within-phase eigenvector. Background patches then project strongly on the resulting direction, producing high scores in non-phase regions for both Phase A and Phase B.

The root cause: PCA finds the direction of spread, not the direction of position. What we want is the direction where Phase A patches *sit* in DINO's feature space — their mean — not the direction they vary along internally.

Experimental validation: cosine similarity between a known Phase A patch and all other patches in the image produces high scores specifically in Phase A regions and low scores elsewhere, confirming that Phase A patches form a coherent directional cluster in ℝ^C.

---

## Goal

Add `cosine_similarity` as a new `FusionOptions` value. For each HR-Dv2 augmentation pass, XRD weights each patch's L2-normalised DINO feature vector to compute a Phase A prototype direction in ℝ^C. All patches are then scored by cosine similarity to that prototype, producing a continuous per-patch score map. HR-Dv2 aggregates score maps across augmentation passes to produce a full-resolution output.

Output shape: `(1, 1, img_h, img_w)` — one continuous score map per call, one call per phase.

---

## Algorithm: `_cosine_similarity`

**Inputs:**
- `x`: `(1, C, H_patch, W_patch)` — DINO features at patch level (on GPU, dtype = model dtype)
- `xrd_map`: `(1, 1, H_xrd, W_xrd)` — pre-transformed XRD map

**Steps:**

1. Interpolate XRD to patch grid → `(N,)` where `N = H_patch × W_patch`, normalise to sum=1 → weights `w`
2. Flatten features → `(N, C)`, cast to **float32** for numerical stability
3. L2-normalise each patch row → `feats_norm (N, C)` — direction only, magnitude removed
4. XRD-weighted sum across patches → `mu_A = (w[:, None] * feats_norm).sum(0)` → `(C,)`
5. L2-normalise prototype → `mu_A_norm = mu_A / ‖mu_A‖` → `(C,)` — average Phase A direction
6. Cosine similarity scores → `scores = feats_norm @ mu_A_norm` → `(N,)` ∈ [-1, 1]
7. Cast back to input dtype, reshape → `(1, 1, H_patch, W_patch)`

**No sign correction required.** Phase A patches cluster around `mu_A` by construction — they always score positively.

**Why this works:** XRD acts as a confidence weight at the patch level (spatial dimension). High-XRD patches strongly pull `mu_A` toward the true Phase A direction. Low-XRD patches barely contribute. The resulting prototype `mu_A_norm` encodes the average direction of Phase A in DINO's C-dimensional feature space. The final dot product reduces C dimensions to 1 scalar per patch — the collapse from high-dimensional to 1-channel output.

---

## Memory Considerations

The implementation holds two `(N, C)` float32 tensors simultaneously (`feats` and `feats_norm`). Memory requirements at a 2000px image:

| Stride | N | Two float32 (N, C) tensors |
|---|---|---|
| 4 | 250,000 | ~8 GB |
| 8 | 62,500 | ~2 GB |
| 16 | 15,625 | ~512 MB |
| 32 | ~3,800 | ~125 MB |

**Current assumption:** minimum 32GB GPU. With ~14GB occupied by 7B model weights, all stride values fit comfortably without chunking. This matches the memory profile of `_weighted_pca`, which also holds multiple full `(N, C)` float32 tensors.

**Future work:** If lower-memory GPUs need to be supported, a two-pass chunked implementation can replace the naive approach with no change to outputs or API — first pass accumulates `mu_A` in `chunk_size` blocks; second pass computes scores in the same blocks. Peak memory per chunk: `chunk_size × C × 4 bytes` (≈16 MB at chunk_size=1024, stride=4).

---

## Integration Points

### 1. `FusionOptions` type alias — `fusion.py:21`

```python
FusionOptions: TypeAlias = Literal[
    "gating", "learned_gating", "attention", "vanilla", "weighted_pca", "cosine_similarity"
]
```

### 2. `XRDFusionMethod.forward_xrd` — new branch alongside `weighted_pca`

```python
if xrd_fusion_method == "cosine_similarity":
    tr_xrd = self.get_tr()[idx]
    return self._cosine_similarity(x, tr_xrd)
```

### 3. `XFuse.forward_sequential` — extend `c_out` condition

```python
# replace:
c_out = 1 if self.xrd_fuse_method == "weighted_pca" else c

# with:
c_out = 1 if self.xrd_fuse_method in ("weighted_pca", "cosine_similarity") else c
```

---

## Output

`forward_sequential` returns `(1, 1, img_h, img_w)` when `xrd_fuse_method == "cosine_similarity"`.

Notebook usage:
```python
net.set_xrd_transforms(xrd_A_tensor, fwd, inv)
score_A = net.forward_sequential(xct_tensor)   # (1, 1, img_h, img_w)
score_map_A = score_A[0, 0]                    # (img_h, img_w) — ready for SAM2

net.set_xrd_transforms(xrd_B_tensor, fwd, inv)
score_B = net.forward_sequential(xct_tensor)
score_map_B = score_B[0, 0]
```

---

## Files Changed

| File | Change |
|---|---|
| `x_fuse/fusion.py` | Add `_cosine_similarity` to `XRDFusionMethod`; branch in `forward_xrd`; update `c_out`; update `FusionOptions` |
| `tests/test_fusion.py` | Tests for `_cosine_similarity` and `forward_xrd` dispatch |

---

## Out of Scope

- Chunked computation (future work, noted above)
- Changes to `train_fusion_step` (cosine_similarity requires no training)
- Changes to `XFuse.__init__` (no new modules needed)
- Relative scoring (score_A - score_B) — can be done in the notebook
