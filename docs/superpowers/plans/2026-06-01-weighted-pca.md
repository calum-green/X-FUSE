# weighted_pca Fusion Method Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add `weighted_pca` as a new `FusionOptions` value in `x_fuse/fusion.py` that produces a per-patch phase score map by running XRD-weighted PCA on DINO patch features, aggregated to full XCT resolution via HR-Dv2.

**Architecture:** `_weighted_pca` is a new method on `XRDFusionMethod` that takes `(1, C, H_patch, W_patch)` DINO features and a `(1, 1, H_xrd, W_xrd)` XRD map and returns a `(1, 1, H_patch, W_patch)` score map. It is routed through the existing `forward_xrd` dispatch. `forward_sequential` uses a dynamic accumulator sized to `c_out=1` for `weighted_pca` so the rest of the loop is unchanged.

**Tech Stack:** PyTorch (`torch.pca_lowrank`, `F.interpolate`), pytest

---

## Files

| File | Action |
|---|---|
| `x_fuse/fusion.py` | Modify — add `_weighted_pca`, branch in `forward_xrd`, 2-line change in `forward_sequential`, update `FusionOptions` |
| `tests/test_fusion.py` | Modify — add tests for `_weighted_pca` and `forward_xrd` dispatch |

---

## Task 1: Implement and test `_weighted_pca`

**Files:**
- Modify: `tests/test_fusion.py`
- Modify: `x_fuse/fusion.py`

- [ ] **Step 1: Write failing tests**

Append to `tests/test_fusion.py`:

```python
# ── XRDFusionMethod._weighted_pca ─────────────────────────────────────────────


def _make_fusion(xrd_map_4d):
    return XRDFusionMethod(
        xrd_img=xrd_map_4d,
        transform=[],
        require_grad=False,
        learned_gating=None,
        spatial_attention=None,
        loss_fn="bce",
    )


def test_weighted_pca_output_shape(dino_flavour, features, xrd_map_4d):
    fusion = _make_fusion(xrd_map_4d)
    result = fusion._weighted_pca(features, xrd_map_4d)
    _, _, H, W = features.shape
    assert result.shape == (1, 1, H, W)


def test_weighted_pca_output_dtype(dino_flavour, features, xrd_map_4d):
    fusion = _make_fusion(xrd_map_4d)
    result = fusion._weighted_pca(features, xrd_map_4d)
    assert result.dtype == features.dtype


def test_weighted_pca_output_is_tensor(dino_flavour, features, xrd_map_4d):
    fusion = _make_fusion(xrd_map_4d)
    result = fusion._weighted_pca(features, xrd_map_4d)
    assert isinstance(result, torch.Tensor)


def test_weighted_pca_xrd_spatial_mismatch(dino_flavour, features, xrd_map_4d):
    fusion = _make_fusion(xrd_map_4d)
    xrd_mismatched = torch.rand(1, 1, 4, 4)
    _, _, H, W = features.shape
    result = fusion._weighted_pca(features, xrd_mismatched)
    assert result.shape == (1, 1, H, W)


def test_weighted_pca_sign_positive_in_high_xrd_region(dino_flavour, features, xrd_map_4d):
    fusion = _make_fusion(xrd_map_4d)
    result = fusion._weighted_pca(features, xrd_map_4d)
    scores = result.float().reshape(-1)
    w = xrd_map_4d.float().reshape(-1)
    w = w / (w.sum() + 1e-8)
    # sign correction guarantees weighted sum of scores is non-negative
    assert (scores * w).sum().item() >= 0
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pytest tests/test_fusion.py -k "weighted_pca" -v
```

Expected: 5 × `FAILED` with `AttributeError: '_weighted_pca'`

- [ ] **Step 3: Implement `_weighted_pca` in `XRDFusionMethod`**

Add after `_spatial_attention` in `x_fuse/fusion.py` (before the `XFuse` class definition at line 334):

```python
    def _weighted_pca(self, x: torch.Tensor, xrd_map: torch.Tensor) -> torch.Tensor:
        """XRD-weighted PCA fusion.

        Finds the direction in channel space maximally common in high-XRD patches
        by computing the first eigenvector of the XRD-weighted feature covariance,
        then projects all patches onto that direction.

        Args:
            x: (1, C, H_patch, W_patch) DINO features at patch level
            xrd_map: (1, 1, H_xrd, W_xrd) pre-transformed XRD map

        Returns:
            (1, 1, H_patch, W_patch) continuous per-patch phase score;
            high score = high XRD intensity.
        """
        _, C, H, W = x.shape
        N = H * W

        # Interpolate XRD to patch grid (same as gating — never upsampled to XCT res)
        xrd_patch = F.interpolate(
            xrd_map.float(),
            size=(H, W),
            mode="bilinear",
            align_corners=False,
        ).reshape(N)  # (N,)

        # Normalise weights to sum=1
        w = xrd_patch / (xrd_patch.sum() + 1e-8)  # (N,)

        # Flatten features to (N, C), cast to float32 for numerical stability
        feats = x.float().reshape(C, N).permute(1, 0)  # (N, C)

        # Weighted mean
        mu = (w[:, None] * feats).sum(0)  # (C,)

        # Centre and apply sqrt weights:
        # eigenvectors of X_c^T diag(w) X_c = right singular vectors of diag(w)^0.5 X_c
        centered = feats - mu  # (N, C)
        wc = w.sqrt()[:, None] * centered  # (N, C)

        # First principal component via low-rank SVD
        _, _, V = torch.pca_lowrank(wc, q=1, center=False, niter=4)
        v = V[:, 0]  # (C,)

        # Score all patches
        scores = feats @ v  # (N,)

        # Sign correction: ensure high score = high XRD intensity
        if (scores * w).sum() < 0:
            scores = -scores

        return scores.to(x.dtype).reshape(1, 1, H, W)
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
pytest tests/test_fusion.py -k "weighted_pca" -v
```

Expected: 5 × `PASSED` (parameterized over all 9 DINO flavours = 45 total)

- [ ] **Step 5: Commit**

```bash
git add tests/test_fusion.py x_fuse/fusion.py
git commit -m "Add _weighted_pca method to XRDFusionMethod"
```

---

## Task 2: Wire `weighted_pca` into `forward_xrd`

**Files:**
- Modify: `tests/test_fusion.py`
- Modify: `x_fuse/fusion.py`

- [ ] **Step 1: Write failing test**

Append to `tests/test_fusion.py`:

```python
# ── XRDFusionMethod.forward_xrd dispatch ──────────────────────────────────────


def test_forward_xrd_weighted_pca_shape(dino_flavour, features, xrd_map_4d):
    fusion = _make_fusion(xrd_map_4d)
    result = fusion.forward_xrd(features, idx=0, xrd_fusion_method="weighted_pca")
    _, _, H, W = features.shape
    assert result.shape == (1, 1, H, W)


def test_forward_xrd_weighted_pca_dtype(dino_flavour, features, xrd_map_4d):
    fusion = _make_fusion(xrd_map_4d)
    result = fusion.forward_xrd(features, idx=0, xrd_fusion_method="weighted_pca")
    assert result.dtype == features.dtype
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pytest tests/test_fusion.py -k "forward_xrd_weighted_pca" -v
```

Expected: 2 × `FAILED` with `AttributeError: 'NoneType' object has no attribute 'shape'` — `forward_xrd` falls through all branches and returns `None` for `"weighted_pca"`.

- [ ] **Step 3: Add branch in `forward_xrd`**

In `x_fuse/fusion.py`, update `forward_xrd` (currently at line 175). Replace:

```python
    def forward_xrd(
        self,
        x: torch.Tensor,  # [B,C,H,W] transformed DINO features
        idx: int,  # index of the transform applied to the input image
        xrd_fusion_method: FusionOptions,
        top_k: int | None = None,
    ):
        # return transformed XRD features ready to be fused with the DINO features
        if xrd_fusion_method == "vanilla":
            return x

        tr_xrd = self.get_tr()[idx]
```

With:

```python
    def forward_xrd(
        self,
        x: torch.Tensor,  # [B,C,H,W] transformed DINO features
        idx: int,  # index of the transform applied to the input image
        xrd_fusion_method: FusionOptions,
        top_k: int | None = None,
    ):
        # return transformed XRD features ready to be fused with the DINO features
        if xrd_fusion_method == "vanilla":
            return x

        if xrd_fusion_method == "weighted_pca":
            tr_xrd = self.get_tr()[idx]
            return self._weighted_pca(x, tr_xrd)

        tr_xrd = self.get_tr()[idx]
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
pytest tests/test_fusion.py -k "forward_xrd_weighted_pca" -v
```

Expected: 2 × `PASSED` (parameterized over 9 DINO flavours = 18 total)

- [ ] **Step 5: Commit**

```bash
git add tests/test_fusion.py x_fuse/fusion.py
git commit -m "Wire weighted_pca into forward_xrd dispatch"
```

---

## Task 3: Update `FusionOptions` and `forward_sequential`

**Files:**
- Modify: `x_fuse/fusion.py`

- [ ] **Step 1: Update `FusionOptions` type alias**

In `x_fuse/fusion.py` line 21, replace:

```python
FusionOptions: TypeAlias = Literal["gating", "learned_gating", "attention", "vanilla"]
```

With:

```python
FusionOptions: TypeAlias = Literal["gating", "learned_gating", "attention", "vanilla", "weighted_pca"]
```

- [ ] **Step 2: Update `forward_sequential` accumulator**

In `x_fuse/fusion.py`, in `XFuse.forward_sequential`, replace the single line:

```python
        out_feature_img = torch.zeros(1, c, img_h, img_w, dtype=self.dtype)
```

With:

```python
        c_out = 1 if self.xrd_fuse_method == "weighted_pca" else c
        out_feature_img = torch.zeros(1, c_out, img_h, img_w, dtype=self.dtype)
```

- [ ] **Step 3: Run full test suite**

```bash
pytest tests/ -v
```

Expected: all existing tests pass; no regressions.

- [ ] **Step 4: Commit**

```bash
git add x_fuse/fusion.py
git commit -m "Add weighted_pca to FusionOptions; dynamic accumulator in forward_sequential"
```

---

## Task 4: Manual notebook verification

No unit test covers `forward_sequential` end-to-end (requires real DINO weights). Verify via notebook.

- [ ] **Step 1: In your notebook, instantiate XFuse with `FUSION_METHOD = "weighted_pca"`**

```python
FUSION_METHOD = "weighted_pca"

net = XFuse(
    DINO_MODEL,
    FUSION_METHOD,
    stride=STRIDE,
    pca_dim=-1,
    track_grad=False,
    dtype=torch.float16,
    device=DEVICE,
    loss_fn=LOSS_FN,
    model_path=MODEL_PATH,
    chk_path=MODEL_PATH if "dv3" in DINO_MODEL else None,
    lib_path=LIB_PATH if "dv3" in DINO_MODEL else None,
)
```

- [ ] **Step 2: Run `forward_sequential` for phase A and verify output shape**

```python
net.set_xrd_transforms(xrd_A_tensor, fwd, inv)
score_A = net.forward_sequential(xct_tensor)
print(score_A.shape)   # expect torch.Size([1, 1, img_h, img_w])

score_map_A = score_A[0, 0]   # (img_h, img_w)
plt.imshow(score_map_A.numpy(), cmap="viridis")
plt.title("Phase A score map")
plt.colorbar()
plt.show()
```

- [ ] **Step 3: Run for phase B and visualise both**

```python
net.set_xrd_transforms(xrd_B_tensor, fwd, inv)
score_B = net.forward_sequential(xct_tensor)
score_map_B = score_B[0, 0]

fig, axs = plt.subplots(1, 3, figsize=(15, 5))
axs[0].imshow(xct_img, cmap="gray")
axs[0].set_title("XCT")
axs[1].imshow(score_map_A.numpy(), cmap="viridis")
axs[1].set_title("Phase A score")
axs[2].imshow(score_map_B.numpy(), cmap="viridis")
axs[2].set_title("Phase B score")
plt.tight_layout()
plt.show()
```

Expected: high-scoring regions in score_A align spatially with high-XRD_A regions and vice versa.
