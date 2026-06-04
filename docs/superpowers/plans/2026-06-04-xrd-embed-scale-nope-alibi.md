# xrd_embed_scale NoPE/ALiBi DINOv3 Support — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Extend `xrd_embed_scale` to work correctly with DINOv3 NoPE and ALiBi variants by adding `_fix_alibi_dv3_attn` — a minimal wrapper around the confirmed `SelfAttention.forward` source that preserves `compute_attention` (and therefore ALiBi bias) while injecting `_xrd_scale`.

**Architecture:** A new `Patch._fix_alibi_dv3_attn()` static method in `patch.py` is identical to the confirmed DINOv3 `SelfAttention.forward` source with `_xrd_scale` injection prepended. `patch_last_block` in `fusion.py` routes nope/alibi dv3 models to the new method instead of `_fix_dv3_attn` (which bypasses `compute_attention` and silently destroys ALiBi bias). `forward_sequential` requires no changes.

**Tech Stack:** PyTorch `MethodType`, existing `Patch` pattern, `pytest`

**Spec:** `docs/superpowers/specs/2026-06-04-xrd-embed-scale-nope-alibi-design.md`

---

## File Map

| File | Change |
|---|---|
| `HR-Dv2/hr_dv2/patch.py:284` | Add `Patch._fix_alibi_dv3_attn()` after `_fix_dv3_attn` |
| `x_fuse/fusion.py:658-665` | Update `patch_last_block` xrd_embed_scale block to route by model type |
| `tests/test_patch.py:136` | Add `MockAttnAlibiDV3`, fixture, and 5 tests at end of file |

---

### Task 1: `_fix_alibi_dv3_attn` — TDD implementation

**Files:**
- Modify: `tests/test_patch.py` (append at end of file)
- Modify: `HR-Dv2/hr_dv2/patch.py:284` (insert after `return forward` / `return forward` of `_fix_dv3_attn`)

---

- [ ] **Step 1: Write the failing tests**

Append everything below to the end of `tests/test_patch.py`:

```python
# ── _fix_alibi_dv3_attn (NoPE / ALiBi DINOv3 SelfAttention) ──────────────────


class MockAttnAlibiDV3(torch.nn.Module):
    """Minimal mock of DINOv3 SelfAttention for NoPE/ALiBi models.

    compute_attention mirrors the confirmed dinov3.layers.attention.SelfAttention
    source: expects 3D qkv (B, N, 3*C), returns (B, N, C).
    forward is replaced entirely by _fix_alibi_dv3_attn.
    """

    def __init__(self, feat_dim: int = 64, num_heads: int = 4):
        super().__init__()
        self.num_heads = num_heads
        self.qkv = torch.nn.Linear(feat_dim, feat_dim * 3, bias=False)
        self.proj = torch.nn.Linear(feat_dim, feat_dim, bias=False)
        self.proj_drop = torch.nn.Dropout(0.0)

    def compute_attention(self, qkv, attn_bias=None, rope=None):
        # Mirrors confirmed source: B, N, _ = qkv.shape (3D input)
        B, N, _ = qkv.shape
        C = self.qkv.in_features
        qkv_r = qkv.reshape(B, N, 3, self.num_heads, C // self.num_heads)
        q, k, v = torch.unbind(qkv_r, 2)
        q, k, v = [t.transpose(1, 2) for t in [q, k, v]]
        x = torch.nn.functional.scaled_dot_product_attention(q, k, v)
        return x.transpose(1, 2).reshape(B, N, C)

    def forward(self, x, attn_bias=None, rope=None):
        return x  # replaced entirely by patch


@pytest.fixture
def patched_alibi_dv3_attn():
    attn = MockAttnAlibiDV3(feat_dim=64, num_heads=4)
    attn.forward = MethodType(Patch._fix_alibi_dv3_attn(), attn)
    return attn


# ── State tests ───────────────────────────────────────────────────────────────


def test_alibi_dv3_xrd_scale_cleared_after_forward(patched_alibi_dv3_attn):
    """_xrd_scale must be cleared after _fix_alibi_dv3_attn forward."""
    B, N, C = 1, 10, 64
    x = torch.randn(B, N, C)
    patched_alibi_dv3_attn._xrd_scale = torch.ones(1, N, 1) * 1.5
    patched_alibi_dv3_attn(x)
    assert getattr(patched_alibi_dv3_attn, "_xrd_scale", None) is None


def test_alibi_dv3_no_xrd_scale_does_not_error(patched_alibi_dv3_attn):
    """Calling _fix_alibi_dv3_attn forward without _xrd_scale must not raise AttributeError."""
    B, N, C = 1, 10, 64
    x = torch.randn(B, N, C)
    assert not hasattr(patched_alibi_dv3_attn, "_xrd_scale")
    try:
        patched_alibi_dv3_attn(x)
    except AttributeError:
        pytest.fail("forward raised AttributeError when _xrd_scale was not set")


def test_alibi_dv3_xrd_scale_changes_output(patched_alibi_dv3_attn):
    """Scaling post-norm x must produce a different output than unscaled."""
    B, N, C = 1, 10, 64
    x = torch.randn(B, N, C)
    out_baseline = patched_alibi_dv3_attn(x).detach().clone()
    patched_alibi_dv3_attn._xrd_scale = torch.ones(1, N, 1) * 2.0
    out_scaled = patched_alibi_dv3_attn(x).detach().clone()
    assert not torch.allclose(out_baseline, out_scaled)


# ── Logic trace tests ─────────────────────────────────────────────────────────


def test_alibi_dv3_compute_attention_called(patched_alibi_dv3_attn):
    """_fix_alibi_dv3_attn must delegate to compute_attention, not bypass it.
    Verifies qkv is 3D (B, N, 3*C) — matching the confirmed DINOv3 source signature."""
    B, N, C = 1, 10, 64
    x = torch.randn(B, N, C)
    call_log = []
    original = patched_alibi_dv3_attn.compute_attention

    def spy(qkv, attn_bias=None, rope=None):
        call_log.append(qkv.shape)
        return original(qkv, attn_bias=attn_bias, rope=rope)

    patched_alibi_dv3_attn.compute_attention = spy
    patched_alibi_dv3_attn(x)
    assert len(call_log) == 1, "compute_attention must be called exactly once"
    assert call_log[0] == (B, N, C * 3), (
        f"qkv must be 3D (B, N, 3*C)={( B, N, C*3)}, got {call_log[0]}"
    )


def test_alibi_dv3_custom_compute_attention_preserved(patched_alibi_dv3_attn):
    """Patching attn.forward must not destroy a custom compute_attention (e.g. ALiBi bias).
    Replaces compute_attention with a constant-output stub and verifies the stub's output
    propagates through proj/proj_drop — confirming the custom implementation is called,
    not bypassed inline."""
    B, N, C = 1, 10, 64
    x = torch.randn(B, N, C)
    sentinel = torch.ones(B, N, C) * 99.0

    def stub_compute(qkv, attn_bias=None, rope=None):
        return sentinel

    original = patched_alibi_dv3_attn.compute_attention
    patched_alibi_dv3_attn.compute_attention = stub_compute
    out_stub = patched_alibi_dv3_attn(x).detach().clone()

    patched_alibi_dv3_attn.compute_attention = original
    out_standard = patched_alibi_dv3_attn(x).detach().clone()

    assert not torch.allclose(out_stub, out_standard), (
        "custom compute_attention output must propagate — not be overridden inline"
    )
```

- [ ] **Step 2: Run tests to confirm they fail**

```bash
pytest tests/test_patch.py::test_alibi_dv3_xrd_scale_cleared_after_forward \
       tests/test_patch.py::test_alibi_dv3_no_xrd_scale_does_not_error \
       tests/test_patch.py::test_alibi_dv3_xrd_scale_changes_output \
       tests/test_patch.py::test_alibi_dv3_compute_attention_called \
       tests/test_patch.py::test_alibi_dv3_custom_compute_attention_preserved \
       -v
```

Expected: all 5 FAIL with `AttributeError: type object 'Patch' has no attribute '_fix_alibi_dv3_attn'`

- [ ] **Step 3: Add `_fix_alibi_dv3_attn` to `HR-Dv2/hr_dv2/patch.py`**

Insert the following immediately after line 284 (`        return forward`) of `_fix_dv3_attn`,
before the `_fix_block_forward_dino` method at line 286:

```python
    @staticmethod
    def _fix_alibi_dv3_attn() -> Callable:
        """Patches NoPE/ALiBi DINOv3 SelfAttention.forward to inject _xrd_scale.

        Unlike _fix_dv3_attn (for vanilla_dv3 with RoPE), this method preserves the
        existing compute_attention call — critical for ALiBi models where compute_attention
        is patched by _inject_alibi_dv3 to add ALiBi distance bias. For NoPE models,
        compute_attention is the standard attention without positional encoding.

        Implementation is the confirmed SelfAttention.forward source with _xrd_scale
        injection prepended. qkv is passed as 3D (B, N, 3*C) — the shape compute_attention
        expects (confirmed from dinov3.layers.attention source).
        """

        def forward(
            self,
            x: torch.Tensor,
            attn_bias=None,
            rope: torch.Tensor = None,
        ) -> torch.Tensor:
            xrd_scale = getattr(self, "_xrd_scale", None)
            if xrd_scale is not None:
                self._xrd_scale = None
                x = x * xrd_scale.to(dtype=x.dtype, device=x.device)

            qkv = self.qkv(x)  # (B, N, 3*C) — 3D, no reshape; matches original forward
            x = self.compute_attention(qkv=qkv, attn_bias=attn_bias, rope=rope)
            x = self.proj(x)
            x = self.proj_drop(x)
            return x

        return forward

```

- [ ] **Step 4: Run tests to confirm they pass**

```bash
pytest tests/test_patch.py::test_alibi_dv3_xrd_scale_cleared_after_forward \
       tests/test_patch.py::test_alibi_dv3_no_xrd_scale_does_not_error \
       tests/test_patch.py::test_alibi_dv3_xrd_scale_changes_output \
       tests/test_patch.py::test_alibi_dv3_compute_attention_called \
       tests/test_patch.py::test_alibi_dv3_custom_compute_attention_preserved \
       -v
```

Expected: all 5 PASS

- [ ] **Step 5: Run full test suite for regressions**

```bash
pytest tests/ -q --tb=short
```

Expected: 462 passed, 4 pre-existing failures (2 CUDA-only, 2 segmentation)

- [ ] **Step 6: Commit**

```bash
git add HR-Dv2/hr_dv2/patch.py tests/test_patch.py
git commit -m "feat: add _fix_alibi_dv3_attn for NoPE/ALiBi DINOv3 xrd_embed_scale support"
```

---

### Task 2: Update `patch_last_block` routing in `fusion.py`

**Files:**
- Modify: `x_fuse/fusion.py:658-665`

The current block at lines 658-665 applies `_fix_dv3_attn` to ALL dv3 models. It must
route vanilla_dv3 to `_fix_dv3_attn` (handles RoPE inline) and nope/alibi to
`_fix_alibi_dv3_attn` (preserves compute_attention).

This change has no isolated unit test (requires a live DINOv3 model). Correctness is
verified by running the full unit suite for regressions, then validated in the notebook.

- [ ] **Step 1: Update `patch_last_block` in `x_fuse/fusion.py`**

Replace lines 658-665:

```python
        if self.xrd_fuse_method == "xrd_embed_scale":
            if "dv3" not in dino_name:
                raise ValueError(
                    f"xrd_embed_scale requires a DINOv3 model, got '{dino_name}'"
                )
            inner = getattr(dino_model, "model", dino_model)
            for blk in inner.blocks:
                blk.attn.forward = MethodType(Patch._fix_dv3_attn(), blk.attn)
```

With:

```python
        if self.xrd_fuse_method == "xrd_embed_scale":
            if "dv3" not in dino_name:
                raise ValueError(
                    f"xrd_embed_scale requires a DINOv3 model, got '{dino_name}'"
                )
            inner = getattr(dino_model, "model", dino_model)
            # vanilla_dv3 uses RoPE and does attention inline (_fix_dv3_attn).
            # nope/alibi dv3 must preserve compute_attention (_fix_alibi_dv3_attn),
            # which carries the ALiBi distance bias injected by _inject_alibi_dv3.
            patch_fn = (
                Patch._fix_dv3_attn()
                if "vanilla_dv3" in dino_name
                else Patch._fix_alibi_dv3_attn()
            )
            for blk in inner.blocks:
                blk.attn.forward = MethodType(patch_fn, blk.attn)
```

- [ ] **Step 2: Run full test suite for regressions**

```bash
pytest tests/ -q --tb=short
```

Expected: 462 passed, 4 pre-existing failures (2 CUDA-only, 2 segmentation)

- [ ] **Step 3: Commit**

```bash
git add x_fuse/fusion.py
git commit -m "feat: route nope/alibi dv3 to _fix_alibi_dv3_attn in patch_last_block"
```
