# xrd_embed_scale for NoPE/ALiBi DINOv3 — Design Spec

## Goal

Extend the `xrd_embed_scale` FusionOption to work correctly with DINOv3 NoPE and ALiBi
variants loaded via `get_dv3_model`. The current implementation applies `_fix_dv3_attn` to
all DINOv3 models, which bypasses `compute_attention` and silently destroys the ALiBi distance
bias injected by `_inject_alibi_dv3`.

---

## Problem: Why the Current Code Breaks ALiBi

`_inject_alibi_dv3` (in `x_fuse/utils.py`) injects the ALiBi distance bias by patching
`block.attn.compute_attention` on every block. The original DINOv3 `SelfAttention.forward`
calls `self.compute_attention(qkv)` internally, so the bias reaches every attention operation.

`_fix_dv3_attn` (used by `xrd_embed_scale` today) replaces `attn.forward` entirely and
performs attention **inline** — it never calls `self.compute_attention`. For ALiBi models this
silently drops all ALiBi distance bias. For NoPE models it happens to work (rope is always
None), but is fragile and inconsistent.

---

## Architecture

### New method: `Patch._fix_alibi_dv3_attn()`

A minimal wrapper around the original DINOv3 `SelfAttention.forward` with one addition: read
and apply `_xrd_scale` before QKV projection. Delegates attention computation to
`self.compute_attention(qkv)`, which preserves whatever is patched on that method.

Confirmed against DINOv3 source (`dinov3.layers.attention.SelfAttention`):

```python
# Original SelfAttention.forward (confirmed source):
def forward(self, x, attn_bias=None, rope=None):
    qkv = self.qkv(x)                                          # (B, N, 3*C) — 3D, no reshape
    attn_v = self.compute_attention(qkv=qkv, attn_bias=attn_bias, rope=rope)
    x = self.proj(attn_v)
    x = self.proj_drop(x)
    return x

# Original compute_attention (confirmed source):
def compute_attention(self, qkv, attn_bias=None, rope=None):
    assert attn_bias is None
    B, N, _ = qkv.shape                                        # expects 3D (B, N, 3*C)
    C = self.qkv.in_features
    qkv = qkv.reshape(B, N, 3, self.num_heads, C // self.num_heads)
    q, k, v = torch.unbind(qkv, 2)
    q, k, v = [t.transpose(1, 2) for t in [q, k, v]]
    if rope is not None:
        q, k = self.apply_rope(q, k, rope)                     # NoPE: skipped (rope=None)
    x = F.scaled_dot_product_attention(q, k, v)
    x = x.transpose(1, 2)
    return x.reshape([B, N, C])
```

`_fix_alibi_dv3_attn` is the original `forward` with `_xrd_scale` injection prepended:

```python
@staticmethod
def _fix_alibi_dv3_attn() -> Callable:
    def forward(self, x, attn_bias=None, rope=None):
        xrd_scale = getattr(self, "_xrd_scale", None)
        if xrd_scale is not None:
            self._xrd_scale = None
            x = x * xrd_scale.to(dtype=x.dtype, device=x.device)

        qkv = self.qkv(x)  # (B, N, 3*C) — 3D, no reshape; matches original forward exactly
        x = self.compute_attention(qkv=qkv, attn_bias=attn_bias, rope=rope)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x
    return forward
```

**Why `rope` is safe to pass**: for both NoPE and ALiBi models, the block never passes
`rope` to `attn.forward` (always `None`). NoPE's `compute_attention` skips `apply_rope`
when `rope is None`. ALiBi's patched `compute_attention` ignores `rope` entirely.

**Why `attn_bias=None` is safe**: NoPE's `compute_attention` asserts `attn_bias is None`
(confirmed in source). ALiBi's patched `compute_attention` ignores `attn_bias` entirely —
it reads `self.m` and the closed-over distance matrix directly. In both cases `attn_bias=None`
is correct.

### `patch_last_block` routing

```
vanilla_dv3   →  _fix_dv3_attn          (handles RoPE inline; no compute_attention call)
nope_dv3      →  _fix_alibi_dv3_attn    (delegates to standard compute_attention)
alibi_dv3     →  _fix_alibi_dv3_attn    (delegates to ALiBi-patched compute_attention)
```

The detection logic in `patch_last_block`:

```python
if self.xrd_fuse_method == "xrd_embed_scale":
    if "dv3" not in dino_name:
        raise ValueError(
            f"xrd_embed_scale requires a DINOv3 model, got '{dino_name}'"
        )
    inner = getattr(dino_model, "model", dino_model)
    patch_fn = (
        Patch._fix_dv3_attn()
        if "vanilla_dv3" in dino_name
        else Patch._fix_alibi_dv3_attn()
    )
    for blk in inner.blocks:
        blk.attn.forward = MethodType(patch_fn, blk.attn)
```

### `forward_sequential` — no changes needed

The XRD scale injection already works for all dv3 variants:

```python
inner = getattr(self.dinov2, "model", self.dinov2)
for blk in inner.blocks:
    blk.attn._xrd_scale = scale
```

`getattr(self.dinov2, "model", self.dinov2)` correctly unwraps `_DV3Wrapper` (which stores
the actual DINOv3 model as `self.model`) for vanilla, NoPE, and ALiBi models alike.

---

## Model Support Matrix

| Model name pattern       | Loader            | Positional encoding | Patch method          | Supported |
|--------------------------|-------------------|---------------------|-----------------------|-----------|
| `vanilla_dv3_vits*`      | `get_dv3_model`   | RoPE                | `_fix_dv3_attn`       | ✓ existing |
| `nope_dinov3_vits*`      | `get_dv3_model`   | None                | `_fix_alibi_dv3_attn` | ✓ this spec |
| `alibi_dinov3_vits*`     | `get_dv3_model`   | ALiBi               | `_fix_alibi_dv3_attn` | ✓ this spec |
| `alibi_vits*` (non-dv3)  | `get_alibi_model` | ALiBi (DINOv2-base) | —                     | ✗ out of scope |

---

## Out of Scope: DINOv2-based ALiBi (`PretrainedViTWrapper`)

Models loaded via `get_alibi_model` (`PretrainedViTWrapper`) are fine-tuned DINOv2 backbones
with ALiBi positional encoding. They differ architecturally from DINOv3:

- Use **LayerNorm** not RMSNorm — post-norm features have different statistical properties
- Have **no `compute_attention` method** — attention is handled via `_fix_mem_eff_attn`
- The existing `_xrd_bias` injection via `_fix_mem_eff_attn` already works for these models;
  `_xrd_scale` injection would require a separate patch method adapted for LayerNorm statistics

The `xrd_embed_scale` ValueError guard (`"dv3" not in dino_name`) correctly rejects these
models. Future work: a `_fix_alibi_dv2_attn` method analogous to `_fix_alibi_dv3_attn` but
adapted for `PretrainedViTWrapper`'s attention interface.

---

## Files Changed

| File | Change |
|---|---|
| `HR-Dv2/hr_dv2/patch.py` | Add `Patch._fix_alibi_dv3_attn()` static method |
| `x_fuse/fusion.py` | Update `patch_last_block` to route nope/alibi dv3 to new method |
| `tests/test_patch.py` | Add `MockAttnAlibiDV3` and 3 tests for `_fix_alibi_dv3_attn` |

---

## Tests

### Mock

`MockAttnAlibiDV3` mirrors the confirmed DINOv3 `SelfAttention` interface. `compute_attention`
expects 3D `qkv` — exactly as confirmed from source (`B, N, _ = qkv.shape`):

```python
class MockAttnAlibiDV3(torch.nn.Module):
    def __init__(self, feat_dim=64, num_heads=4):
        super().__init__()
        self.num_heads = num_heads
        self.qkv = torch.nn.Linear(feat_dim, feat_dim * 3, bias=False)
        self.proj = torch.nn.Linear(feat_dim, feat_dim, bias=False)
        self.proj_drop = torch.nn.Dropout(0.0)

    def compute_attention(self, qkv, attn_bias=None, rope=None):
        # Mirrors original compute_attention: expects 3D (B, N, 3*C)
        B, N, _ = qkv.shape
        C = self.qkv.in_features
        qkv_r = qkv.reshape(B, N, 3, self.num_heads, C // self.num_heads)
        q, k, v = torch.unbind(qkv_r, 2)
        q, k, v = [t.transpose(1, 2) for t in [q, k, v]]
        x = torch.nn.functional.scaled_dot_product_attention(q, k, v)
        return x.transpose(1, 2).reshape(B, N, C)

    def forward(self, x, attn_bias=None, rope=None):
        return x  # replaced by patch
```

### Logic trace

The full call chain exercised by the tests:

```
block.forward(x)
└── attn.forward(norm1(x), rope=None)          ← _fix_alibi_dv3_attn replaces this
    ├── read & clear _xrd_scale
    ├── x = x * scale  (if set)
    ├── qkv = self.qkv(x)                      → (B, N, 3*C), 3D
    ├── x = self.compute_attention(qkv=qkv, attn_bias=None, rope=None)
    │   ├── NoPE path:  assert attn_bias is None ✓
    │   │               reshape → q, k, v
    │   │               rope is None → skip apply_rope
    │   │               SDPA(q, k, v) → (B, N, C)
    │   └── ALiBi path: _compute_attn from _inject_alibi_dv3
    │                   reads self.m + distance_matrix
    │                   chunked SDPA with ALiBi bias → (B, N, C)
    ├── x = self.proj(x)
    └── x = self.proj_drop(x)
```

### Test list

**State tests** (parallel to `_fix_dv3_attn` tests):
- `test_alibi_dv3_xrd_scale_cleared_after_forward` — `_xrd_scale` is `None` after call
- `test_alibi_dv3_no_xrd_scale_does_not_error` — no `AttributeError` when scale absent
- `test_alibi_dv3_xrd_scale_changes_output` — scaled output differs from unscaled

**Logic trace tests** (verifies `compute_attention` is not bypassed):

```python
def test_alibi_dv3_compute_attention_called(patched_alibi_dv3_attn):
    """_fix_alibi_dv3_attn must delegate to compute_attention, not bypass it.
    Verifies qkv is 3D (B, N, 3*C) — matching the confirmed source signature."""
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
    assert call_log[0] == (B, N, C * 3), f"qkv must be 3D (B, N, 3*C), got {call_log[0]}"


def test_alibi_dv3_custom_compute_attention_preserved(patched_alibi_dv3_attn):
    """Patching attn.forward must not destroy a custom compute_attention (e.g. ALiBi bias).
    Replaces compute_attention with a constant-output stub and verifies the stub's
    output propagates through proj/proj_drop — confirming the custom implementation
    is called rather than bypassed."""
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

    assert not torch.allclose(out_stub, out_standard), \
        "custom compute_attention output must propagate — not be overridden inline"
```
