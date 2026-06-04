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

```python
@staticmethod
def _fix_alibi_dv3_attn() -> Callable:
    def forward(self, x, attn_bias=None, rope=None):
        xrd_scale = getattr(self, "_xrd_scale", None)
        if xrd_scale is not None:
            self._xrd_scale = None
            x = x * xrd_scale.to(dtype=x.dtype, device=x.device)

        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3 * self.num_heads, C // self.num_heads)
        x = self.compute_attention(qkv, attn_bias=attn_bias, rope=rope)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x
    return forward
```

**Why `rope` is safe to pass**: for both NoPE and ALiBi models, the block never passes
`rope` to `attn.forward` (rope is always `None`). `compute_attention` accepts it as a
keyword arg and ignores it.

**Why `attn_bias=None` is safe**: NoPE's `compute_attention` asserts `attn_bias is None`.
ALiBi's patched `compute_attention` ignores `attn_bias` entirely (it reads `self.m` and the
closed-over distance matrix directly). In both cases `attn_bias=None` is correct.

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

`MockAttnAlibiDV3` — mock with `compute_attention` that returns a plausible `(B, N, C)`
tensor:

```python
class MockAttnAlibiDV3(torch.nn.Module):
    def __init__(self, feat_dim=64, num_heads=4):
        super().__init__()
        self.num_heads = num_heads
        self.qkv = torch.nn.Linear(feat_dim, feat_dim * 3, bias=False)
        self.proj = torch.nn.Linear(feat_dim, feat_dim, bias=False)
        self.proj_drop = torch.nn.Dropout(0.0)

    def compute_attention(self, qkv, attn_bias=None, rope=None):
        B, N, _ = qkv.shape
        C = self.qkv.in_features
        return qkv.reshape(B, N, 3, self.num_heads, C // self.num_heads)[:, :, 0].reshape(B, N, C)

    def forward(self, x, attn_bias=None, rope=None):
        return x  # replaced by patch
```

Tests:
- `test_alibi_dv3_xrd_scale_cleared_after_forward` — `_xrd_scale` is None after call
- `test_alibi_dv3_no_xrd_scale_does_not_error` — no AttributeError when scale absent
- `test_alibi_dv3_xrd_scale_changes_output` — scaled output differs from unscaled
- `test_alibi_dv3_compute_attention_called` (optional) — verify `compute_attention` is invoked
