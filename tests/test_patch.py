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
    assert getattr(patched_attn, "_xrd_bias", None) is None


def test_no_xrd_bias_does_not_error(patched_attn):
    """Calling forward without _xrd_bias set must not raise AttributeError."""
    B, N, C = 1, 10, 64
    x = torch.randn(B, N, C)
    assert not hasattr(patched_attn, "_xrd_bias")
    try:
        patched_attn(x)
    except AttributeError:
        pytest.fail("forward raised AttributeError when _xrd_bias was not set")
    except Exception:
        pass  # other errors (e.g. no xformers) are acceptable


# ── _fix_dv3_attn (vanilla_dv3 SelfAttention) ────────────────────────────────


class MockAttnDV3(torch.nn.Module):
    """Minimal mock of DINOv3 SelfAttention with the interface _fix_dv3_attn expects:
    qkv, compute_attention, proj, proj_drop."""

    def __init__(self, feat_dim: int = 64, num_heads: int = 4):
        super().__init__()
        self.num_heads = num_heads
        self.qkv = torch.nn.Linear(feat_dim, feat_dim * 3, bias=False)
        self.proj = torch.nn.Linear(feat_dim, feat_dim, bias=False)
        self.proj_drop = torch.nn.Dropout(0.0)

    def apply_rope(self, q, k, rope):
        return q, k  # identity — no real rope in mock

    def compute_attention(self, qkv, attn_bias=None, rope=None):
        C = qkv.shape[-1] // 3
        return qkv[:, :, :C]  # return q-slice as proxy; shape (B, N, C)

    def forward(self, x, attn_bias=None, rope=None):
        return x  # replaced entirely by the patch


@pytest.fixture
def patched_dv3_attn():
    attn = MockAttnDV3(feat_dim=64, num_heads=4)
    attn.forward = MethodType(Patch._fix_dv3_attn(), attn)
    return attn


def test_dv3_xrd_bias_cleared_after_forward(patched_dv3_attn):
    """_xrd_bias must be cleared after _fix_dv3_attn forward."""
    B, N, C = 1, 10, 64
    x = torch.randn(B, N, C)
    xrd_bias = torch.zeros(1, 1, 1, N)
    patched_dv3_attn._xrd_bias = xrd_bias
    patched_dv3_attn(x)
    assert getattr(patched_dv3_attn, "_xrd_bias", None) is None


def test_dv3_no_xrd_bias_does_not_error(patched_dv3_attn):
    """Calling _fix_dv3_attn forward without _xrd_bias must not raise AttributeError."""
    B, N, C = 1, 10, 64
    x = torch.randn(B, N, C)
    assert not hasattr(patched_dv3_attn, "_xrd_bias")
    try:
        patched_dv3_attn(x)
    except AttributeError:
        pytest.fail("forward raised AttributeError when _xrd_bias was not set")


def test_dv3_rope_kwarg_accepted(patched_dv3_attn):
    """rope= must be forwarded to compute_attention without TypeError."""
    B, N, C = 1, 10, 64
    x = torch.randn(B, N, C)
    rope = torch.randn(N, C // 4)
    patched_dv3_attn(x, rope=rope)  # must not raise


def test_dv3_xrd_scale_cleared_after_forward(patched_dv3_attn):
    """_xrd_scale must be cleared after forward."""
    B, N, C = 1, 10, 64
    x = torch.randn(B, N, C)
    patched_dv3_attn._xrd_scale = torch.ones(1, N, 1) * 1.5
    patched_dv3_attn(x)
    assert getattr(patched_dv3_attn, "_xrd_scale", None) is None


def test_dv3_xrd_scale_changes_output(patched_dv3_attn):
    """Scaling post-RMSNorm x must produce a different output than unscaled."""
    B, N, C = 1, 10, 64
    x = torch.randn(B, N, C)
    out_baseline = patched_dv3_attn(x).detach().clone()
    patched_dv3_attn._xrd_scale = torch.ones(1, N, 1) * 2.0
    out_scaled = patched_dv3_attn(x).detach().clone()
    assert not torch.allclose(out_baseline, out_scaled)


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
    """Calling _fix_alibi_dv3_attn forward without _xrd_scale must not
    raise AttributeError."""
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
    assert call_log[0] == (
        B,
        N,
        C * 3,
    ), f"qkv must be 3D (B, N, 3*C)={(B, N, C * 3)}, got {call_log[0]}"


def test_alibi_dv3_custom_compute_attention_preserved(patched_alibi_dv3_attn):
    """Patching attn.forward must not destroy a custom compute_attention.
    Replaces compute_attention with a constant-output stub and verifies the
    stub's output propagates through proj/proj_drop — confirming the custom
    implementation is called,
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

    assert not torch.allclose(
        out_stub, out_standard
    ), "custom compute_attention output must propagate — not be overridden inline"
