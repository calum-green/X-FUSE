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
