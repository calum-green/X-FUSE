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
