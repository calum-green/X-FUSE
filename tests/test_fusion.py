import pytest
import torch
from x_fuse.fusion import XRDCrossAttention, LearnedChannelGating, XRDFusionMethod


# ── XRDCrossAttention ──────────────────────────────────────────────────────────

def test_cross_attn_output_shape(dino_flavour, features, xrd_map_4d):
    model = XRDCrossAttention(
        feat_dim=dino_flavour["feat_dim"], num_heads=dino_flavour["num_heads"]
    )
    out = model(features, xrd_map_4d)
    assert out.shape == features.shape


def test_cross_attn_output_dtype(dino_flavour, features, xrd_map_4d):
    model = XRDCrossAttention(
        feat_dim=dino_flavour["feat_dim"], num_heads=dino_flavour["num_heads"]
    )
    out = model(features, xrd_map_4d)
    assert out.dtype == features.dtype


def test_cross_attn_output_is_tensor(dino_flavour, features, xrd_map_4d):
    model = XRDCrossAttention(
        feat_dim=dino_flavour["feat_dim"], num_heads=dino_flavour["num_heads"]
    )
    out = model(features, xrd_map_4d)
    assert isinstance(out, torch.Tensor)


def test_cross_attn_xrd_2d_input(dino_flavour, features):
    model = XRDCrossAttention(
        feat_dim=dino_flavour["feat_dim"], num_heads=dino_flavour["num_heads"]
    )
    xrd_2d = torch.rand(8, 8)
    out = model(features, xrd_2d)
    assert out.shape == features.shape


def test_cross_attn_xrd_3d_input(dino_flavour, features):
    model = XRDCrossAttention(
        feat_dim=dino_flavour["feat_dim"], num_heads=dino_flavour["num_heads"]
    )
    xrd_3d = torch.rand(1, 8, 8)
    out = model(features, xrd_3d)
    assert out.shape == features.shape
