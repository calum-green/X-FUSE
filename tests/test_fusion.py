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


# ── LearnedChannelGating ───────────────────────────────────────────────────────

def test_gating_bce_loss_scalar(dino_flavour, features, xrd_map_4d):
    gating = LearnedChannelGating(feat_dim=dino_flavour["feat_dim"], loss_fn="bce")
    loss = gating.compute_loss(features, xrd_map_4d)
    assert loss.ndim == 0
    assert loss.item() > 0


def test_gating_pearson_loss_scalar(dino_flavour, features, xrd_map_4d):
    gating = LearnedChannelGating(feat_dim=dino_flavour["feat_dim"], loss_fn="pearson")
    loss = gating.compute_loss(features, xrd_map_4d)
    assert loss.ndim == 0
    assert 0.0 <= loss.item() <= 2.0


def test_gating_forward_no_mask_passthrough(dino_flavour, features, xrd_map_4d):
    gating = LearnedChannelGating(feat_dim=dino_flavour["feat_dim"])
    result = gating(features, xrd_map_4d)
    assert result is features


def test_gating_select_top_k_channel_count(dino_flavour, features, xrd_map_4d):
    k = 10
    gating = LearnedChannelGating(feat_dim=dino_flavour["feat_dim"])
    gating.select_top_k(k)
    result = gating(features, xrd_map_4d)
    n_nonzero = (result.abs().sum(dim=(0, 2, 3)) > 0).sum().item()
    assert n_nonzero == k


def test_gating_forward_shape(dino_flavour, features, xrd_map_4d):
    gating = LearnedChannelGating(feat_dim=dino_flavour["feat_dim"])
    result = gating(features, xrd_map_4d)
    assert result.shape == features.shape


def test_gating_forward_dtype(dino_flavour, features, xrd_map_4d):
    gating = LearnedChannelGating(feat_dim=dino_flavour["feat_dim"])
    gating.select_top_k(10)
    result = gating(features, xrd_map_4d)
    assert result.dtype == features.dtype
