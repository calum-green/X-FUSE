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


# ── XRDFusionMethod._direct_correlation_gating ────────────────────────────────


@pytest.mark.parametrize("loss_fn", ["pearson", "bce"])
def test_dcg_output_shape(dino_flavour, features, xrd_map_4d, loss_fn):
    fusion = XRDFusionMethod(
        xrd_img=xrd_map_4d,
        transform=[],
        require_grad=False,
        learned_gating=None,
        spatial_attention=None,
        loss_fn=loss_fn,
    )
    result = fusion._direct_correlation_gating(features, xrd_map_4d)
    assert result.shape == features.shape


@pytest.mark.parametrize("loss_fn", ["pearson", "bce"])
def test_dcg_output_dtype(dino_flavour, features, xrd_map_4d, loss_fn):
    fusion = XRDFusionMethod(
        xrd_img=xrd_map_4d,
        transform=[],
        require_grad=False,
        learned_gating=None,
        spatial_attention=None,
        loss_fn=loss_fn,
    )
    result = fusion._direct_correlation_gating(features, xrd_map_4d)
    assert result.dtype == features.dtype


@pytest.mark.parametrize("loss_fn", ["pearson", "bce"])
def test_dcg_output_is_tensor(dino_flavour, features, xrd_map_4d, loss_fn):
    fusion = XRDFusionMethod(
        xrd_img=xrd_map_4d,
        transform=[],
        require_grad=False,
        learned_gating=None,
        spatial_attention=None,
        loss_fn=loss_fn,
    )
    result = fusion._direct_correlation_gating(features, xrd_map_4d)
    assert isinstance(result, torch.Tensor)


@pytest.mark.parametrize("loss_fn", ["pearson", "bce"])
def test_dcg_gate_values_in_range(dino_flavour, features, xrd_map_4d, loss_fn):
    # Gates are in [0, 1], so |output[c]| <= |features[c]| elementwise.
    fusion = XRDFusionMethod(
        xrd_img=xrd_map_4d,
        transform=[],
        require_grad=False,
        learned_gating=None,
        spatial_attention=None,
        loss_fn=loss_fn,
    )
    result = fusion._direct_correlation_gating(features, xrd_map_4d)
    assert torch.all(result.abs() <= features.abs() + 1e-6)


@pytest.mark.parametrize("loss_fn", ["pearson"])
def test_dcg_top_k_zeroing(dino_flavour, features, xrd_map_4d, loss_fn):
    # BCE can legitimately produce all-zero gates when BCE > log(2) for all
    # channels with random inputs, so this channel-count assertion is pearson-only.
    k = 10
    fusion = XRDFusionMethod(
        xrd_img=xrd_map_4d,
        transform=[],
        require_grad=False,
        learned_gating=None,
        spatial_attention=None,
        loss_fn=loss_fn,
    )
    result = fusion._direct_correlation_gating(features, xrd_map_4d, top_k=k)
    n_nonzero = (result.abs().sum(dim=(0, 2, 3)) > 0).sum().item()
    assert n_nonzero == k


@pytest.mark.parametrize("loss_fn", ["pearson"])
def test_dcg_default_top_k(dino_flavour, features, xrd_map_4d, loss_fn):
    # BCE can legitimately produce all-zero gates when BCE > log(2) for all
    # channels with random inputs, so this channel-count assertion is pearson-only.
    fusion = XRDFusionMethod(
        xrd_img=xrd_map_4d,
        transform=[],
        require_grad=False,
        learned_gating=None,
        spatial_attention=None,
        loss_fn=loss_fn,
    )
    result = fusion._direct_correlation_gating(features, xrd_map_4d)
    expected_k = features.shape[1] // 4
    n_nonzero = (result.abs().sum(dim=(0, 2, 3)) > 0).sum().item()
    assert n_nonzero == expected_k


def test_dcg_bce_top_k_pipeline(dino_flavour, features, xrd_map_4d):
    # Verifies the BCE path runs without error for both explicit and default top_k.
    fusion = XRDFusionMethod(
        xrd_img=xrd_map_4d,
        transform=[],
        require_grad=False,
        learned_gating=None,
        spatial_attention=None,
        loss_fn="bce",
    )
    result_explicit = fusion._direct_correlation_gating(features, xrd_map_4d, top_k=10)
    result_default = fusion._direct_correlation_gating(features, xrd_map_4d)
    assert result_explicit.shape == features.shape
    assert result_explicit.dtype == features.dtype
    assert result_default.shape == features.shape
    assert result_default.dtype == features.dtype


@pytest.mark.parametrize("loss_fn", ["pearson", "bce"])
def test_dcg_xrd_spatial_mismatch(dino_flavour, features, xrd_map_4d, loss_fn):
    fusion = XRDFusionMethod(
        xrd_img=xrd_map_4d,
        transform=[],
        require_grad=False,
        learned_gating=None,
        spatial_attention=None,
        loss_fn=loss_fn,
    )
    xrd_mismatched = torch.rand(1, 1, 4, 4)
    result = fusion._direct_correlation_gating(features, xrd_mismatched)
    assert result.shape == features.shape


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


def test_weighted_pca_sign_positive_in_high_xrd_region(
    dino_flavour, features, xrd_map_4d
):
    fusion = _make_fusion(xrd_map_4d)
    result = fusion._weighted_pca(features, xrd_map_4d)
    scores = result.float().reshape(-1)
    w = xrd_map_4d.float().reshape(-1)
    w = w / (w.sum() + 1e-8)
    # sign correction guarantees weighted sum of scores is non-negative
    assert (scores * w).sum().item() >= 0


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


# ── XRDFusionMethod._cosine_similarity ────────────────────────────────────────


def test_cosine_similarity_output_shape(dino_flavour, features, xrd_map_4d):
    fusion = _make_fusion(xrd_map_4d)
    result = fusion._cosine_similarity(features, xrd_map_4d)
    _, _, H, W = features.shape
    assert result.shape == (1, 1, H, W)


def test_cosine_similarity_output_dtype(dino_flavour, features, xrd_map_4d):
    fusion = _make_fusion(xrd_map_4d)
    result = fusion._cosine_similarity(features, xrd_map_4d)
    assert result.dtype == features.dtype


def test_cosine_similarity_output_is_tensor(dino_flavour, features, xrd_map_4d):
    fusion = _make_fusion(xrd_map_4d)
    result = fusion._cosine_similarity(features, xrd_map_4d)
    assert isinstance(result, torch.Tensor)


def test_cosine_similarity_xrd_spatial_mismatch(dino_flavour, features, xrd_map_4d):
    fusion = _make_fusion(xrd_map_4d)
    xrd_mismatched = torch.rand(1, 1, 4, 4)
    _, _, H, W = features.shape
    result = fusion._cosine_similarity(features, xrd_mismatched)
    assert result.shape == (1, 1, H, W)


def test_cosine_similarity_scores_in_range(dino_flavour, features, xrd_map_4d):
    fusion = _make_fusion(xrd_map_4d)
    result = fusion._cosine_similarity(features, xrd_map_4d)
    scores = result.float()
    assert scores.min().item() >= -1.0 - 1e-5
    assert scores.max().item() <= 1.0 + 1e-5


def test_cosine_similarity_high_xrd_scores_positive(dino_flavour, features, xrd_map_4d):
    fusion = _make_fusion(xrd_map_4d)
    result = fusion._cosine_similarity(features, xrd_map_4d)
    scores = result.float().reshape(-1)
    w = xrd_map_4d.float().reshape(-1)
    w = w / (w.sum() + 1e-8)
    assert (scores * w).sum().item() >= 0


# ── XRDFusionMethod.forward_xrd cosine_similarity dispatch ────────────────────


def test_forward_xrd_cosine_similarity_shape(dino_flavour, features, xrd_map_4d):
    fusion = _make_fusion(xrd_map_4d)
    result = fusion.forward_xrd(features, idx=0, xrd_fusion_method="cosine_similarity")
    _, _, H, W = features.shape
    assert result.shape == (1, 1, H, W)


def test_forward_xrd_cosine_similarity_dtype(dino_flavour, features, xrd_map_4d):
    fusion = _make_fusion(xrd_map_4d)
    result = fusion.forward_xrd(features, idx=0, xrd_fusion_method="cosine_similarity")
    assert result.dtype == features.dtype


# ── XRDFusionMethod.forward_xrd xrd_attn_weight dispatch ─────────────────────


def test_forward_xrd_attn_weight_returns_x(dino_flavour, features, xrd_map_4d):
    fusion = _make_fusion(xrd_map_4d)
    result = fusion.forward_xrd(features, idx=0, xrd_fusion_method="xrd_attn_weight")
    assert result is features


def test_forward_xrd_attn_weight_shape(dino_flavour, features, xrd_map_4d):
    fusion = _make_fusion(xrd_map_4d)
    result = fusion.forward_xrd(features, idx=0, xrd_fusion_method="xrd_attn_weight")
    assert result.shape == features.shape


def test_forward_xrd_attn_weight_dtype(dino_flavour, features, xrd_map_4d):
    fusion = _make_fusion(xrd_map_4d)
    result = fusion.forward_xrd(features, idx=0, xrd_fusion_method="xrd_attn_weight")
    assert result.dtype == features.dtype


# ── XRDFusionMethod.forward_xrd xrd_embed_scale dispatch ──────────────────────


def test_forward_xrd_embed_scale_returns_x(dino_flavour, features, xrd_map_4d):
    fusion = _make_fusion(xrd_map_4d)
    result = fusion.forward_xrd(features, idx=0, xrd_fusion_method="xrd_embed_scale")
    assert result is features


def test_forward_xrd_embed_scale_shape(dino_flavour, features, xrd_map_4d):
    fusion = _make_fusion(xrd_map_4d)
    result = fusion.forward_xrd(features, idx=0, xrd_fusion_method="xrd_embed_scale")
    assert result.shape == features.shape


def test_forward_xrd_embed_scale_dtype(dino_flavour, features, xrd_map_4d):
    fusion = _make_fusion(xrd_map_4d)
    result = fusion.forward_xrd(features, idx=0, xrd_fusion_method="xrd_embed_scale")
    assert result.dtype == features.dtype
