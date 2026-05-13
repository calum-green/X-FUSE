import numpy as np
import torch
from x_fuse.utils import invert_image, xct_contrast, overlay_mask


def test_invert_image_float():
    img = np.array([[0.0, 0.5], [1.0, 0.25]], dtype=np.float32)
    result = invert_image(img)
    expected = np.array([[1.0, 0.5], [0.0, 0.75]], dtype=np.float32)
    np.testing.assert_allclose(result, expected)


def test_invert_image_preserves_shape():
    img = np.random.rand(64, 64).astype(np.float32)
    assert invert_image(img).shape == img.shape


def test_xct_contrast_empty_mask_returns_zero():
    pca = np.zeros((4, 4), dtype=np.float32)
    xct = np.random.rand(4, 4).astype(np.float32)
    assert xct_contrast(1.0, pca, xct) == 0.0


def test_xct_contrast_full_mask_returns_zero():
    pca = np.ones((4, 4), dtype=np.float32)
    xct = np.random.rand(4, 4).astype(np.float32)
    assert xct_contrast(0.0, pca, xct) == 0.0


def test_xct_contrast_returns_positive():
    pca = np.array([[0.0, 0.0, 1.0, 1.0]] * 4, dtype=np.float32)
    xct = np.array([[0.1, 0.1, 0.9, 0.9]] * 4, dtype=np.float32)
    score = xct_contrast(0.5, pca, xct)
    assert score > 0.0


def test_overlay_mask_shape():
    xct = np.random.rand(32, 32).astype(np.float32)
    mask = np.zeros((32, 32), dtype=bool)
    mask[10:20, 10:20] = True
    result = overlay_mask(xct, mask)
    assert result.shape == (32, 32, 3)


def test_overlay_mask_clipped():
    xct = np.ones((8, 8), dtype=np.float32)
    mask = np.ones((8, 8), dtype=bool)
    result = overlay_mask(xct, mask)
    assert result.max() <= 1.0
    assert result.min() >= 0.0


# ---------------------------------------------------------------------------
# Config tests
# ---------------------------------------------------------------------------

from x_fuse.config import XFuseConfig  # noqa: E402


def minimal_config(name="test_run") -> XFuseConfig:
    return XFuseConfig(name=name)


def test_config_defaults():
    cfg = minimal_config()
    assert cfg.dataset_type == "diad"
    assert cfg.fusion_method == "gating"
    assert cfg.stride == 4
    assert cfg.vis_data is True
    assert cfg.vis_dino_features is False
    assert cfg.device is None
    assert cfg.phases == ["Na", "Zn"]
    assert cfg.xct_sample_idx == -1
    assert cfg.xrdct_sample_idx == -1


def test_config_roundtrip(tmp_path):
    cfg = XFuseConfig(
        name="roundtrip",
        dataset_type="porespy",
        phases=["alpha", "beta"],
        xct_sample_idx=5,
        xrdct_sample_idx=10,
        invert=True,
    )
    yaml_path = str(tmp_path / "cfg.yaml")
    cfg.to_yaml(yaml_path)
    loaded = XFuseConfig.from_yaml(yaml_path)
    assert loaded.name == "roundtrip"
    assert loaded.dataset_type == "porespy"
    assert loaded.phases == ["alpha", "beta"]
    assert loaded.xct_sample_idx == 5
    assert loaded.xrdct_sample_idx == 10
    assert loaded.invert is True


def test_config_replace_does_not_mutate():
    cfg = minimal_config("original")
    cfg2 = cfg.replace(name="copy", fusion_method="attention")
    assert cfg.name == "original"
    assert cfg.fusion_method == "gating"
    assert cfg2.name == "copy"
    assert cfg2.fusion_method == "attention"


def test_config_output_path():
    cfg = XFuseConfig(name="myrun", output_dir="results/")
    assert str(cfg.output_path) == "results/myrun"


def test_config_resolve_device_override():
    cfg = XFuseConfig(name="x", device="cpu")
    assert cfg.resolve_device() == "cpu"


def test_config_resolve_device_auto():
    import torch

    cfg = XFuseConfig(name="x")
    device = cfg.resolve_device()
    expected = "cuda" if torch.cuda.is_available() else "cpu"
    assert device == expected


# ---------------------------------------------------------------------------
# _snap_to_multiple_of_16 / _extract_sample tests
# ---------------------------------------------------------------------------

from x_fuse.pipeline import _snap_to_multiple_of_16, _extract_sample  # noqa: E402


def test_snap_to_multiple_of_16():
    assert _snap_to_multiple_of_16(1850) == 1840
    assert _snap_to_multiple_of_16(224) == 224
    assert _snap_to_multiple_of_16(300) == 288
    assert _snap_to_multiple_of_16(16) == 16
    assert _snap_to_multiple_of_16(15) == 0


def test_extract_sample_uses_separate_indices():
    xct_raw = np.zeros((5, 32, 32), dtype=np.float32)
    xct_raw[2] = 1.0
    xrd_raw = {
        "Na": np.zeros((3, 8, 8), dtype=np.float32),
        "Zn": np.zeros((3, 8, 8), dtype=np.float32),
    }
    xrd_raw["Na"][1] = 2.0
    xrd_raw["Zn"][1] = 3.0

    cfg = XFuseConfig(name="t", xct_sample_idx=2, xrdct_sample_idx=1)
    xct_s, xrd_s = _extract_sample(xct_raw, xrd_raw, cfg)

    assert xct_s.shape == (32, 32)
    assert xct_s.mean() == pytest.approx(1.0)
    assert xrd_s["Na"].shape == (8, 8)
    assert xrd_s["Na"].mean() == pytest.approx(2.0)
    assert xrd_s["Zn"].mean() == pytest.approx(3.0)


# ---------------------------------------------------------------------------
# run_data tests
# ---------------------------------------------------------------------------

import pytest  # noqa: E402
from unittest.mock import patch  # noqa: E402
from x_fuse.pipeline import run_data  # noqa: E402


def _make_diad_config(tmp_path, name="test") -> XFuseConfig:
    # Create stub paths so _validate_paths passes
    (tmp_path / "xct.h5").touch()
    (tmp_path / "phases").mkdir(exist_ok=True)
    return XFuseConfig(
        name=name,
        output_dir=str(tmp_path),
        dataset_type="diad",
        xct_path=str(tmp_path / "xct.h5"),
        phase_folder=str(tmp_path / "phases"),
        phases=["Na", "Zn"],
        vis_data=False,
    )


@patch("x_fuse.pipeline.load_diad_xrdct")
@patch("x_fuse.pipeline.load_diad_xct_zn13x")
def test_run_data_saves_expected_files(mock_xct, mock_xrd, tmp_path):
    mock_xct.return_value = np.random.rand(3, 32, 32).astype(np.float32)
    mock_xrd.return_value = {
        "Na": np.random.rand(5, 8, 8).astype(np.float32),
        "Zn": np.random.rand(5, 8, 8).astype(np.float32),
    }
    cfg = _make_diad_config(tmp_path)
    run_data(cfg)
    data_dir = tmp_path / "test" / "data"
    assert (data_dir / "xct.npy").exists()
    assert (data_dir / "Na_xrd.npy").exists()
    assert (data_dir / "Zn_xrd.npy").exists()
    assert (data_dir / "data_summary.txt").exists()


@patch("x_fuse.pipeline.load_diad_xrdct")
@patch("x_fuse.pipeline.load_diad_xct_zn13x")
def test_run_data_custom_phases(mock_xct, mock_xrd, tmp_path):
    mock_xct.return_value = np.random.rand(3, 32, 32).astype(np.float32)
    mock_xrd.return_value = {
        "alpha": np.random.rand(3, 8, 8).astype(np.float32),
    }
    cfg = XFuseConfig(
        name="custom",
        output_dir=str(tmp_path),
        dataset_type="diad",
        phases=["alpha"],
        vis_data=False,
    )
    run_data(cfg)
    data_dir = tmp_path / "custom" / "data"
    assert (data_dir / "alpha_xrd.npy").exists()
    assert not (data_dir / "Na_xrd.npy").exists()


def test_run_data_missing_xct_path_raises(tmp_path):
    cfg = XFuseConfig(
        name="bad",
        output_dir=str(tmp_path),
        dataset_type="diad",
        xct_path="/nonexistent/path.h5",
        phase_folder=str(tmp_path),
        phases=["Na"],
        vis_data=False,
    )
    with pytest.raises(FileNotFoundError, match="xct_path"):
        run_data(cfg)


@patch("x_fuse.pipeline.load_diad_xrdct")
@patch("x_fuse.pipeline.load_diad_xct_zn13x")
def test_run_data_invert_flag(mock_xct, mock_xrd, tmp_path):
    xct_arr = np.ones((3, 32, 32), dtype=np.float32) * 0.3
    mock_xct.return_value = xct_arr
    mock_xrd.return_value = {"Na": np.ones((3, 8, 8), dtype=np.float32) * 0.4}
    cfg = XFuseConfig(
        name="inv",
        output_dir=str(tmp_path),
        dataset_type="diad",
        phases=["Na"],
        invert=True,
        vis_data=False,
    )
    run_data(cfg)
    xct_saved = np.load(tmp_path / "inv" / "data" / "xct.npy")
    # After inversion, values near 0.3 should become near 0.7
    assert xct_saved.mean() > 0.5


# ---------------------------------------------------------------------------
# _load_raw_data print tests
# ---------------------------------------------------------------------------


@patch("x_fuse.pipeline.load_diad_xrdct")
@patch("x_fuse.pipeline.load_diad_xct_zn13x")
def test_load_raw_data_prints_loading_steps(mock_xct, mock_xrd, tmp_path, capsys):
    mock_xct.return_value = np.random.rand(3, 32, 32).astype(np.float32)
    mock_xrd.return_value = {
        "Na": np.random.rand(5, 8, 8).astype(np.float32),
        "Zn": np.random.rand(5, 8, 8).astype(np.float32),
    }
    cfg = _make_diad_config(tmp_path)
    run_data(cfg)
    out = capsys.readouterr().out
    assert "loading XCT" in out
    assert "loading XRDCT" in out


# ---------------------------------------------------------------------------
# run_features / run_segment / run_refine tests
# ---------------------------------------------------------------------------

from unittest.mock import MagicMock  # noqa: E402
from x_fuse.pipeline import run_features, run_segment, run_refine  # noqa: E402


def _write_fake_data(tmp_path, name, phases, img_size=56):
    data_dir = tmp_path / name / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    xct = np.random.rand(img_size, img_size).astype(np.float32)
    np.save(data_dir / "xct.npy", xct)
    for phase in phases:
        xrd = np.random.rand(8, 8).astype(np.float32)
        np.save(data_dir / f"{phase}_xrd.npy", xrd)
    return xct


def _write_fake_features(tmp_path, name, phases, img_size=56, n_components=3):
    feat_dir = tmp_path / name / "features"
    feat_dir.mkdir(parents=True, exist_ok=True)
    for phase in phases:
        pca = np.random.rand(img_size * img_size, n_components).astype(np.float32)
        np.save(feat_dir / f"{phase}_pca.npy", pca)


def _write_fake_masks(tmp_path, name, phases, img_size=56):
    seg_dir = tmp_path / name / "segment"
    seg_dir.mkdir(parents=True, exist_ok=True)
    (seg_dir / "threshold.txt").write_text("0.5")
    for phase in phases:
        mask = np.random.randint(0, 2, (img_size, img_size), dtype=np.uint8)
        np.save(seg_dir / f"{phase}_mask.npy", mask)


@patch("x_fuse.fusion.XFuse")
def test_run_features_saves_expected_files(mock_xfuse_cls, tmp_path):
    phases = ["Na", "Zn"]
    img_size = 56
    n_comp = 3
    _write_fake_data(tmp_path, "run", phases, img_size)

    fake_feats = torch.zeros(1, 384, img_size, img_size)
    mock_net = MagicMock()
    mock_net.forward_sequential.return_value = fake_feats
    mock_xfuse_cls.return_value = mock_net

    cfg = XFuseConfig(
        name="run",
        output_dir=str(tmp_path),
        phases=phases,
        n_components=n_comp,
        vis_fused_maps=False,
        vis_pca_components=False,
        device="cpu",
    )

    with (
        patch(
            "x_fuse.pipeline.tr.to_numpy",
            return_value=np.zeros((384, img_size, img_size)),
        ),
        patch(
            "x_fuse.pipeline.tr.flatten",
            return_value=np.zeros((img_size * img_size, 384)),
        ),
        patch(
            "x_fuse.pipeline.do_single_pca",
            return_value=np.zeros((img_size * img_size, n_comp)),
        ),
        patch(
            "x_fuse.pipeline.rescale_pca",
            return_value=np.zeros((img_size * img_size, n_comp)),
        ),
        patch(
            "x_fuse.pipeline.load_img",
            return_value=(torch.zeros(3, img_size, img_size), None),
        ),
    ):
        run_features(cfg)

    feat_dir = tmp_path / "run" / "features"
    for phase in phases:
        assert (feat_dir / f"{phase}_feats.npy").exists()
        assert (feat_dir / f"{phase}_pca.npy").exists()


def test_run_features_missing_data_raises(tmp_path):
    cfg = XFuseConfig(
        name="missing",
        output_dir=str(tmp_path),
        phases=["Na"],
        device="cpu",
    )
    with pytest.raises(FileNotFoundError, match="data"):
        run_features(cfg)


def test_run_segment_saves_masks_and_threshold(tmp_path):
    phases = ["Na", "Zn"]
    img_size = 56
    _write_fake_data(tmp_path, "seg", phases, img_size)
    _write_fake_features(tmp_path, "seg", phases, img_size)

    cfg = XFuseConfig(
        name="seg",
        output_dir=str(tmp_path),
        phases=phases,
        vis_masks=False,
    )
    run_segment(cfg, threshold=0.3)

    seg_dir = tmp_path / "seg" / "segment"
    assert (seg_dir / "threshold.txt").read_text() == "0.3"
    for phase in phases:
        assert (seg_dir / f"{phase}_mask.npy").exists()
        mask = np.load(seg_dir / f"{phase}_mask.npy")
        assert mask.dtype == np.uint8
        assert set(np.unique(mask)).issubset({0, 1})


def test_run_segment_missing_features_raises(tmp_path):
    cfg = XFuseConfig(
        name="miss",
        output_dir=str(tmp_path),
        phases=["Na"],
    )
    with pytest.raises(FileNotFoundError, match="features"):
        run_segment(cfg, threshold=0.5)


@patch("sam2.build_sam.build_sam2")
@patch("sam2.sam2_image_predictor.SAM2ImagePredictor")
def test_run_refine_saves_refined_masks_and_config(
    mock_predictor_cls, mock_build, tmp_path
):
    phases = ["Na"]
    img_size = 56
    _write_fake_data(tmp_path, "ref", phases, img_size)
    _write_fake_masks(tmp_path, "ref", phases, img_size)

    fake_mask = np.ones((1, img_size, img_size), dtype=bool)
    fake_scores = np.array([0.95])
    mock_predictor = MagicMock()
    mock_predictor.predict.return_value = (fake_mask, fake_scores, None)
    mock_predictor_cls.return_value = mock_predictor
    mock_build.return_value = MagicMock()

    cfg = XFuseConfig(
        name="ref",
        output_dir=str(tmp_path),
        phases=phases,
        vis_sam2=False,
        device="cpu",
    )
    run_refine(cfg)

    refine_dir = tmp_path / "ref" / "refine"
    assert (refine_dir / "Na_refined_mask.npy").exists()
    assert (refine_dir / "config.yaml").exists()


def test_run_refine_missing_segment_raises(tmp_path):
    cfg = XFuseConfig(
        name="miss",
        output_dir=str(tmp_path),
        phases=["Na"],
    )
    with pytest.raises(FileNotFoundError, match="segment"):
        run_refine(cfg)


@patch("x_fuse.pipeline.load_diad_xrdct")
@patch("x_fuse.pipeline.load_diad_xct_zn13x")
def test_run_data_vis_saves_overview_image(mock_xct, mock_xrd, tmp_path):
    mock_xct.return_value = np.random.rand(3, 32, 32).astype(np.float32)
    mock_xrd.return_value = {"Na": np.random.rand(3, 8, 8).astype(np.float32)}
    (tmp_path / "xct.h5").touch()
    (tmp_path / "phases").mkdir()
    cfg = XFuseConfig(
        name="vistest",
        output_dir=str(tmp_path),
        dataset_type="diad",
        xct_path=str(tmp_path / "xct.h5"),
        phase_folder=str(tmp_path / "phases"),
        phases=["Na"],
        vis_data=True,
    )
    run_data(cfg)
    assert (tmp_path / "vistest" / "data" / "data_overview.png").exists()
