import numpy as np
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
    assert cfg.img_size == 224
    assert cfg.stride == 4
    assert cfg.vis_data is True
    assert cfg.vis_dino_features is False
    assert cfg.device is None
    assert cfg.phases == ["Na", "Zn"]


def test_config_roundtrip(tmp_path):
    cfg = XFuseConfig(
        name="roundtrip",
        dataset_type="porespy",
        phases=["alpha", "beta"],
        img_size=128,
        invert=True,
    )
    yaml_path = str(tmp_path / "cfg.yaml")
    cfg.to_yaml(yaml_path)
    loaded = XFuseConfig.from_yaml(yaml_path)
    assert loaded.name == "roundtrip"
    assert loaded.dataset_type == "porespy"
    assert loaded.phases == ["alpha", "beta"]
    assert loaded.img_size == 128
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
        img_size=56,
        vis_data=False,
    )


@patch("x_fuse.pipeline.load_diad_xrdct")
@patch("x_fuse.pipeline.load_diad_xct_zn13x")
def test_run_data_saves_expected_files(mock_xct, mock_xrd, tmp_path):
    mock_xct.return_value = np.random.rand(56, 56, 5).astype(np.float32)
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
    mock_xct.return_value = np.random.rand(56, 56, 3).astype(np.float32)
    mock_xrd.return_value = {
        "alpha": np.random.rand(3, 8, 8).astype(np.float32),
    }
    cfg = XFuseConfig(
        name="custom",
        output_dir=str(tmp_path),
        dataset_type="diad",
        phases=["alpha"],
        img_size=56,
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
    xct_arr = np.ones((56, 56, 3), dtype=np.float32) * 0.3
    mock_xct.return_value = xct_arr
    mock_xrd.return_value = {"Na": np.ones((3, 8, 8), dtype=np.float32) * 0.4}
    cfg = XFuseConfig(
        name="inv",
        output_dir=str(tmp_path),
        dataset_type="diad",
        phases=["Na"],
        img_size=56,
        invert=True,
        vis_data=False,
    )
    run_data(cfg)
    xct_saved = np.load(tmp_path / "inv" / "data" / "xct.npy")
    # After inversion, values near 0.3 should become near 0.7
    assert xct_saved.mean() > 0.5
