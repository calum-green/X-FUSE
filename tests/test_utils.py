import pytest
import torch
import numpy as np
import porespy as ps
from x_fuse.utils import xrd_to_tensor, get_multiphase, downsample_xrdct

# ── xrd_to_tensor ──────────────────────────────────────────────────────────────


@pytest.fixture
def xrd_np():
    rng = np.random.default_rng(0)
    return rng.integers(0, 256, (64, 64), dtype=np.uint8)


def test_xrd_to_tensor_is_tensor(xrd_np):
    result = xrd_to_tensor(xrd_np, device="cpu")
    assert isinstance(result, torch.Tensor)


def test_xrd_to_tensor_shape(xrd_np):
    result = xrd_to_tensor(xrd_np, device="cpu")
    assert result.shape == (1, 1, 64, 64)


def test_xrd_to_tensor_dtype(xrd_np):
    result = xrd_to_tensor(xrd_np, device="cpu")
    assert result.dtype == torch.float32


def test_xrd_to_tensor_range(xrd_np):
    result = xrd_to_tensor(xrd_np, device="cpu")
    assert result.min().item() >= 0.0
    assert result.max().item() <= 1.0


def test_xrd_to_tensor_device(xrd_np):
    result = xrd_to_tensor(xrd_np, device="cpu")
    assert result.device == torch.device("cpu")


# ── get_multiphase ─────────────────────────────────────────────────────────────


@pytest.fixture
def multiphase_args():
    im1_kwargs = {"shape": [64, 64], "blobiness": 1.0, "porosity": 0.5}
    im2_kwargs = {"r": 5, "clearance": 1}
    return ps.generators.blobs, im1_kwargs, ps.generators.random_spheres, im2_kwargs


def test_get_multiphase_return_types(multiphase_args):
    im1, im1_kw, im2, im2_kw = multiphase_args
    gray, phase_A, phase_B = get_multiphase(
        im1, im1_kw, im2, im2_kw, extract_spheres=True, dataset_size=2
    )
    assert all(isinstance(x, np.ndarray) for x in gray)
    assert all(isinstance(x, np.ndarray) for x in phase_A)
    assert all(isinstance(x, np.ndarray) for x in phase_B)


def test_get_multiphase_dataset_size(multiphase_args):
    im1, im1_kw, im2, im2_kw = multiphase_args
    gray, phase_A, phase_B = get_multiphase(
        im1, im1_kw, im2, im2_kw, extract_spheres=True, dataset_size=3
    )
    assert len(gray) == 3
    assert len(phase_A) == 3
    assert len(phase_B) == 3


def test_get_multiphase_mask_binary(multiphase_args):
    im1, im1_kw, im2, im2_kw = multiphase_args
    _, phase_A, phase_B = get_multiphase(
        im1, im1_kw, im2, im2_kw, extract_spheres=True, dataset_size=2
    )
    for mask in phase_A + phase_B:
        assert set(np.unique(mask)).issubset({0.0, 1.0})


def test_get_multiphase_gray_range(multiphase_args):
    im1, im1_kw, im2, im2_kw = multiphase_args
    gray, _, _ = get_multiphase(
        im1, im1_kw, im2, im2_kw, extract_spheres=True, dataset_size=2
    )
    for img in gray:
        assert img.min() >= 0.0
        assert (
            img.max() <= 1.0 + 1e-6
        )  # float32/float64 dtype mismatch in normalisation


def test_get_multiphase_shape_consistency(multiphase_args):
    im1, im1_kw, im2, im2_kw = multiphase_args
    gray, phase_A, phase_B = get_multiphase(
        im1, im1_kw, im2, im2_kw, extract_spheres=True, dataset_size=3
    )
    shapes = [x.shape for x in gray + phase_A + phase_B]
    assert len(set(shapes)) == 1


# ── downsample_xrdct ───────────────────────────────────────────────────────────


@pytest.fixture
def downsample_inputs():
    rng = np.random.default_rng(1)
    gray = [rng.random((64, 64)).astype(np.float32) for _ in range(3)]
    phase_A = [rng.random((64, 64)).astype(np.float32) for _ in range(3)]
    phase_B = [rng.random((64, 64)).astype(np.float32) for _ in range(3)]
    return gray, phase_A, phase_B


def test_downsample_output_shape(downsample_inputs):
    gray, phase_A, phase_B = downsample_inputs
    _, ds_A, ds_B = downsample_xrdct(gray, phase_A, phase_B, factor=4)
    assert ds_A[0].shape == (16, 16)
    assert ds_B[0].shape == (16, 16)


def test_downsample_output_dtype(downsample_inputs):
    gray, phase_A, phase_B = downsample_inputs
    _, ds_A, ds_B = downsample_xrdct(gray, phase_A, phase_B, factor=4)
    assert ds_A[0].dtype == np.float32
    assert ds_B[0].dtype == np.float32


def test_downsample_list_lengths(downsample_inputs):
    gray, phase_A, phase_B = downsample_inputs
    _, ds_A, ds_B = downsample_xrdct(gray, phase_A, phase_B, factor=2)
    assert len(ds_A) == 3
    assert len(ds_B) == 3
