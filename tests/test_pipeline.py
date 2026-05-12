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
