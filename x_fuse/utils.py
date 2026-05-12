import porespy as ps
import numpy as np
import matplotlib.pyplot as plt
import cv2
import torch
from PIL import Image, ImageOps
from typing import Any
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
from types import MethodType
from torch.nn.modules.utils import _pair
from .alibi import (
    PretrainedViTWrapper,
    MODEL_LIST,
    AlibiVitWrapper,
    DistanceMatrixWrapper,
    get_alibi_slope,
)
import h5py as h5


def get_multiphase(
    imA, imA_kwargs, imB, imB_kwargs, extract_spheres=False, vis=False, dataset_size=1
):
    """
    Input: imA, imB :-> images for phase A and phase B (e.g. blobs with different
    porosities)
    Output: (imgA, phaseA, phaseB):-> list(imgA.astype(float32)),
        list(phaseA.astype(float32)), list(phaseB.astype(float32))
        Combined Phase Image and Phase A and Phase B masks
    """
    SEED = 1337
    gray_imgs = []
    phase_A_masks = []
    phase_B_masks = []

    for i in range(dataset_size):
        imA_kwargs["seed"] = SEED + i
        imB_kwargs["seed"] = SEED + i

        im_A = imA(**imA_kwargs)
        im_B = imB(**imB_kwargs) if not extract_spheres else imB(im=im_A, **imB_kwargs)

        # Build multi-phase image:
        # 0 = background, 1 = phase A only, 2 = phase B only, 3 = overlap
        multi_phase = np.zeros_like(im_A, dtype=np.uint8)
        multi_phase[im_A & ~im_B] = 1  # Phase A only
        multi_phase[~im_A & im_B] = 2  # Phase B only
        multi_phase[im_A & im_B] = (
            2 if not extract_spheres else 3
        )  # Overlap is Phase B too

        gray_img = np.zeros_like(multi_phase, dtype=float)
        gray_img[multi_phase == 1] = np.random.uniform(
            0.8, 1.2, size=(multi_phase == 1).sum()
        )  # Phase A: mid-gray
        gray_img[multi_phase == 2] = np.random.uniform(
            1.6, 2, size=(multi_phase == 2).sum()
        )  # Phase B: lighter gray
        if extract_spheres:  # Overlap: even lighter gray
            gray_img[multi_phase == 3] = np.random.uniform(
                2.5, 3.0, size=(multi_phase == 3).sum()
            )

        # Get phase A and phase B as float16 masks
        phase_A = im_A.astype(np.float32)
        phase_B = im_B.astype(np.float32)

        if extract_spheres is True:
            # Extract spheres from the images
            phase_B_mask = np.zeros_like(phase_B)
            phase_B_mask[phase_A == phase_B] = (
                0  # Keep only the part of phase B that does not overlap with phase A
            )
            phase_B_mask[phase_A != phase_B] = (
                1  # Keep only the part of phase B that does not overlap with phase A
            )
            phase_B = phase_B_mask.astype(np.float32)

        # Normalize to [0, 1] (optional, but values are already in this range)
        gray_img = (gray_img - gray_img.min()) / (
            gray_img.max() - gray_img.min()
        ).astype(np.float32)

        gray_imgs.append(gray_img)
        phase_A_masks.append(phase_A)
        phase_B_masks.append(phase_B)

        if vis is True:
            # Visualize
            fig, ax = plt.subplots(1, 4, figsize=(24, 12))
            ax[0].imshow(multi_phase, cmap="gray")
            ax[0].set_title("Multi-phase image (0=bg, 1=A, 2=B, 3=overlap)")
            ax[0].axis("off")
            ax[1].imshow(gray_img, cmap="gray")
            ax[1].set_title("Grayscale multi-phase image")
            ax[1].axis("off")
            ax[2].imshow(phase_A, cmap="gray")
            ax[2].set_title("Phase A")
            ax[2].axis("off")
            ax[3].imshow(phase_B, cmap="gray")
            ax[3].set_title("Phase B")
            ax[3].axis("off")
            plt.colorbar(ax[1].imshow(gray_img, cmap="gray"), ax=ax[1])
            plt.tight_layout()
            plt.show()

        elif vis is False:
            pass

    return gray_imgs, phase_A_masks, phase_B_masks


def downsample_xrdct(gray_imgs, phase_A_masks, phase_B_masks, factor=2, vis=False):
    """
    Downsample the phase_A and phase_B masks by factor
    """
    gray_imgs = gray_imgs
    downsampled_phase_A_masks = []
    downsampled_phase_B_masks = []

    phase_A_shape = phase_A_masks[0].shape
    phase_B_shape = phase_B_masks[0].shape

    for i in range(len(gray_imgs)):
        phase_A = phase_A_masks[i]
        phase_B = phase_B_masks[i]

        # Downsample each image by the specified factor
        downsampled_phase_A = cv2.resize(
            phase_A,
            (phase_A_shape[1] // factor, phase_A_shape[0] // factor),
            interpolation=cv2.INTER_LANCZOS4,
        )
        downsampled_phase_B = cv2.resize(
            phase_B,
            (phase_B_shape[1] // factor, phase_B_shape[0] // factor),
            interpolation=cv2.INTER_LANCZOS4,
        )

        downsampled_phase_A_masks.append(downsampled_phase_A.astype(np.float32))
        downsampled_phase_B_masks.append(downsampled_phase_B.astype(np.float32))

    if vis is True:
        # Visualize the downsampled masks
        fig, ax = plt.subplots(len(gray_imgs), 3, figsize=(24, 12))
        for i in range(len(gray_imgs)):
            ax[i, 0].imshow(gray_imgs[i], cmap="gray")
            ax[i, 0].set_title(f"Grayscale Multi-phase Image {i+1}")
            ax[i, 0].axis("off")
            ax[i, 1].imshow(downsampled_phase_A_masks[i], cmap="gray")
            ax[i, 1].set_title(f"Downsampled Phase A Mask {i+1}")
            ax[i, 1].axis("off")
            ax[i, 2].imshow(downsampled_phase_B_masks[i], cmap="gray")
            ax[i, 2].set_title(f"Downsampled Phase B Mask {i+1}")
            ax[i, 2].axis("off")
        plt.tight_layout()
        plt.show()

    return gray_imgs, downsampled_phase_A_masks, downsampled_phase_B_masks


def get_ps_images(IMG_SIZE=224):
    ps_dict = {
        "blobs": ps.generators.blobs,
        "rand_spheres": ps.generators.random_spheres,
    }

    im1 = ps_dict["blobs"]
    im2 = ps_dict["rand_spheres"]

    im1_kwargs = {"shape": [IMG_SIZE, IMG_SIZE], "blobiness": 0.5, "porosity": 0.1}
    im2_kwargs = {"r": 40, "clearance": 10}

    gray, phase_A_masks, phase_B_masks = get_multiphase(
        im1,
        im1_kwargs,
        im2,
        im2_kwargs,
        extract_spheres=True,
        vis=False,
        dataset_size=10,
    )
    gray_imgs, low_A, low_B = downsample_xrdct(
        gray, phase_A_masks, phase_B_masks, factor=10, vis=False
    )

    return gray_imgs, low_A, low_B


def load_xrd_h5(xrd_path):
    """
    Load the XRD data from the path, normalise, convert to tensor
    """
    xrd_img = h5.File(xrd_path, "r")["xrd"][:]
    return xrd_img


def xrd_to_tensor(xrd_img, device):
    """
    Convert the XRD image to a tensor on DEVICE, min-max normalised to [0, 1].
    """
    t = transforms.ToTensor()(Image.fromarray(xrd_img).convert("L"))  # (1, H, W)
    mn, mx = t.min(), t.max()
    t = (t - mn) / (mx - mn)

    return t.unsqueeze(0).to(device)  # (1, 1, H, W)


def get_alibi_model(
    model_type: str,
    model_path: str,
    device: str,
    stride: int = 14,
) -> PretrainedViTWrapper:
    """
    Load the ALiBi model from the specified directory and return a
    PretrainedViTWrapper instance.
    """
    weights = torch.load(model_path, weights_only=True, map_location=device)
    slope_type = "learned" if "_l" in model_type else "constant"
    add_cls = False if "nr" in model_type else True
    n_reg_tokens = 0 if "nr" in model_type else 4
    arch_to_model = {
        "vits": MODEL_LIST[1] if n_reg_tokens > 0 else MODEL_LIST[0],
        "vitb": MODEL_LIST[3],
        "vitl": MODEL_LIST[15],
        "vitg": MODEL_LIST[16],
    }
    base_model = next(
        (v for k, v in arch_to_model.items() if k in model_type), MODEL_LIST[1]
    )

    model = AlibiVitWrapper(
        base_model,
        stride=stride,
        device=device,
        slope_type=slope_type,
        normalize=True,
        wrap=True,
        add_cls=add_cls,
        n_reg_tokens=n_reg_tokens,
    )
    model.load_state_dict(weights)
    return model


def _inject_alibi_dv3(
    dv3_model: nn.Module,
    distance_matrix: DistanceMatrixWrapper,
    slope_type: str,
    device: str,
) -> None:
    """Patch DINOv3 attention compute_attention in-place to add ALiBi distance bias.

    Mirrors dinosaw's inject_alibi_into_dv3 without requiring that package.
    Constant/fixed slopes are non-persistent (not in checkpoints).
    Learned slopes are registered as Parameters so load_state_dict populates them.
    """
    num_heads = dv3_model.blocks[0].attn.num_heads
    m_init = get_alibi_slope(num_heads, slope_type=slope_type, device=device)
    is_learned = isinstance(m_init, nn.Parameter)

    def _compute_attn(self, qkv, attn_bias=None, rope=None):
        B, N, _ = qkv.shape
        C = self.qkv.in_features
        qkv_r = qkv.reshape(B, N, 3, self.num_heads, C // self.num_heads)
        q, k, v = qkv_r.unbind(2)
        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)
        bias = (
            self.m.to(device=q.device, dtype=q.dtype)
            * distance_matrix.matrix.to(device=q.device, dtype=q.dtype)
        ).unsqueeze(0)
        x = F.scaled_dot_product_attention(q, k, v, attn_mask=bias)
        return x.transpose(1, 2).reshape(B, N, C)

    for block in dv3_model.blocks:
        if is_learned:
            block.attn.register_parameter("m", nn.Parameter(m_init.data.clone()))
        else:
            block.attn.register_buffer("m", m_init.clone(), persistent=False)
        block.attn.compute_attention = MethodType(_compute_attn, block.attn)


def get_dv3_model(
    model_type: str,
    model_path: str,
    lib_path: str,
    device: str,
    stride: int = 16,
) -> nn.Module:
    """Load a DINOv3 checkpoint (NoPE or ALiBi) saved from the dinosaw AlibiVitWrapper.

    Checkpoints from that wrapper have 'model.*' keys because the DINOv3 hub model
    is stored as self.model. We recreate the same structure, optionally inject ALiBi,
    then load. If 'nope' is in model_type, ALiBi injection is skipped entirely.
    """
    weights = torch.load(model_path, weights_only=True, map_location=device)
    n_reg_tokens = 0 if "nr" in model_type else 4
    add_cls = "nr" not in model_type
    use_alibi = "alibi" in model_type

    model_dict: dict[str, str] = {
        "vits16": "dinov3_vits16",
        "vits16plus": "dinov3_vits16plus",
        "vitl16": "dinov3_vitl16",
        "vith16": "dinov3_vith16",
        "vith16plus": "dinov3_vith16plus",
        "vit7b16": "dinov3_vit7b16",
    }

    hub_fn = model_dict[
        model_type.split("_")[2]
    ]  # format {alibi/nope}_dinov3_{arch}[_extra]
    dv3 = torch.hub.load(lib_path, hub_fn, source="local", pretrained=False)

    if stride != dv3.patch_embed.proj.stride[0]:
        dv3.patch_embed.proj.stride = _pair(stride)

    distance_matrix = None
    if use_alibi:
        slope_type = "learned" if "_l" in model_type else "constant"
        distance_matrix = DistanceMatrixWrapper(
            n_tokens_h=16,
            n_tokens_w=16,
            n_reg_tokens=n_reg_tokens,
            normalize=True,
            wrap=True,
            add_cls=add_cls,
        )
        _inject_alibi_dv3(dv3, distance_matrix, slope_type, device)

    class _DV3Wrapper(nn.Module):
        def __init__(self, model, dist_mat, stride_val, n_reg):
            super().__init__()
            self.model = model
            self.distance_matrix = dist_mat
            self.stride = stride_val
            self.patch_size = 16
            self.n_reg_tokens = n_reg
            self.n_cls_tokens = 1

        def forward_features(self, x, make_2D=False, **kwargs):
            b, _, h, w = x.shape
            p, s = self.patch_size, self.stride
            n_h = (h - p) // s + 1
            n_w = (w - p) // s + 1
            if self.distance_matrix is not None:
                self.distance_matrix.update(n_h, n_w)
            feats = self.model.forward_features(x)["x_norm_patchtokens"]
            feats = feats.permute(0, 2, 1)  # (B, C, N)
            if make_2D:
                feats = feats.reshape(b, -1, n_h, n_w)
            return feats

        def forward_feats_attn(self, x, masks=None, attn_choice="none"):
            feats = self.forward_features(x)  # (B, C, N)
            feats = feats.permute(0, 2, 1)  # (B, N, C)
            return {"x_norm_patchtokens": feats, "masks": masks}

    wrapper = _DV3Wrapper(dv3, distance_matrix, stride, n_reg=n_reg_tokens)
    result = wrapper.load_state_dict(weights, strict=False)
    unexpected = result.unexpected_keys
    missing = [k for k in result.missing_keys if "rope" not in k]
    if unexpected or missing:
        raise RuntimeError(
            f"Checkpoint mismatch in _DV3Wrapper.\n"
            f"  Unexpected keys: {unexpected}\n"
            f"  Missing keys (non-rope): {missing}"
        )
    wrapper.to(device)
    return wrapper


def invert_image(image: np.ndarray) -> np.ndarray:
    return np.ones_like(image) - image


def load_img(img: np.ndarray, transform) -> tuple:
    unnormalize = transforms.Normalize(
        mean=(-0.485 / 0.229, -0.456 / 0.224, -0.406 / 0.225),
        std=(1 / 0.229, 1 / 0.224, 1 / 0.225),
    )
    img_uint8 = (img * 255).astype(np.uint8)
    image = ImageOps.autocontrast(Image.fromarray(img_uint8).convert("RGB"))
    tensor = transform(image)
    trans_img = transforms.ToPILImage()(unnormalize(tensor))
    return tensor, trans_img


def overlay_mask(
    xct: np.ndarray,
    mask: np.ndarray,
    color: tuple = (1, 0.2, 0.2),
    alpha: float = 0.4,
) -> np.ndarray:
    mask = mask.astype(bool)
    rgb = np.stack([xct] * 3, axis=-1)
    overlay = rgb.copy()
    overlay[mask] = (1 - alpha) * rgb[mask] + alpha * np.array(color)
    return np.clip(overlay, 0.0, 1.0)


def xct_contrast(
    threshold: float, pca_component: np.ndarray, xct_img: np.ndarray
) -> float:
    mask = pca_component > threshold
    if mask.sum() == 0 or mask.all():
        return 0.0
    inside = xct_img[mask].mean()
    outside = xct_img[~mask].mean()
    return float(abs(inside - outside))


def get_SAM2_score(
    pca_comp: np.ndarray,
    img_size: int,
    predictor: Any,
    threshold: float,
    get_mask: bool = False,
):
    binary_map = pca_comp[:, 0].reshape(img_size, img_size) > threshold
    mask_input = cv2.resize(binary_map.astype(np.float32), (256, 256))[None]
    mask_input = (mask_input * 2 - 1) * 10
    with torch.inference_mode():
        if not get_mask:
            _, scores, _ = predictor.predict(
                mask_input=mask_input, multimask_output=False
            )
        else:
            masks, scores, _ = predictor.predict(
                mask_input=mask_input, multimask_output=False
            )
    return float(scores[0]) if not get_mask else (masks, float(scores[0]))


def auto_threshold(
    pca_comp: np.ndarray,
    img_size: int,
    predictor: Any,
    n_iter: int = 10,
    lr: float = 0.1,
) -> tuple:
    eps = 0.01 * (pca_comp[:, 0].max() - pca_comp[:, 0].min())
    t_range = np.linspace(0, 1, 20)
    # Initialise best before sweep to avoid NameError if no iteration improves score
    best_t = float(t_range[0])
    best_score = get_SAM2_score(pca_comp, img_size, predictor, best_t)
    for t in t_range:
        score = get_SAM2_score(pca_comp, img_size, predictor, float(t))
        if score > best_score:
            best_score = score
            best_t = float(t)
    t = best_t
    history = [(t, best_score)]
    for _ in range(n_iter):
        grad = (
            get_SAM2_score(pca_comp, img_size, predictor, t + eps)
            - get_SAM2_score(pca_comp, img_size, predictor, t - eps)
        ) / (2 * eps)
        t = float(np.clip(t - lr * grad, 0.0, 1.0))
        score = get_SAM2_score(pca_comp, img_size, predictor, t)
        if score > best_score:
            best_score = score
            best_t = t
            history.append((t, score))
        if abs(score - best_score) < 1e-5:
            break
    best_mask = get_SAM2_score(pca_comp, img_size, predictor, best_t, get_mask=True)[0][
        0
    ].astype(bool)
    return best_mask, best_t, best_score, history
