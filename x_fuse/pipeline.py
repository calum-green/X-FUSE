import os
import warnings
from pathlib import Path

import cv2
import numpy as np
import torch
import matplotlib.pyplot as plt
from PIL import Image, ImageOps

import hr_dv2.transform as tr
from tqdm.auto import tqdm
from hr_dv2.utils import do_single_pca, rescale_pca

from .config import XFuseConfig
from .loaders import load_diad_xct_zn13x, load_diad_xrdct, load_xrdct_phase
from .utils import invert_image, load_img, overlay_mask, xrd_to_tensor, get_ps_images


# ---------------------------------------------------------------------------
# Stage 0
# ---------------------------------------------------------------------------


def run_data(config: XFuseConfig) -> None:
    """Validate, load, format, and save input data. CPU-only."""
    print(f"\n[X-FUSE] Stage 0 - data  ({config.name})")
    _set_cache_env(config)
    print("  validating paths...")
    _validate_paths(config)
    xct_raw, xrd_raw = _load_raw_data(config)
    print(
        f"  extracting sample (XCT idx={config.xct_sample_idx},"
        f" XRDCT idx={config.xrdct_sample_idx})..."
    )
    xct_sample, xrd_sample = _extract_sample(xct_raw, xrd_raw, config)
    if config.invert:
        print("  inverting images...")
        xct_sample = invert_image(xct_sample)
        xrd_sample = {p: invert_image(a) for p, a in xrd_sample.items()}
    _validate_arrays(xct_sample, xrd_sample)
    img_size = _snap_to_multiple_of_16(xct_sample.shape[0])
    print(f"  effective img_size: {img_size}")
    print("  transforming XCT image...")
    img_tr = tr.get_input_transform(img_size, img_size)
    xct_transformed = _apply_xct_transform(xct_sample, img_tr)
    out_dir = config.output_path / "data"
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"  saving to {out_dir}/")
    np.save(out_dir / "xct.npy", xct_transformed.astype(np.float32))
    for phase, arr in xrd_sample.items():
        np.save(out_dir / f"{phase}_xrd.npy", arr.astype(np.float32))
    _write_data_summary(out_dir, xct_transformed, xrd_sample, config, img_size)
    if config.vis_data:
        _save_data_overview(out_dir, xct_transformed, xrd_sample)
    print("  done.")


# ---------------------------------------------------------------------------
# Stage 1
# ---------------------------------------------------------------------------


def run_features(config: XFuseConfig) -> None:
    """XFuse forward pass + PCA. Requires GPU."""
    print(f"\n[X-FUSE] Stage 1 - features  ({config.name})")
    from .fusion import XFuse

    device = config.resolve_device()
    print(f"  device: {device}")

    data_dir = config.output_path / "data"
    _check_stage_inputs(
        data_dir,
        ["xct.npy"] + [f"{p}_xrd.npy" for p in config.phases],
        prior_stage="data",
    )

    xct = np.load(data_dir / "xct.npy")
    xrd_dict = {p: np.load(data_dir / f"{p}_xrd.npy") for p in config.phases}

    shift_dists = list(config.shift_distances)
    fwd_shift, inv_shift = tr.get_shift_transforms(shift_dists, "Moore")
    fwd_flip, inv_flip = tr.get_flip_transforms()
    fwd, inv = tr.combine_transforms(fwd_shift, fwd_flip, inv_shift, inv_flip)

    img_size = xct.shape[0]
    img_tr = tr.get_input_transform(img_size, img_size)
    xct_tensor, _ = load_img(xct, img_tr)
    xct_tensor = xct_tensor.to(torch.float16).to(device)

    feat_dir = config.output_path / "features"
    feat_dir.mkdir(parents=True, exist_ok=True)

    for phase in tqdm(config.phases, desc="  phases", unit="phase"):
        tqdm.write(f"    [{phase}] building XFuse model...")
        xrd_tensor = xrd_to_tensor(xrd_dict[phase], device)

        net = XFuse(
            config.dino_model,
            config.fusion_method,
            stride=config.stride,
            pca_dim=-1,
            track_grad=False,
            dtype=torch.float16,
            device=device,
            loss_fn=config.loss_fn,
            model_path=config.model_path,
            chk_path=config.chk_path,
            lib_path=config.lib_path,
        )
        net.set_xrd_transforms(xrd_tensor, fwd, inv)
        tqdm.write(f"    [{phase}] running forward pass...")
        feats = net.forward_sequential(xct_tensor, top_k=config.top_k)

        feats_cpu = feats[0].cpu()
        feats_np = tr.to_numpy(feats_cpu)
        feats_flat = tr.flatten(
            feats_np, feats_np.shape[1], feats_np.shape[2], feats_np.shape[0]
        )
        np.save(feat_dir / f"{phase}_feats.npy", feats_np.astype(np.float32))

        tqdm.write(f"    [{phase}] PCA ({config.n_components} components)...")
        pcaed = rescale_pca(
            do_single_pca(
                feats_flat, config.n_components, n_samples=config.n_samples_pca
            )
        )
        np.save(feat_dir / f"{phase}_pca.npy", pcaed.astype(np.float32))

        _save_features_figures(feat_dir, phase, feats_np, pcaed, img_size, config)
    print(f"  features saved to {feat_dir}/")


# ---------------------------------------------------------------------------
# Stage 2
# ---------------------------------------------------------------------------


def run_segment(config: XFuseConfig, threshold: float) -> None:
    """Apply threshold to PCA component 0. CPU-only."""
    print(f"\n[X-FUSE] Stage 2 - segment  ({config.name})")
    print(f"  threshold: {threshold}")
    feat_dir = config.output_path / "features"
    data_dir = config.output_path / "data"
    _check_stage_inputs(
        feat_dir, [f"{p}_pca.npy" for p in config.phases], prior_stage="features"
    )
    _check_stage_inputs(data_dir, ["xct.npy"], prior_stage="data")

    xct = np.load(data_dir / "xct.npy")
    img_size = xct.shape[0]
    seg_dir = config.output_path / "segment"
    seg_dir.mkdir(parents=True, exist_ok=True)
    (seg_dir / "threshold.txt").write_text(str(threshold))

    for phase in tqdm(config.phases, desc="  phases", unit="phase"):
        pcaed = np.load(feat_dir / f"{phase}_pca.npy")
        pca_map = pcaed[:, 0].reshape(img_size, img_size)
        mask = (pca_map > threshold).astype(np.uint8)
        np.save(seg_dir / f"{phase}_mask.npy", mask)

        if config.vis_masks:
            _save_mask_figure(seg_dir, phase, xct, mask, threshold)
    print(f"  masks saved to {seg_dir}/")


# ---------------------------------------------------------------------------
# Stage 3
# ---------------------------------------------------------------------------


def run_refine(config: XFuseConfig) -> None:
    """SAM2 refinement of binary masks. Requires GPU."""
    print(f"\n[X-FUSE] Stage 3 - refine  ({config.name})")
    from sam2.build_sam import build_sam2
    from sam2.sam2_image_predictor import SAM2ImagePredictor

    device = config.resolve_device()
    print(f"  device: {device}")

    seg_dir = config.output_path / "segment"
    data_dir = config.output_path / "data"
    _check_stage_inputs(
        seg_dir, [f"{p}_mask.npy" for p in config.phases], prior_stage="segment"
    )
    _check_stage_inputs(data_dir, ["xct.npy"], prior_stage="data")

    xct = np.load(data_dir / "xct.npy")

    sam2_cfg = f"{config.sam2_folder}{config.sam2_model}.yaml"
    sam2_ckpt = f"{config.sam2_folder}{config.sam2_model}.pt"
    print("  loading SAM2 model...")
    sam2_model = build_sam2(sam2_cfg, sam2_ckpt, device=device)
    predictor = SAM2ImagePredictor(sam2_model)

    xct_pil = ImageOps.autocontrast(
        Image.fromarray((xct * 255).astype(np.uint8)).convert("RGB")
    )
    predictor.set_image(np.array(xct_pil))

    refine_dir = config.output_path / "refine"
    refine_dir.mkdir(parents=True, exist_ok=True)

    for phase in tqdm(config.phases, desc="  phases", unit="phase"):
        mask = np.load(seg_dir / f"{phase}_mask.npy").astype(np.float32)
        mask_input = cv2.resize(mask, (256, 256))[None]
        mask_input = (mask_input * 2 - 1) * 10
        with torch.inference_mode():
            masks, scores, _ = predictor.predict(
                mask_input=mask_input, multimask_output=False
            )
        refined = masks[0].astype(bool)
        np.save(refine_dir / f"{phase}_refined_mask.npy", refined)
        tqdm.write(f"    [{phase}] SAM2 score: {float(scores[0]):.3f}")

        if config.vis_sam2:
            _save_sam2_figure(refine_dir, phase, xct, refined, float(scores[0]))

    config.to_yaml(str(refine_dir / "config.yaml"))
    print(f"  refined masks saved to {refine_dir}/")


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


def _set_cache_env(config: XFuseConfig) -> None:
    if config.torch_cache:
        for var in ("HUGGINGFACE_HUB_CACHE", "TORCH_HOME", "HF_HUB_CACHE", "HF_HOME"):
            os.environ[var] = config.torch_cache


def _validate_paths(config: XFuseConfig) -> None:
    if config.dataset_type == "diad":
        if config.xct_path and not Path(config.xct_path).exists():
            raise FileNotFoundError(f"xct_path not found: {config.xct_path}")
        if config.phase_folder and not Path(config.phase_folder).exists():
            raise FileNotFoundError(f"phase_folder not found: {config.phase_folder}")
        if config.entry_names is None:
            warnings.warn(
                "entry_names not set in config — loader will use its hardcoded "
                "defaults. Set dataset.entry_names in the YAML to silence this.",
                UserWarning,
                stacklevel=3,
            )
    if config.model_path and not Path(config.model_path).exists():
        raise FileNotFoundError(f"model_path not found: {config.model_path}")
    if config.lib_path and not Path(config.lib_path).exists():
        raise FileNotFoundError(f"lib_path not found: {config.lib_path}")


def _load_raw_data(config: XFuseConfig) -> tuple:
    if config.dataset_type == "diad":
        print("  loading XCT...")
        xct_raw = load_diad_xct_zn13x(config.xct_path)
        if config.entry_names:
            xrd_raw = {}
            for phase in config.phases:
                print(f"  loading XRDCT [{phase}]...")
                [arr] = load_xrdct_phase(
                    config.phase_folder,
                    phases=[phase],
                    shape=(21, 20, 20),
                    crop=slice(5, -1),
                    entry_names={phase: config.entry_names[phase]},
                )
                xrd_raw[phase] = np.flip(arr, axis=2).copy()
        else:
            print(f"  loading XRDCT ({', '.join(config.phases)})...")
            xrd_raw = load_diad_xrdct(config.phase_folder, phases=config.phases)
        return xct_raw, xrd_raw

    if config.dataset_type == "porespy":
        if len(config.phases) != 2:
            raise ValueError(
                f"porespy dataset requires exactly 2 phases, got {config.phases}"
            )
        print("  loading porespy data...")
        gray_imgs, low_A, low_B = get_ps_images()
        xrd_raw = {
            config.phases[0]: np.stack(low_A),
            config.phases[1]: np.stack(low_B),
        }
        return np.stack(gray_imgs), xrd_raw

    raise ValueError(f"Unknown dataset_type: {config.dataset_type!r}")


def _snap_to_multiple_of_16(n: int) -> int:
    return (n // 16) * 16


def _extract_sample(xct_raw: np.ndarray, xrd_raw: dict, config: XFuseConfig) -> tuple:
    xct_sample = xct_raw[config.xct_sample_idx]
    xrd_sample = {p: arr[config.xrdct_sample_idx] for p, arr in xrd_raw.items()}
    return xct_sample, xrd_sample


def _validate_arrays(xct: np.ndarray, xrd_dict: dict) -> None:
    if xct.ndim != 2:
        raise ValueError(f"XCT sample must be 2D, got shape {xct.shape}")
    for phase, arr in xrd_dict.items():
        if arr.ndim != 2:
            raise ValueError(
                f"XRD sample for phase '{phase}' must be 2D, got shape {arr.shape}"
            )


def _apply_xct_transform(xct: np.ndarray, transform) -> np.ndarray:
    lo, hi = float(xct.min()), float(xct.max())
    if hi > lo:
        xct = (xct - lo) / (hi - lo)
    tensor, _ = load_img(xct, transform)
    return tensor.permute(1, 2, 0).numpy()[:, :, 0]


def _write_data_summary(
    out_dir: Path, xct: np.ndarray, xrd_dict: dict, config: XFuseConfig, img_size: int
) -> None:
    lines = [
        f"dataset_type: {config.dataset_type}",
        f"phases: {config.phases}",
        f"xct_sample_idx: {config.xct_sample_idx}",
        f"xrdct_sample_idx: {config.xrdct_sample_idx}",
        f"img_size (effective): {img_size}",
        f"xct shape: {xct.shape}, min: {xct.min():.4f}, max: {xct.max():.4f}",
    ]
    for phase, arr in xrd_dict.items():
        lines.append(
            f"{phase}_xrd shape: {arr.shape}, "
            f"min: {arr.min():.4f}, max: {arr.max():.4f}"
        )
    (out_dir / "data_summary.txt").write_text("\n".join(lines))


def _save_data_overview(out_dir: Path, xct: np.ndarray, xrd_dict: dict) -> None:
    n_phases = len(xrd_dict)
    fig, axs = plt.subplots(1, 1 + n_phases, figsize=(6 * (1 + n_phases), 5))
    axs[0].imshow(xct, cmap="gray")
    axs[0].set_title("XCT")
    axs[0].axis("off")
    for i, (phase, arr) in enumerate(xrd_dict.items()):
        axs[i + 1].imshow(arr, cmap="gray", interpolation="lanczos")
        axs[i + 1].set_title(f"XRD — {phase}")
        axs[i + 1].axis("off")
    plt.tight_layout()
    fig.savefig(out_dir / "data_overview.png", dpi=150, bbox_inches="tight")
    plt.show()
    plt.close(fig)


def _check_stage_inputs(directory: Path, filenames: list, prior_stage: str) -> None:
    for fname in filenames:
        p = directory / fname
        if not p.exists():
            raise FileNotFoundError(
                f"Expected '{p}' but it was not found. "
                f"Run --stage {prior_stage} first."
            )


def _save_features_figures(
    feat_dir: Path,
    phase: str,
    feats_np: np.ndarray,
    pcaed: np.ndarray,
    img_size: int,
    config: XFuseConfig,
) -> None:
    if config.vis_fused_maps:
        fig, ax = plt.subplots(figsize=(5, 5))
        ax.imshow(feats_np[0], cmap="viridis")
        ax.set_title(f"Fused features ch0 — {phase}")
        ax.axis("off")
        fig.savefig(feat_dir / f"{phase}_fused.png", dpi=150, bbox_inches="tight")
        plt.show()
        plt.close(fig)

    if config.vis_pca_components:
        n = config.n_components
        fig, axs = plt.subplots(1, n, figsize=(3 * n, 3))
        h = w = img_size
        for i in range(n):
            axs[i].imshow(pcaed[:, i].reshape(h, w), cmap="viridis")
            axs[i].set_title(f"PCA {i + 1}")
            axs[i].axis("off")
        plt.suptitle(f"PCA components — {phase}")
        plt.tight_layout()
        fig.savefig(feat_dir / f"{phase}_pca_grid.png", dpi=150, bbox_inches="tight")
        plt.show()
        plt.close(fig)


def _save_mask_figure(
    seg_dir: Path, phase: str, xct: np.ndarray, mask: np.ndarray, threshold: float
) -> None:
    fig, axs = plt.subplots(1, 2, figsize=(10, 5))
    axs[0].imshow(xct, cmap="gray")
    axs[0].set_title("XCT")
    axs[0].axis("off")
    axs[1].imshow(overlay_mask(xct, mask.astype(bool)))
    axs[1].set_title(f"{phase} mask (threshold={threshold:.3f})")
    axs[1].axis("off")
    plt.tight_layout()
    fig.savefig(seg_dir / f"{phase}_mask.png", dpi=150, bbox_inches="tight")
    plt.show()
    plt.close(fig)


def _save_sam2_figure(
    refine_dir: Path,
    phase: str,
    xct: np.ndarray,
    refined: np.ndarray,
    score: float,
) -> None:
    fig, axs = plt.subplots(1, 2, figsize=(10, 5))
    axs[0].imshow(xct, cmap="gray")
    axs[0].set_title("XCT")
    axs[0].axis("off")
    axs[1].imshow(overlay_mask(xct, refined))
    axs[1].set_title(f"{phase} SAM2 refined (score={score:.3f})")
    axs[1].axis("off")
    plt.tight_layout()
    fig.savefig(refine_dir / f"{phase}_sam2_overlay.png", dpi=150, bbox_inches="tight")
    plt.show()
    plt.close(fig)
