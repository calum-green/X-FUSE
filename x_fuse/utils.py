import porespy as ps
import numpy as np
import matplotlib.pyplot as plt
import cv2


def get_multiphase(
    imA, imA_kwargs, imB, imB_kwargs, extract_spheres=False, vis=False, dataset_size=1
):
    """
    Input: imA, imB :-> images for phase A and phase B (e.g. blobs with different porosities)
    Output: (imgA, phaseA, phaseB):-> list(imgA.astype(float32)), list(phaseA.astype(float32)), list(phaseB.astype(float32))
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
        im_B = (
            imB(**imB_kwargs)
            if extract_spheres is False
            else imB(im_or_shape=im_A, **imB_kwargs)
        )

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


def downsample_xrdct(gray_imgs, phase_A_masks, phase_B_masks, factor=2):
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

    return gray_imgs, downsampled_phase_A_masks, downsampled_phase_B_masks
