# this file stores the data loaders and data loading functions for the X-FUSE framework

import numpy as np
import h5py
import os


def load_xct(xct_path: str, recon: str = "Tomopy", **kwargs) -> np.ndarray:
    assert xct_path.endswith(".h5"), "Expected a .h5 file for the xct data."

    recon_methods = {
        "Tomopy": "4-TomopyRecon-tomo",
        "Astra": "4-AstraReconGpu-tomo",
    }

    with h5py.File(xct_path, "r") as f:
        return np.transpose(np.array(f[recon_methods[recon]]["data"]), (1, 2, 0))


def load_xrdct_phase(
    phase_folder: str, phases: list, shape: list, crop: slice = slice(None), **kwargs
) -> list[np.ndarray]:
    xrdct_data = []
    for i in range(len(phases)):
        files = [f for f in os.listdir(phase_folder) if phases[i] in f][crop]

        assert (
            len(files) > 0
        ), f"No files found for phase {phases[i]} in folder {phase_folder}."
        assert any(
            f.endswith(".nxs") for f in files
        ), f"Expected .nxs files for phase {phases[i]}, but found none."
        assert (
            len(files) == shape[0]
        ), f"Expected {shape[0]} files for phase {phases[i]}, but found {len(files)}."

        xrdct_i = np.zeros(shape)
        for idx in range(shape[0]):
            with h5py.File(os.path.join(phase_folder, files[idx]), "r") as f:
                xrdct_i[idx] += np.array(f["data"][:]).squeeze()
        xrdct_data.append(xrdct_i)

    return xrdct_data


def load_diad_xct_zn13x(xct_path: str, **kwargs) -> np.ndarray:
    return load_xct(xct_path, recon="Astra")[150:2000, 350:2200, 350:2200]


def load_diad_xrdct(phase_folder: str, phases: list, **kwargs) -> dict[str, np.ndarray]:
    phase_arrays = load_xrdct_phase(
        phase_folder, shape=(21, 20, 20), phases=phases, crop=slice(5, -1), **kwargs
    )

    return {
        phase: np.flip(arr, axis=2).copy() for phase, arr in zip(phases, phase_arrays)
    }


def load_i13_xct(xct_path: str, **kwargs) -> np.ndarray:
    return load_xct(xct_path, recon="Tomopy")
