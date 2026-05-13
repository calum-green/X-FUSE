from dataclasses import dataclass, field
from dataclasses import replace as _replace
from pathlib import Path
from typing import Optional

import yaml


@dataclass
class XFuseConfig:
    # required
    name: str

    # run
    output_dir: str = "outputs/"

    # environment
    torch_cache: str = ""

    # dataset
    dataset_type: str = "diad"
    xct_path: Optional[str] = None
    phase_folder: Optional[str] = None
    phases: list = field(default_factory=lambda: ["Na", "Zn"])
    entry_names: Optional[dict] = None
    n_samples: int = 10
    downsample_factor: int = 10
    xct_sample_idx: int = -1
    xrdct_sample_idx: int = -1

    # model
    dino_model: str = "nope_dv3_vits16plus_1625"
    model_path: Optional[str] = None
    chk_path: Optional[str] = None
    lib_path: Optional[str] = None
    stride: int = 4
    fusion_method: str = "gating"
    loss_fn: str = "bce"
    top_k: Optional[int] = None
    invert: bool = False
    device: Optional[str] = None

    # augmentation
    shift_distances: list = field(default_factory=lambda: [1, 2])
    use_flip: bool = True

    # pca
    n_components: int = 10
    n_samples_pca: int = 5000

    # sam2
    sam2_folder: str = ""
    sam2_model: str = "sam2.1_hiera_small"

    # vis
    vis_data: bool = True
    vis_dino_features: bool = False
    vis_fused_maps: bool = True
    vis_pca_components: bool = True
    vis_masks: bool = True
    vis_sam2: bool = True

    @classmethod
    def from_yaml(cls, path: str) -> "XFuseConfig":
        with open(path) as f:
            d = yaml.safe_load(f)
        run = d.get("run", {})
        env = d.get("environment", {})
        ds = d.get("dataset", {})
        model = d.get("model", {})
        aug = d.get("augmentation", {})
        pca = d.get("pca", {})
        sam2 = d.get("sam2", {})
        vis = d.get("vis", {})
        return cls(
            name=run["name"],
            output_dir=run.get("output_dir", "outputs/"),
            torch_cache=env.get("torch_cache", ""),
            dataset_type=ds.get("type", "diad"),
            xct_path=ds.get("xct_path"),
            phase_folder=ds.get("phase_folder"),
            phases=ds.get("phases", ["Na", "Zn"]),
            entry_names=ds.get("entry_names"),
            n_samples=ds.get("n_samples", 10),
            downsample_factor=ds.get("downsample_factor", 10),
            xct_sample_idx=ds.get("xct_sample_idx", -1),
            xrdct_sample_idx=ds.get("xrdct_sample_idx", -1),
            dino_model=model.get("dino_model", "nope_dv3_vits16plus_1625"),
            model_path=model.get("model_path"),
            chk_path=model.get("chk_path"),
            lib_path=model.get("lib_path"),
            stride=model.get("stride", 4),
            fusion_method=model.get("fusion_method", "gating"),
            loss_fn=model.get("loss_fn", "bce"),
            top_k=model.get("top_k"),
            invert=model.get("invert", False),
            device=model.get("device"),
            shift_distances=aug.get("shift_distances", [1, 2]),
            use_flip=aug.get("use_flip", True),
            n_components=pca.get("n_components", 10),
            n_samples_pca=pca.get("n_samples", 5000),
            sam2_folder=sam2.get("folder", ""),
            sam2_model=sam2.get("model", "sam2.1_hiera_small"),
            vis_data=vis.get("data", True),
            vis_dino_features=vis.get("dino_features", False),
            vis_fused_maps=vis.get("fused_maps", True),
            vis_pca_components=vis.get("pca_components", True),
            vis_masks=vis.get("masks", True),
            vis_sam2=vis.get("sam2", True),
        )

    def to_yaml(self, path: str) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        d = {
            "run": {"name": self.name, "output_dir": self.output_dir},
            "environment": {"torch_cache": self.torch_cache},
            "dataset": {
                "type": self.dataset_type,
                "xct_path": self.xct_path,
                "phase_folder": self.phase_folder,
                "phases": self.phases,
                "entry_names": self.entry_names,
                "n_samples": self.n_samples,
                "downsample_factor": self.downsample_factor,
                "xct_sample_idx": self.xct_sample_idx,
                "xrdct_sample_idx": self.xrdct_sample_idx,
            },
            "model": {
                "dino_model": self.dino_model,
                "model_path": self.model_path,
                "chk_path": self.chk_path,
                "lib_path": self.lib_path,
                "stride": self.stride,
                "fusion_method": self.fusion_method,
                "loss_fn": self.loss_fn,
                "top_k": self.top_k,
                "invert": self.invert,
                "device": self.device,
            },
            "augmentation": {
                "shift_distances": self.shift_distances,
                "use_flip": self.use_flip,
            },
            "pca": {"n_components": self.n_components, "n_samples": self.n_samples_pca},
            "sam2": {"folder": self.sam2_folder, "model": self.sam2_model},
            "vis": {
                "data": self.vis_data,
                "dino_features": self.vis_dino_features,
                "fused_maps": self.vis_fused_maps,
                "pca_components": self.vis_pca_components,
                "masks": self.vis_masks,
                "sam2": self.vis_sam2,
            },
        }
        with open(path, "w") as f:
            yaml.dump(d, f, default_flow_style=False, sort_keys=False)

    def replace(self, **kwargs) -> "XFuseConfig":
        return _replace(self, **kwargs)

    @property
    def output_path(self) -> Path:
        return Path(self.output_dir) / self.name

    def resolve_device(self) -> str:
        if self.device is not None:
            return self.device
        import torch

        return "cuda" if torch.cuda.is_available() else "cpu"
