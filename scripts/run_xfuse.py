#!/usr/bin/env python3
"""X-FUSE pipeline CLI.

Usage:
    python scripts/run_xfuse.py --config configs/my_run.yaml --stage data
    python scripts/run_xfuse.py --config configs/my_run.yaml --stage features
    python scripts/run_xfuse.py --config configs/my_run.yaml --stage segment \
        --threshold 0.2
    python scripts/run_xfuse.py --config configs/my_run.yaml --stage refine
    python scripts/run_xfuse.py --config configs/my_run.yaml --stage features \
        --vis-all
"""
import argparse

from x_fuse.config import XFuseConfig
from x_fuse.pipeline import run_data, run_features, run_refine, run_segment


def main():
    parser = argparse.ArgumentParser(description="X-FUSE pipeline runner")
    parser.add_argument("--config", required=True, help="Path to YAML config file")
    parser.add_argument(
        "--stage",
        required=True,
        choices=["data", "features", "segment", "refine"],
        help="Pipeline stage to run",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=None,
        help="Threshold for --stage segment (required for that stage)",
    )
    parser.add_argument(
        "--vis-all",
        action="store_true",
        help="Override all vis.* flags to True without editing the YAML",
    )
    args = parser.parse_args()

    config = XFuseConfig.from_yaml(args.config)

    if args.vis_all:
        config = config.replace(
            vis_data=True,
            vis_dino_features=True,
            vis_fused_maps=True,
            vis_pca_components=True,
            vis_masks=True,
            vis_sam2=True,
        )

    if args.stage == "data":
        run_data(config)
    elif args.stage == "features":
        run_features(config)
    elif args.stage == "segment":
        if args.threshold is None:
            parser.error("--threshold is required for --stage segment")
        run_segment(config, args.threshold)
    elif args.stage == "refine":
        run_refine(config)


if __name__ == "__main__":
    main()
