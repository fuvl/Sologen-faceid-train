#!/usr/bin/env python3

from __future__ import annotations

import argparse
import os
import pathlib
import subprocess
import sys


def ensure_repo(repo_dir: pathlib.Path) -> None:
    if repo_dir.exists():
        return
    repo_dir.parent.mkdir(parents=True, exist_ok=True)
    subprocess.check_call(
        [
            "git",
            "clone",
            "--depth",
            "1",
            "https://github.com/xuexingyu24/MobileFaceNet_Tutorial_Pytorch.git",
            str(repo_dir),
        ]
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a small ArcFace-style MobileFaceNet CoreML model for iOS.")
    parser.add_argument("--repo-dir", default="build-tools/faceid/MobileFaceNet_Tutorial_Pytorch", help="Where to clone the PyTorch repo.")
    parser.add_argument("--output", default="build-tools/faceid/ArcFace.mlpackage", help="Output path (.mlpackage or .mlmodel).")
    parser.add_argument("--deployment-target", default="iOS18", help="Minimum deployment target (e.g. iOS16, iOS17, iOS18).")
    args = parser.parse_args()

    repo_dir = pathlib.Path(args.repo_dir).expanduser().resolve()
    ensure_repo(repo_dir)

    # Import their MobileFaceNet definition + weights (kept out of git under build-tools/).
    sys.path.insert(0, str(repo_dir))
    from face_model import MobileFaceNet  # type: ignore

    import torch  # type: ignore
    import coremltools as ct  # type: ignore

    weights_path = repo_dir / "Weights" / "MobileFace_Net"
    if not weights_path.exists():
        raise FileNotFoundError(f"Missing weights: {weights_path}")

    state = torch.load(weights_path, map_location="cpu", weights_only=True)
    net = MobileFaceNet(512)
    net.load_state_dict(state)
    net.eval()

    example = torch.zeros(1, 3, 112, 112)
    traced = torch.jit.trace(net, example)

    # Their preprocessing is equivalent to:
    #   x = (pixel * (2/255)) - 1    (channel order as provided; use BGR to match OpenCV pipelines)
    scale = 2.0 / 255.0
    bias = [-1.0, -1.0, -1.0]
    inputs = [
        ct.ImageType(
            name="image",
            shape=example.shape,
            color_layout=ct.colorlayout.BGR,
            scale=scale,
            bias=bias,
        )
    ]

    deployment_target = getattr(ct.target, args.deployment_target)

    mlmodel = ct.convert(
        traced,
        inputs=inputs,
        minimum_deployment_target=deployment_target,
        compute_units=ct.ComputeUnit.CPU_AND_NE,
    )

    out_path = pathlib.Path(args.output).expanduser().resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    mlmodel.save(str(out_path))
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
