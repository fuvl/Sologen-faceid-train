#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Export a trained ZImageFaceIDAdapter (.pth) to CoreML (.mlpackage).")
    parser.add_argument("--weights", default="build-tools/faceid/ZImageFaceIDAdapter.pth")
    parser.add_argument("--output", default="build-tools/faceid/ZImageFaceIDAdapter.mlpackage")
    parser.add_argument("--deployment-target", default="iOS18")
    args = parser.parse_args()

    import numpy as np  # type: ignore
    import torch  # type: ignore
    import coremltools as ct  # type: ignore

    from train_zimage_faceid_adapter import ZImageFaceIDAdapter  # type: ignore

    root = Path(__file__).resolve().parents[2]
    weights_path = (root / args.weights).resolve() if not Path(args.weights).is_absolute() else Path(args.weights)
    out_path = (root / args.output).resolve() if not Path(args.output).is_absolute() else Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    payload = torch.load(weights_path, map_location="cpu", weights_only=False)
    token_count = int(payload["token_count"])
    adapter = ZImageFaceIDAdapter(token_count=token_count)
    adapter.load_state_dict(payload["state_dict"])
    adapter.eval()

    example = torch.zeros(1, 512)
    traced = torch.jit.trace(adapter, example)

    deployment_target = getattr(ct.target, args.deployment_target)
    mlmodel = ct.convert(
        traced,
        inputs=[
            ct.TensorType(name="embedding", shape=example.shape, dtype=np.float32),
        ],
        outputs=[
            ct.TensorType(name="tokens", dtype=np.float16),
        ],
        minimum_deployment_target=deployment_target,
        compute_units=ct.ComputeUnit.CPU_AND_NE,
    )
    mlmodel.save(str(out_path))
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
