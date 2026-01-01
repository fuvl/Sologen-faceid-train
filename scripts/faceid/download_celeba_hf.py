#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
from pathlib import Path

from huggingface_hub import snapshot_download


def _default_cache_dir(root: Path) -> Path:
    return root / "build-tools" / "zimage-train" / "hf-cache"


def main() -> None:
    parser = argparse.ArgumentParser(description="Download CelebA parquet shards from Hugging Face.")
    parser.add_argument("--repo", default="flwrlabs/celeba", help="Hugging Face dataset repo id.")
    parser.add_argument(
        "--revision",
        default=None,
        help="Optional git revision (commit hash / tag). Defaults to the latest.",
    )
    parser.add_argument(
        "--cache-dir",
        default=None,
        help="Cache directory. Defaults to build-tools/zimage-train/hf-cache.",
    )
    parser.add_argument(
        "--config",
        default="img_align+identity+attr",
        help="Dataset config folder name on the repo (e.g. img_align+identity+attr).",
    )
    parser.add_argument(
        "--splits",
        default="train,valid,test",
        help="Comma-separated splits to download: train,valid,test.",
    )
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[2]
    cache_dir = Path(args.cache_dir).expanduser() if args.cache_dir else _default_cache_dir(root)
    cache_dir.mkdir(parents=True, exist_ok=True)

    splits = {s.strip() for s in args.splits.split(",") if s.strip()}
    allow_patterns: list[str] = ["README.md", "LICENSE", ".gitattributes"]
    for split in sorted(splits):
        allow_patterns.append(f"{args.config}/{split}-*.parquet")

    # This env var is honored by HF tooling; useful when running the script from Xcode / GUI.
    os.environ.setdefault("HF_HOME", str(cache_dir))

    print(f"[download] repo={args.repo}")
    print(f"[download] revision={args.revision or 'latest'}")
    print(f"[download] cache_dir={cache_dir}")
    print(f"[download] config={args.config}")
    print(f"[download] splits={sorted(splits)}")
    print(f"[download] allow_patterns={allow_patterns}")

    snapshot_path = snapshot_download(
        repo_id=args.repo,
        repo_type="dataset",
        revision=args.revision,
        cache_dir=str(cache_dir),
        allow_patterns=allow_patterns,
    )
    print(f"[download] snapshot_path={snapshot_path}")


if __name__ == "__main__":
    main()

