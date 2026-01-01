#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
from pathlib import Path

from huggingface_hub import snapshot_download


def _default_cache_dir(root: Path) -> Path:
    return root / "build-tools" / "zimage-train" / "hf-cache"


def main() -> None:
    parser = argparse.ArgumentParser(description="Download Z-Image Turbo (partial) weights from Hugging Face.")
    parser.add_argument("--repo", default="Tongyi-MAI/Z-Image-Turbo", help="Hugging Face repo id.")
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
        "--include",
        default="transformer,vae,tokenizer,text_encoder",
        help="Comma-separated components to download: transformer, vae, text_encoder, tokenizer, assets.",
    )
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[2]
    cache_dir = Path(args.cache_dir).expanduser() if args.cache_dir else _default_cache_dir(root)
    cache_dir.mkdir(parents=True, exist_ok=True)

    include = {s.strip() for s in args.include.split(",") if s.strip()}
    allow_patterns: list[str] = []
    if "transformer" in include:
        allow_patterns += ["transformer/*"]
    if "vae" in include:
        allow_patterns += ["vae/*"]
    if "text_encoder" in include:
        allow_patterns += ["text_encoder/*"]
    if "tokenizer" in include:
        allow_patterns += ["tokenizer/*"]
    if "assets" in include:
        allow_patterns += ["assets/*"]
    # Always keep config files if present.
    allow_patterns += ["*.json", "*.txt", "*.md", ".gitattributes"]

    # This env var is honored by HF tooling; useful when running the script from Xcode / GUI.
    os.environ.setdefault("HF_HOME", str(cache_dir))

    print(f"[download] repo={args.repo}")
    print(f"[download] revision={args.revision or 'latest'}")
    print(f"[download] cache_dir={cache_dir}")
    print(f"[download] allow_patterns={allow_patterns}")

    snapshot_path = snapshot_download(
        repo_id=args.repo,
        revision=args.revision,
        cache_dir=str(cache_dir),
        allow_patterns=allow_patterns,
    )
    print(f"[download] snapshot_path={snapshot_path}")


if __name__ == "__main__":
    main()
