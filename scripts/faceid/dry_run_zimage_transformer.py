#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import torch
from huggingface_hub import snapshot_download
from diffusers.models.autoencoders import AutoencoderKL
from diffusers.models.transformers import ZImageTransformer2DModel


def main() -> None:
    parser = argparse.ArgumentParser(description="Sanity-check Z-Image transformer+VAE loading and a single forward.")
    parser.add_argument("--repo", default="Tongyi-MAI/Z-Image-Turbo")
    parser.add_argument("--cache-dir", default=None, help="HF cache dir (same used by download script).")
    parser.add_argument(
        "--load-weights",
        action="store_true",
        help="Actually load the transformer+VAE weights and run a single forward (requires lots of RAM).",
    )
    parser.add_argument("--height", type=int, default=512)
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--prompt-len", type=int, default=16)
    parser.add_argument("--token-count", type=int, default=4, help="Extra identity tokens appended to prompt embeds.")
    args = parser.parse_args()

    cache_dir = Path(args.cache_dir).expanduser() if args.cache_dir else None

    snapshot_path = snapshot_download(
        repo_id=args.repo,
        cache_dir=str(cache_dir) if cache_dir else None,
        allow_patterns=["transformer/*", "vae/*", "tokenizer/*", "*.json", "*.txt", "*.md", ".gitattributes"],
        local_files_only=True,
    )
    print("[dry-run] snapshot_path:", snapshot_path)

    if not args.load_weights:
        print("[dry-run] OK (files present). Re-run with --load-weights to execute a forward pass.")
        return

    device = torch.device("cpu")
    dtype = torch.float16

    print("[dry-run] Loading transformer weights (this can require >24GB RAM)...")
    transformer = ZImageTransformer2DModel.from_pretrained(
        args.repo,
        subfolder="transformer",
        cache_dir=str(cache_dir) if cache_dir else None,
        torch_dtype=dtype,
        low_cpu_mem_usage=True,
    )
    transformer.eval()

    print("[dry-run] Loading VAE weights...")
    vae = AutoencoderKL.from_pretrained(
        args.repo,
        subfolder="vae",
        cache_dir=str(cache_dir) if cache_dir else None,
        torch_dtype=dtype,
        low_cpu_mem_usage=True,
    )
    vae.eval()

    # Fake latents (NHWC handled internally by Z-Image; pipeline uses NCHW input latents).
    latent_h = args.height // 8
    latent_w = args.width // 8
    latents = torch.randn((1, transformer.in_channels, latent_h, latent_w), device=device, dtype=torch.float32)

    # Fake prompt embeds (B=1) + extra identity tokens appended.
    prompt_embeds = torch.zeros((args.prompt_len + args.token_count, 2560), device=device, dtype=dtype)
    prompt_embeds_list = [prompt_embeds]

    # Z-Image expects x as list of [C, F=1, H, W] tensors.
    latent_model_input = latents.to(dtype=transformer.dtype).unsqueeze(2)
    latent_model_input_list = list(latent_model_input.unbind(dim=0))

    # Flow matching time is normalized [0,1]; t=0 at start (sigma=1), t=1 at end (sigma=0).
    t = torch.tensor([0.5], device=device, dtype=torch.float32)

    with torch.no_grad():
        out_list = transformer(latent_model_input_list, t, prompt_embeds_list, return_dict=False)[0]
    out = torch.stack(out_list, dim=0)
    print("[dry-run] OK. output shape:", tuple(out.shape), "dtype:", out.dtype)


if __name__ == "__main__":
    main()
