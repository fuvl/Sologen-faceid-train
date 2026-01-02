#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import random
import sys
import time
from contextlib import nullcontext
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms
from transformers import AutoModel, AutoTokenizer

from diffusers.models.autoencoders import AutoencoderKL
from diffusers.models.transformers import ZImageTransformer2DModel


@dataclass(frozen=True)
class TrainConfig:
    repo: str
    cache_dir: Path | None
    data_dir: Path | None
    identity_file: Path | None
    hf_dataset: str | None
    hf_config: str
    hf_split: str
    output: Path
    steps: int
    batch_size: int
    height: int
    width: int
    prompt_len: int
    prompt_template: str
    token_count: int
    lr: float
    device: torch.device
    dtype: torch.dtype
    attention_backend: str | None
    prompt_mode: str
    prompt_cache_size: int
    text_encoder_on_cpu: bool
    token_scale: float
    grad_clip: float
    gradient_checkpointing: bool
    resume: Path | None
    save_every: int


class ZImageFaceIDAdapter(nn.Module):
    def __init__(
        self,
        face_dim: int = 512,
        embed_dim: int = 2560,
        token_count: int = 4,
        hidden_dim: int = 1024,
        token_scale: float = 0.01,
    ):
        super().__init__()
        self.token_count = token_count
        self.embed_dim = embed_dim
        self.token_scale = float(token_scale)

        out = nn.Linear(hidden_dim, token_count * embed_dim)
        # Start from a near-no-op: initial FaceID tokens should not destabilize the frozen base model.
        nn.init.zeros_(out.weight)
        nn.init.zeros_(out.bias)
        self.net = nn.Sequential(nn.Linear(face_dim, hidden_dim), nn.SiLU(), out)

    def forward(self, face_embedding: torch.Tensor) -> torch.Tensor:
        # face_embedding: [B, 512]
        out = self.net(face_embedding)  # [B, token_count*embed_dim]
        out = out * self.token_scale
        return out.view(face_embedding.shape[0], self.token_count, self.embed_dim)


def l2_normalize(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    return x / (x.norm(dim=-1, keepdim=True) + eps)


def _ensure_mobilefacenet_repo(repo_dir: Path) -> None:
    if repo_dir.exists():
        return
    repo_dir.parent.mkdir(parents=True, exist_ok=True)
    import subprocess

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


def _load_mobilefacenet(repo_dir: Path, device: torch.device) -> nn.Module:
    # MobileFaceNet is kept under build-tools (ignored by git). This is a training-only dependency.
    _ensure_mobilefacenet_repo(repo_dir)
    sys.path.insert(0, str(repo_dir))
    from face_model import MobileFaceNet  # type: ignore

    weights_path = repo_dir / "Weights" / "MobileFace_Net"
    state = torch.load(weights_path, map_location="cpu", weights_only=True)
    net = MobileFaceNet(512)
    net.load_state_dict(state)
    net.eval().to(device)
    for p in net.parameters():
        p.requires_grad_(False)
    return net


def _iter_image_paths(data_dir: Path) -> list[Path]:
    exts = {".jpg", ".jpeg", ".png", ".webp"}
    paths = [p for p in data_dir.rglob("*") if p.is_file() and p.suffix.lower() in exts]
    paths.sort()
    return paths


def _parse_identity_file(identity_file: Path, images_dir: Path) -> tuple[list[int], dict[int, list[Path]]]:
    id_to_paths: dict[int, list[Path]] = {}
    missing = 0
    with identity_file.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) < 2:
                continue
            filename, identity = parts[0], parts[1]
            p = images_dir / filename
            if not p.is_file():
                missing += 1
                continue
            try:
                ident = int(identity)
            except ValueError:
                continue
            id_to_paths.setdefault(ident, []).append(p)

    id_to_paths = {k: v for k, v in id_to_paths.items() if len(v) >= 2}
    if not id_to_paths:
        raise SystemExit(f"No identities with >=2 images found for identity_file={identity_file}")

    valid_ids = sorted(id_to_paths.keys())
    if missing:
        print(f"[data] identity_file missing_paths={missing}")
    print(f"[data] identities={len(valid_ids)} (>=2 images)")
    return valid_ids, id_to_paths


def _load_hf_dataset_index(
    dataset_id: str,
    config_name: str,
    split: str,
    cache_dir: Path | None,
) -> tuple[object, list[int], dict[int, list[int]]]:
    try:
        from datasets import load_dataset  # type: ignore
    except Exception as e:  # pragma: no cover
        raise SystemExit("datasets is required for --hf-dataset; install scripts/faceid requirements") from e

    ds = load_dataset(
        dataset_id,
        config_name,
        split=split,
        cache_dir=str(cache_dir) if cache_dir else None,
    )
    celeb_ids = ds["celeb_id"]
    id_to_indices: dict[int, list[int]] = {}
    for i, cid in enumerate(celeb_ids):
        id_to_indices.setdefault(int(cid), []).append(i)
    id_to_indices = {k: v for k, v in id_to_indices.items() if len(v) >= 2}
    if not id_to_indices:
        raise SystemExit(f"No identities with >=2 images found for hf={dataset_id}/{config_name}:{split}")
    valid_ids = sorted(id_to_indices.keys())
    print(f"[data] hf_dataset={dataset_id} config={config_name} split={split} rows={len(ds)}")
    print(f"[data] identities={len(valid_ids)} (>=2 images)")
    return ds, valid_ids, id_to_indices


_CELEBA_PROMPT_KEYS: list[str] = [
    "Male",
    "Young",
    "Smiling",
    "Eyeglasses",
    "Bangs",
    "Bald",
    "Black_Hair",
    "Blond_Hair",
    "Brown_Hair",
    "Gray_Hair",
    "Wavy_Hair",
    "Straight_Hair",
    "Mustache",
    "Goatee",
    "No_Beard",
    "Wearing_Hat",
    "Wearing_Earrings",
    "Wearing_Necklace",
    "Wearing_Necktie",
    "Wearing_Lipstick",
]


def _celeba_bool(row: dict, key: str) -> bool:
    v = row.get(key, False)
    try:
        return bool(v)
    except Exception:
        return False


def _build_celeba_prompt(row: dict) -> str:
    male = _celeba_bool(row, "Male")
    young = _celeba_bool(row, "Young")
    smiling = _celeba_bool(row, "Smiling")

    subject = "man" if male else "woman"
    age = "young " if young else ""

    parts: list[str] = [f"a portrait photo of a {age}{subject}"]

    # Hair / face
    if _celeba_bool(row, "Bald"):
        parts.append("bald")
    else:
        hair_color = None
        for key, label in [
            ("Black_Hair", "black"),
            ("Brown_Hair", "brown"),
            ("Blond_Hair", "blond"),
            ("Gray_Hair", "gray"),
        ]:
            if _celeba_bool(row, key):
                hair_color = label
                break
        if hair_color:
            parts.append(f"with {hair_color} hair")
        if _celeba_bool(row, "Bangs"):
            parts.append("with bangs")
        if _celeba_bool(row, "Wavy_Hair"):
            parts.append("wavy hair")
        elif _celeba_bool(row, "Straight_Hair"):
            parts.append("straight hair")

    if _celeba_bool(row, "Eyeglasses"):
        parts.append("wearing glasses")

    if smiling:
        parts.append("smiling")

    if _celeba_bool(row, "No_Beard"):
        parts.append("clean shaven")
    else:
        if _celeba_bool(row, "Mustache"):
            parts.append("with a mustache")
        if _celeba_bool(row, "Goatee"):
            parts.append("with a goatee")

    # Accessories
    if _celeba_bool(row, "Wearing_Hat"):
        parts.append("wearing a hat")
    if _celeba_bool(row, "Wearing_Earrings"):
        parts.append("wearing earrings")
    if _celeba_bool(row, "Wearing_Necklace"):
        parts.append("wearing a necklace")
    if _celeba_bool(row, "Wearing_Necktie"):
        parts.append("wearing a necktie")
    if _celeba_bool(row, "Wearing_Lipstick") and not male:
        parts.append("wearing lipstick")

    parts.append("high quality")

    # Keep it short and stable.
    return ", ".join(parts)


class _PromptEmbedCache:
    def __init__(self, max_items: int) -> None:
        self._max_items = max(0, int(max_items))
        self._items: "OrderedDict[str, torch.Tensor]" = OrderedDict()

    def get(self, key: str) -> torch.Tensor | None:
        if self._max_items <= 0:
            return None
        v = self._items.get(key)
        if v is None:
            return None
        self._items.move_to_end(key)
        return v

    def put(self, key: str, value: torch.Tensor) -> None:
        if self._max_items <= 0:
            return
        self._items[key] = value
        self._items.move_to_end(key)
        while len(self._items) > self._max_items:
            self._items.popitem(last=False)


def wrap_zimage_prompt(prompt: str) -> str:
    # Qwen chat-format prompt template used by Z-Image Turbo.
    return f"<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n"


class _ZImageTextEncoder:
    def __init__(
        self,
        repo: str,
        cache_dir: Path | None,
        prompt_len: int,
        prompt_template: str,
        device: torch.device,
        dtype: torch.dtype,
        on_cpu: bool,
    ) -> None:
        self.prompt_len = int(prompt_len)
        self.prompt_template = str(prompt_template)
        model_device = torch.device("cpu") if on_cpu else device

        tokenizer_kwargs = {
            "cache_dir": str(cache_dir) if cache_dir else None,
            "subfolder": "tokenizer",
            "use_fast": True,
        }
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(repo, **tokenizer_kwargs)
        except Exception:
            self.tokenizer = AutoTokenizer.from_pretrained(repo, trust_remote_code=True, **tokenizer_kwargs)

        if getattr(self.tokenizer, "pad_token_id", None) is None:
            eos = getattr(self.tokenizer, "eos_token", None)
            if eos is not None:
                self.tokenizer.pad_token = eos

        model_kwargs = {
            "cache_dir": str(cache_dir) if cache_dir else None,
            "subfolder": "text_encoder",
            "torch_dtype": dtype,
            "low_cpu_mem_usage": True,
        }
        try:
            self.model = AutoModel.from_pretrained(repo, **model_kwargs)
        except Exception:
            self.model = AutoModel.from_pretrained(repo, trust_remote_code=True, **model_kwargs)

        self.model.eval().to(model_device)
        for p in self.model.parameters():
            p.requires_grad_(False)
        self._device = device
        self._model_device = model_device
        self._dtype = dtype

    @torch.inference_mode()
    def encode(self, prompt: str) -> torch.Tensor:
        if self.prompt_template == "zimage_chat":
            prompt = wrap_zimage_prompt(prompt)
        tok = self.tokenizer(
            prompt,
            add_special_tokens=False,
            padding="max_length",
            truncation=True,
            max_length=self.prompt_len,
            return_tensors="pt",
        )
        tok = {k: v.to(self._model_device) for k, v in tok.items()}
        try:
            out = self.model(**tok, return_dict=True)
        except TypeError:
            out = self.model(**tok)
        hidden = getattr(out, "last_hidden_state", None)
        if hidden is None:
            hidden = out[0]

        if hidden.shape[1] != self.prompt_len:
            hidden = hidden[:, : self.prompt_len, :]

        if hidden.shape[-1] != 2560:
            raise SystemExit(f"text_encoder hidden size mismatch: expected 2560, got {hidden.shape[-1]}")

        # Keep as [T, 2560] on CPU for caching.
        emb = hidden[0].to(dtype=self._dtype).detach().to("cpu")
        return emb

    def to_model_device(self) -> torch.device:
        return self._model_device

    def to_train_device(self, emb_cpu: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
        return emb_cpu.to(self._device, dtype=dtype, non_blocking=True)


def _tensor_stats(name: str, t: torch.Tensor) -> str:
    t = t.detach()
    t32 = t.float()
    nan = torch.isnan(t32).sum().item()
    inf = torch.isinf(t32).sum().item()
    mn = t32.amin().item()
    mx = t32.amax().item()
    mean = t32.mean().item()
    return f"{name}: min={mn:.6g} max={mx:.6g} mean={mean:.6g} nan={nan} inf={inf} shape={list(t.shape)} dtype={t.dtype}"


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a Z-Image FaceID adapter (scaffold).")
    parser.add_argument("--repo", default="Tongyi-MAI/Z-Image-Turbo")
    parser.add_argument("--cache-dir", default="build-tools/zimage-train/hf-cache")
    parser.add_argument("--data-dir", default="build-tools/faceid/MobileFaceNet_Tutorial_Pytorch/images")
    parser.add_argument(
        "--identity-file",
        default=None,
        help="Optional identity mapping file (filename -> id). When set, samples (ref,target) pairs within an id.",
    )
    parser.add_argument(
        "--hf-dataset",
        default=None,
        help="Optional Hugging Face dataset id (e.g. flwrlabs/celeba). When set, ignores --data-dir.",
    )
    parser.add_argument(
        "--hf-config",
        default="img_align+identity+attr",
        help="HF dataset config name (for flwrlabs/celeba: img_align+identity+attr).",
    )
    parser.add_argument("--hf-split", default="train", help="HF dataset split (train/valid/test).")
    parser.add_argument("--output", default="build-tools/faceid/ZImageFaceIDAdapter.pth")
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--height", type=int, default=512)
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--prompt-len", type=int, default=16)
    parser.add_argument(
        "--prompt-template",
        choices=["zimage_chat", "plain"],
        default="zimage_chat",
        help="Text prompt template used before tokenization. Must match your runtime for FaceID to work.",
    )
    parser.add_argument("--token-count", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda", "mps"], default="auto")
    parser.add_argument("--dtype", choices=["fp16", "bf16", "fp32"], default="fp16")
    parser.add_argument(
        "--attention-backend",
        default=None,
        help="Attention backend for Z-Image transformer (e.g. 'native', '_native_flash', '_native_efficient').",
    )
    parser.add_argument(
        "--prompt-mode",
        choices=["zeros", "celeba_attrs"],
        default=None,
        help="Prompt conditioning mode. Defaults to celeba_attrs for --hf-dataset, otherwise zeros.",
    )
    parser.add_argument("--prompt-cache-size", type=int, default=4096, help="In-memory cache size for prompt embeds.")
    parser.add_argument(
        "--text-encoder-on-cpu",
        action="store_true",
        help="Run the Z-Image text encoder on CPU to reduce GPU memory usage (slower).",
    )
    parser.add_argument(
        "--token-scale",
        type=float,
        default=0.01,
        help="Initial scale factor for adapter-produced FaceID tokens (smaller = more stable).",
    )
    parser.add_argument(
        "--grad-clip",
        type=float,
        default=1.0,
        help="Max grad norm for adapter params (<=0 disables).",
    )
    parser.add_argument(
        "--gradient-checkpointing",
        action="store_true",
        help="Enable gradient checkpointing in the frozen transformer (saves memory, slower).",
    )
    parser.add_argument(
        "--resume",
        default=None,
        help="Optional path to an existing adapter checkpoint (.pth) to resume from.",
    )
    parser.add_argument(
        "--save-every",
        type=int,
        default=500,
        help="Checkpoint every N steps (<=0 disables periodic saves).",
    )
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[2]
    cache_dir = Path(args.cache_dir).expanduser() if args.cache_dir else None
    data_dir = (root / args.data_dir).resolve() if not Path(args.data_dir).is_absolute() else Path(args.data_dir)
    identity_file = None
    if args.identity_file:
        identity_file = (root / args.identity_file).resolve() if not Path(args.identity_file).is_absolute() else Path(args.identity_file)
    out_path = (root / args.output).resolve() if not Path(args.output).is_absolute() else Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if args.device == "auto":
        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")
    else:
        device = torch.device(args.device)

    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    if device.type == "mps" and not (hasattr(torch.backends, "mps") and torch.backends.mps.is_available()):
        raise SystemExit("MPS requested but not available; rerun with --device cpu/cuda")

    if args.dtype == "fp32":
        dtype = torch.float32
    elif args.dtype == "bf16":
        dtype = torch.bfloat16
    else:
        dtype = torch.float16

    if device.type in {"cpu", "mps"} and dtype == torch.bfloat16:
        # bfloat16 support is spotty outside CUDA; fall back to fp16/fp32.
        dtype = torch.float16 if device.type == "mps" else torch.float32

    attention_backend = args.attention_backend
    if attention_backend is None and device.type == "cuda":
        # Default to a memory-efficient SDPA kernel for training.
        attention_backend = "_native_flash"

    cfg = TrainConfig(
        repo=args.repo,
        cache_dir=cache_dir,
        data_dir=None if args.hf_dataset else data_dir,
        identity_file=identity_file,
        hf_dataset=args.hf_dataset,
        hf_config=args.hf_config,
        hf_split=args.hf_split,
        output=out_path,
        steps=args.steps,
        batch_size=args.batch_size,
        height=args.height,
        width=args.width,
        prompt_len=args.prompt_len,
        prompt_template=args.prompt_template,
        token_count=args.token_count,
        lr=args.lr,
        device=device,
        dtype=dtype,
        attention_backend=attention_backend,
        prompt_mode=args.prompt_mode or ("celeba_attrs" if args.hf_dataset else "zeros"),
        prompt_cache_size=args.prompt_cache_size,
        text_encoder_on_cpu=args.text_encoder_on_cpu,
        token_scale=args.token_scale,
        grad_clip=args.grad_clip,
        gradient_checkpointing=args.gradient_checkpointing,
        resume=(Path(args.resume).expanduser().resolve() if args.resume else None),
        save_every=int(args.save_every),
    )

    ds = None
    valid_ids: list[int] | None = None
    id_to_items = None
    image_paths: list[Path] | None = None
    if cfg.hf_dataset:
        ds, valid_ids, id_to_items = _load_hf_dataset_index(
            cfg.hf_dataset,
            cfg.hf_config,
            cfg.hf_split,
            cfg.cache_dir,
        )
    else:
        image_paths = _iter_image_paths(cfg.data_dir or Path("."))
        if not image_paths:
            raise SystemExit(f"No images found under {cfg.data_dir}")
        if cfg.identity_file:
            valid_ids, id_to_items = _parse_identity_file(cfg.identity_file, cfg.data_dir or Path("."))

    if ds is not None:
        print("[train] data=hf")
    else:
        print(f"[train] data=dir images={len(image_paths or [])} dir={cfg.data_dir}")
    print(f"[train] repo={cfg.repo}")
    print(f"[train] cache_dir={cfg.cache_dir}")
    print(f"[train] device={cfg.device}")
    print(f"[train] dtype={cfg.dtype}")
    print(f"[train] attention_backend={cfg.attention_backend}")
    print(f"[train] prompt_mode={cfg.prompt_mode}")
    print(f"[train] prompt_template={cfg.prompt_template}")
    print(f"[train] prompt_cache_size={cfg.prompt_cache_size}")
    print(f"[train] text_encoder_on_cpu={cfg.text_encoder_on_cpu}")
    print(f"[train] token_scale={cfg.token_scale}")
    print(f"[train] grad_clip={cfg.grad_clip}")
    print(f"[train] gradient_checkpointing={cfg.gradient_checkpointing}")
    print(f"[train] resume={cfg.resume}")
    print(f"[train] save_every={cfg.save_every}")
    print(f"[train] out={cfg.output}")

    # Load base models (frozen).
    # Note: transformer weights are very large; expect significant RAM/VRAM use.
    transformer = ZImageTransformer2DModel.from_pretrained(
        cfg.repo,
        subfolder="transformer",
        cache_dir=str(cfg.cache_dir) if cfg.cache_dir else None,
        torch_dtype=cfg.dtype,
        low_cpu_mem_usage=True,
    ).to(cfg.device)
    transformer.eval()
    if cfg.gradient_checkpointing:
        transformer.enable_gradient_checkpointing()
    for p in transformer.parameters():
        p.requires_grad_(False)

    vae = AutoencoderKL.from_pretrained(
        cfg.repo,
        subfolder="vae",
        cache_dir=str(cfg.cache_dir) if cfg.cache_dir else None,
        torch_dtype=cfg.dtype,
        low_cpu_mem_usage=True,
    ).to(cfg.device)
    vae.eval()
    for p in vae.parameters():
        p.requires_grad_(False)

    prompt_cache = _PromptEmbedCache(cfg.prompt_cache_size)
    text_encoder = None
    if cfg.prompt_mode != "zeros":
        text_encoder = _ZImageTextEncoder(
            repo=cfg.repo,
            cache_dir=cfg.cache_dir,
            prompt_len=cfg.prompt_len,
            prompt_template=cfg.prompt_template,
            device=cfg.device,
            dtype=cfg.dtype,
            on_cpu=cfg.text_encoder_on_cpu,
        )
        print(f"[train] text_encoder_device={text_encoder.to_model_device()}")

    # Load face embedder (frozen).
    mobilefacenet_repo = root / "build-tools" / "faceid" / "MobileFaceNet_Tutorial_Pytorch"
    face_model = _load_mobilefacenet(mobilefacenet_repo, cfg.device)

    # Image preprocessing
    to_face = transforms.Compose(
        [
            transforms.CenterCrop(min(cfg.height, cfg.width)),
            transforms.Resize((112, 112), interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.ToTensor(),  # [0,1], RGB
        ]
    )
    to_vae = transforms.Compose(
        [
            transforms.Resize((cfg.height, cfg.width), interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.CenterCrop((cfg.height, cfg.width)),
            transforms.ToTensor(),  # [0,1]
        ]
    )

    adapter = ZImageFaceIDAdapter(token_count=cfg.token_count, token_scale=cfg.token_scale).to(cfg.device)
    opt = torch.optim.AdamW(adapter.parameters(), lr=cfg.lr)

    base_prompt = torch.zeros((cfg.prompt_len, 2560), device=cfg.device, dtype=transformer.dtype)

    start_step = 0
    if cfg.resume is not None:
        ckpt = torch.load(str(cfg.resume), map_location="cpu")
        ckpt_token_count = int(ckpt.get("token_count", cfg.token_count))
        if ckpt_token_count != cfg.token_count:
            raise SystemExit(f"resume token_count mismatch: expected {cfg.token_count}, got {ckpt_token_count}")
        state = ckpt.get("state_dict")
        if not isinstance(state, dict):
            raise SystemExit("resume checkpoint missing state_dict")
        adapter.load_state_dict(state, strict=True)
        opt_state = ckpt.get("optimizer")
        if isinstance(opt_state, dict):
            try:
                opt.load_state_dict(opt_state)
            except Exception:
                print("[train] resume: optimizer state could not be loaded; continuing with fresh optimizer.")
        start_step = int(ckpt.get("step", 0))
        if start_step < 0:
            start_step = 0
        print(f"[train] resumed from {cfg.resume} at step={start_step}")

    def _save_checkpoint(path: Path, step_idx: int) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(path.suffix + ".tmp")
        torch.save(
            {
                "step": int(step_idx),
                "token_count": cfg.token_count,
                "token_scale": cfg.token_scale,
                "prompt_len": cfg.prompt_len,
                "state_dict": adapter.state_dict(),
                "optimizer": opt.state_dict(),
            },
            str(tmp),
        )
        tmp.replace(path)

    try:
        from diffusers.models.attention_dispatch import attention_backend as attention_backend_ctx  # type: ignore
    except Exception as e:  # pragma: no cover
        raise SystemExit("diffusers attention backend support is required; please update diffusers.") from e

    run_start = time.perf_counter()
    ema_step_s: float | None = None

    attn_backend_name = cfg.attention_backend

    for step in range(start_step, cfg.steps):
        step_start = time.perf_counter()
        opt.zero_grad(set_to_none=True)

        images_rgb = []
        faces_rgb = []
        prompts: list[str] = []
        for _ in range(cfg.batch_size):
            if ds is not None and valid_ids is not None and id_to_items is not None:
                celeb_id = random.choice(valid_ids)
                i_ref, i_tgt = random.sample(id_to_items[celeb_id], k=2)
                ref_row = ds[i_ref]
                tgt_row = ds[i_tgt]
                ref_img = ref_row["image"].convert("RGB")
                tgt_img = tgt_row["image"].convert("RGB")
                if cfg.prompt_mode == "celeba_attrs":
                    prompts.append(_build_celeba_prompt(tgt_row))
            elif valid_ids is not None and id_to_items is not None:
                ident = random.choice(valid_ids)
                p_ref, p_tgt = random.sample(id_to_items[ident], k=2)
                with Image.open(p_ref) as im:
                    ref_img = im.convert("RGB")
                with Image.open(p_tgt) as im:
                    tgt_img = im.convert("RGB")
            else:
                assert image_paths is not None
                p = random.choice(image_paths)
                with Image.open(p) as im:
                    ref_img = im.convert("RGB")
                tgt_img = ref_img

            images_rgb.append(to_vae(tgt_img))
            faces_rgb.append(to_face(ref_img))

        # VAE input expects [-1,1].
        images = torch.stack(images_rgb, dim=0).to(cfg.device, dtype=vae.dtype) * 2.0 - 1.0
        with torch.no_grad():
            posterior = vae.encode(images).latent_dist
            latents = posterior.sample()
            latents = (latents - vae.config.shift_factor) * vae.config.scaling_factor

        # Face embedding preproc: BGR in [-1,1] at 112×112.
        faces = torch.stack(faces_rgb, dim=0).to(cfg.device, dtype=torch.float32)
        faces = faces[:, [2, 1, 0], :, :]  # RGB -> BGR
        faces = faces * 2.0 - 1.0

        with torch.no_grad():
            face_emb = face_model(faces).float()
            face_emb = l2_normalize(face_emb)

        tokens = adapter(face_emb).to(dtype=transformer.dtype)  # [B, T, 2560]

        prompt_embeds_list = []
        for i in range(cfg.batch_size):
            if cfg.prompt_mode == "zeros" or text_encoder is None:
                prompt_embeds = base_prompt
            else:
                prompt = prompts[i] if i < len(prompts) else ""
                cache_key = hashlib.sha1(prompt.encode("utf-8")).hexdigest()
                cached = prompt_cache.get(cache_key)
                if cached is None:
                    cached = text_encoder.encode(prompt)
                    prompt_cache.put(cache_key, cached)
                prompt_embeds = text_encoder.to_train_device(cached, dtype=transformer.dtype)
            prompt_embeds_list.append(torch.cat([prompt_embeds, tokens[i]], dim=0))

        # Flow matching:
        #   x_t = (1 - sigma) * x0 + sigma * x1,   x1 ~ N(0,1),   v = x1 - x0
        noise = torch.randn_like(latents)
        sigma = torch.rand((cfg.batch_size,), device=cfg.device, dtype=torch.float32)
        while sigma.ndim < latents.ndim:
            sigma = sigma.unsqueeze(-1)
        noisy_latents = (1.0 - sigma) * latents + sigma * noise
        target_v = (noise.float() - latents.float()).to(device=cfg.device)

        t = (1.0 - sigma.flatten()).to(torch.float32)  # normalized time [0,1]

        latent_model_input_list = list(noisy_latents.to(dtype=transformer.dtype).unsqueeze(2).unbind(dim=0))

        # Keep the attention backend context active for both forward and backward recomputation
        # when gradient checkpointing is enabled.
        with (attention_backend_ctx(attn_backend_name) if attn_backend_name else nullcontext()):
            out_list = transformer(latent_model_input_list, t, prompt_embeds_list, return_dict=False)[0]
            out = torch.stack(out_list, dim=0).squeeze(2).float()

            # Pipeline negates model output before scheduler step.
            pred_v = -out  # keep fp32 for stable loss
            loss = torch.mean((pred_v - target_v) ** 2)
            if not torch.isfinite(loss).item():
                print("[train] ERROR: non-finite loss; aborting.")
                print(_tensor_stats("latents", latents))
                print(_tensor_stats("noise", noise))
                print(_tensor_stats("noisy_latents", noisy_latents))
                print(_tensor_stats("tokens", tokens))
                print(_tensor_stats("out", out))
                print(_tensor_stats("target_v", target_v))
                raise SystemExit(2)
            loss.backward()
        if cfg.grad_clip and cfg.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(adapter.parameters(), cfg.grad_clip)
        opt.step()

        step_s = time.perf_counter() - step_start
        if ema_step_s is None:
            ema_step_s = step_s
        else:
            # Smooth per-step time (helps ETA not jump around).
            ema_step_s = 0.95 * ema_step_s + 0.05 * step_s

        if (step + 1) % 5 == 0 or step == 0:
            eta_s = (cfg.steps - (step + 1)) * (ema_step_s or step_s)
            elapsed_s = time.perf_counter() - run_start
            print(
                f"[train] step {step+1}/{cfg.steps} "
                f"loss={loss.item():.6f} "
                f"dt={step_s:.3f}s "
                f"avg_dt={((ema_step_s or step_s)):.3f}s "
                f"elapsed={elapsed_s/60.0:.1f}m "
                f"eta={eta_s/3600.0:.1f}h"
            )
        if cfg.save_every > 0 and (step + 1) % cfg.save_every == 0:
            ckpt_path = cfg.output.with_suffix(f".step{step+1}.pth")
            _save_checkpoint(ckpt_path, step + 1)
            print(f"[train] checkpoint {ckpt_path}")

    _save_checkpoint(cfg.output, cfg.steps)
    print(f"[train] saved {cfg.output}")


if __name__ == "__main__":
    main()
