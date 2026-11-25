#!/usr/bin/env python3
"""Model-agnostic image generation CLI for BlackInk / DeepSeek Harness."""

from __future__ import annotations

import argparse
import fcntl
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

DEFAULT_CONFIG_PATH = Path.home() / ".config" / "ingine" / "image-gen.json"
LOCK_PATH = Path("/tmp/blackink-gen-image.lock")

DEFAULT_CONFIG: dict[str, Any] = {
    "backend": "sana-sprint-0.6b",
    "model_id": "Efficient-Large-Model/Sana_Sprint_0.6B_1024px_diffusers",
    "dtype": "bfloat16",
    "threads": 8,
    "default_size": "512x512",
    "default_steps": 2,
    "default_guidance": 4.5,
}


def load_config() -> dict[str, Any]:
    cfg = dict(DEFAULT_CONFIG)
    cfg_path = Path(os.environ.get("INGINE_IMAGE_CONFIG", str(DEFAULT_CONFIG_PATH)))
    if cfg_path.is_file():
        with cfg_path.open("r", encoding="utf-8") as fh:
            user_cfg = json.load(fh)
        if isinstance(user_cfg, dict):
            cfg.update(user_cfg)
    if "INGINE_IMAGE_BACKEND" in os.environ:
        cfg["backend"] = os.environ["INGINE_IMAGE_BACKEND"]
    if "INGINE_IMAGE_THREADS" in os.environ:
        cfg["threads"] = int(os.environ["INGINE_IMAGE_THREADS"])
    return cfg


def parse_size(raw: str) -> tuple[int, int]:
    m = re.fullmatch(r"(\d{2,4})[xX](\d{2,4})", raw.strip())
    if not m:
        raise ValueError(f"Invalid size '{raw}', expected WxH such as 512x512 or 768x512")
    w, h = int(m.group(1)), int(m.group(2))
    if w < 128 or h < 128 or w > 2048 or h > 2048 or (w % 32) != 0 or (h % 32) != 0:
        raise ValueError("Width and height must be multiples of 32 between 128 and 2048")
    return w, h


def _run_sana_sprint(
    cfg: dict[str, Any],
    prompt: str,
    width: int,
    height: int,
    steps: int,
    guidance: float,
    seed: int | None,
    out_path: Path,
) -> None:
    import torch
    from diffusers import SanaSprintPipeline

    threads = max(1, int(cfg.get("threads", 8)))
    torch.set_num_threads(threads)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass

    dtype_name = str(cfg.get("dtype", "bfloat16")).lower()
    dtype = {
        "float32": torch.float32,
        "fp32": torch.float32,
        "float16": torch.float16,
        "fp16": torch.float16,
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
    }.get(dtype_name, torch.bfloat16)

    model_id = str(cfg.get("model_id") or DEFAULT_CONFIG["model_id"])
    pipe = SanaSprintPipeline.from_pretrained(model_id, torch_dtype=dtype)
    pipe.to("cpu")
    pipe.set_progress_bar_config(disable=True)

    generator = torch.Generator(device="cpu").manual_seed(seed) if seed is not None else None
    result = pipe(
        prompt=prompt,
        width=width,
        height=height,
        num_inference_steps=steps,
        guidance_scale=guidance,
        generator=generator,
    )
    image = result.images[0]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(out_path)


def _run_sd_cli(
    cfg: dict[str, Any],
    prompt: str,
    width: int,
    height: int,
    steps: int,
    guidance: float,
    seed: int | None,
    out_path: Path,
) -> None:
    sd_bin = str(cfg.get("sd_cli_bin", "sd-cli"))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        sd_bin,
        "-p",
        prompt,
        "-W",
        str(width),
        "-H",
        str(height),
        "--steps",
        str(steps),
        "--cfg-scale",
        str(guidance),
        "-t",
        str(int(cfg.get("threads", 8))),
        "-o",
        str(out_path),
    ]
    if seed is not None:
        cmd.extend(["-s", str(seed)])
    for flag, key in [
        ("--diffusion-model", "diffusion_model"),
        ("-m", "model_path"),
        ("--vae", "vae_path"),
        ("--clip_l", "clip_l"),
        ("--clip_g", "clip_g"),
        ("--t5xxl", "t5xxl"),
        ("--llm", "llm_path"),
        ("--sampling-method", "sampler"),
    ]:
        val = cfg.get(key)
        if val:
            cmd.extend([flag, str(val)])
    for extra in cfg.get("extra_args", []):
        cmd.append(str(extra))
    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if proc.returncode != 0 or not out_path.is_file():
        raise RuntimeError(f"Image generation failed with exit code {proc.returncode}")


def main() -> int:
    parser = argparse.ArgumentParser(
        prog="gen-image",
        description="Generate an image asset from a text prompt.",
    )
    parser.add_argument("--prompt", required=True, help="Detailed English prompt describing the image")
    parser.add_argument("--out", required=True, help="Output file path (.png, .jpg, .webp)")
    parser.add_argument("--size", default=None, help="Output dimensions WxH (e.g. 512x512, 768x512)")
    parser.add_argument("--steps", type=int, default=None, help="Optional refinement steps")
    parser.add_argument("--guidance", type=float, default=None, help="Optional guidance scale")
    parser.add_argument("--seed", type=int, default=None, help="Optional random seed")
    parser.add_argument("--quiet", action="store_true", help="Suppress JSON output")
    args = parser.parse_args()

    cfg = load_config()
    width, height = parse_size(args.size or str(cfg.get("default_size", "512x512")))
    steps = args.steps if args.steps is not None else int(cfg.get("default_steps", 2))
    guidance = args.guidance if args.guidance is not None else float(cfg.get("default_guidance", 4.5))
    out_path = Path(args.out).expanduser().resolve()

    t0 = time.monotonic()
    LOCK_PATH.parent.mkdir(parents=True, exist_ok=True)
    with LOCK_PATH.open("w") as lock_fh:
        fcntl.flock(lock_fh.fileno(), fcntl.LOCK_EX)
        backend = str(cfg.get("backend", "sana-sprint-0.6b"))
        if backend in ("sana-sprint-0.6b", "sana-sprint-1.6b", "sana-sprint"):
            _run_sana_sprint(cfg, args.prompt, width, height, steps, guidance, args.seed, out_path)
        elif backend == "sd-cli":
            _run_sd_cli(cfg, args.prompt, width, height, steps, guidance, args.seed, out_path)
        else:
            raise RuntimeError("Configured image generation backend is unavailable")

    elapsed = round(time.monotonic() - t0, 2)
    if not args.quiet:
        print(
            json.dumps(
                {
                    "ok": True,
                    "path": str(out_path),
                    "width": width,
                    "height": height,
                    "elapsed_sec": elapsed,
                }
            )
        )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(json.dumps({"ok": False, "error": str(exc)}), file=sys.stderr)
        raise SystemExit(1)
