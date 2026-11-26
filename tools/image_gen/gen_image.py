#!/usr/bin/env python3
"""Model-agnostic image generation CLI for BlackInk / DeepSeek Harness."""

from __future__ import annotations

import argparse
import fcntl
import json
import os
import re
import signal
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

os.environ.setdefault("NUMBA_CACHE_DIR", "/tmp/blackink-numba-cache")

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


def _find_ingine_serve() -> tuple[int, list[str], int] | None:
    proc_root = Path("/proc")
    if not proc_root.is_dir():
        return None
    for entry in proc_root.iterdir():
        if not entry.name.isdigit():
            continue
        try:
            raw = (entry / "cmdline").read_bytes()
        except OSError:
            continue
        if not raw:
            continue
        argv = [part.decode("utf-8", errors="replace") for part in raw.split(b"\x00") if part]
        if not argv:
            continue
        if Path(argv[0]).name == "ingine-serve":
            port = 8090
            for idx, arg in enumerate(argv[:-1]):
                if arg == "--port" and argv[idx + 1].isdigit():
                    port = int(argv[idx + 1])
                    break
            return int(entry.name), argv, port
    return None


def _stop_ingine_serve(pid: int, timeout_sec: float = 15.0) -> None:
    os.kill(pid, signal.SIGTERM)
    deadline = time.monotonic() + timeout_sec
    while time.monotonic() < deadline:
        if not Path(f"/proc/{pid}").exists():
            time.sleep(0.5)
            return
        time.sleep(0.2)
    os.kill(pid, signal.SIGKILL)
    time.sleep(0.8)


def _launch_ingine_serve(argv: list[str]) -> None:
    log_path = Path("/tmp/ingine-serve.log")
    with log_path.open("ab") as log_fh:
        subprocess.Popen(
            argv,
            stdin=subprocess.DEVNULL,
            stdout=log_fh,
            stderr=subprocess.STDOUT,
            start_new_session=True,
            close_fds=True,
        )


def _wait_ingine_health(port: int, timeout_sec: float = 90.0) -> None:
    deadline = time.monotonic() + timeout_sec
    health_url = f"http://127.0.0.1:{port}/health"
    while time.monotonic() < deadline:
        try:
            with urllib.request.urlopen(health_url, timeout=2.0) as resp:
                if resp.status == 200:
                    return
        except Exception:
            pass
        time.sleep(0.5)
    raise RuntimeError("Timed out waiting for language service to recover after GPU handoff")


def _run_sd_cli(
    cfg: dict[str, Any],
    prompt: str,
    negative_prompt: str,
    width: int,
    height: int,
    steps: int,
    guidance: float,
    seed: int | None,
    out_path: Path,
    use_cuda: bool = False,
) -> None:
    sd_bin = str(
        (cfg.get("sd_cli_cuda_bin") if use_cuda else None)
        or cfg.get("sd_cli_bin", "sd-cli")
    )
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
    if negative_prompt:
        cmd.extend(["-n", negative_prompt])
    if seed is not None:
        cmd.extend(["-s", str(seed)])
    for flag, key in [
        ("--diffusion-model", "diffusion_model"),
        ("-m", "model_path"),
        ("--vae", "vae_path"),
        ("--taesd", "taesd_path"),
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
    env = os.environ.copy()
    ld_extra = cfg.get("ld_library_path")
    if use_cuda and ld_extra:
        cur_ld = env.get("LD_LIBRARY_PATH", "")
        env["LD_LIBRARY_PATH"] = f"{ld_extra}:{cur_ld}" if cur_ld else str(ld_extra)
    proc = subprocess.run(cmd, env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if proc.returncode != 0 or not out_path.is_file():
        raise RuntimeError(f"Image generation failed with exit code {proc.returncode}")


def _remove_background(image_path: Path) -> None:
    Path(os.environ["NUMBA_CACHE_DIR"]).mkdir(parents=True, exist_ok=True)
    from PIL import Image
    from rembg import new_session, remove

    with Image.open(image_path) as img:
        rgba = img.convert("RGBA")
    session = new_session("u2net")
    cutout = remove(rgba, session=session, post_process_mask=True)
    cutout.save(image_path)


def _prepare_prompts(prompt: str, negative_prompt: str, transparent: bool) -> tuple[str, str]:
    eff_prompt = prompt.strip()
    eff_neg = negative_prompt
    if transparent:
        eff_prompt = re.sub(
            r"\b(?:with\s+)?(?:a\s+)?(?:transparent|no|clear|blank|isolated|plain|solid|white)\s+background\b",
            "",
            eff_prompt,
            flags=re.IGNORECASE,
        ).strip(" ,.")
        eff_prompt = (
            f"{eff_prompt}, centered single subject isolated on a plain solid white studio background, "
            "no scenery, no shadows, clean sharp edges"
        )
        extra_neg = "scenery, landscape, trees, floor, ground, room, clutter, busy background, drop shadow"
        eff_neg = f"{eff_neg}, {extra_neg}".strip(" ,")
    return eff_prompt, eff_neg


def generate_batch(items: list[dict[str, Any]]) -> dict[str, Any]:
    if not items:
        raise ValueError("Batch requires at least one image item")

    cfg = load_config()
    backend = str(cfg.get("backend", "sana-sprint-0.6b"))
    gpu_handoff = bool(cfg.get("gpu_handoff", False))
    base_neg = str(cfg.get("negative_prompt", ""))

    prepared: list[dict[str, Any]] = []
    for idx, item in enumerate(items):
        if not isinstance(item, dict) or "prompt" not in item or "out" not in item:
            raise ValueError(f"Batch item {idx} must include 'prompt' and 'out'")
        transparent = bool(item.get("transparent", False))
        out_path = Path(str(item["out"])).expanduser().resolve()
        if transparent and out_path.suffix.lower() not in (".png", ".webp"):
            raise ValueError("--transparent requires an output format with alpha channel (.png or .webp)")
        width, height = parse_size(str(item.get("size") or cfg.get("default_size", "512x512")))
        eff_steps = int(item["steps"]) if item.get("steps") is not None else int(cfg.get("default_steps", 4))
        eff_guidance = (
            float(item["guidance"]) if item.get("guidance") is not None else float(cfg.get("default_guidance", 1.0))
        )
        seed = int(item["seed"]) if item.get("seed") is not None else None
        eff_prompt, eff_neg = _prepare_prompts(str(item["prompt"]), base_neg, transparent)
        prepared.append(
            {
                "prompt": eff_prompt,
                "negative_prompt": eff_neg,
                "out_path": out_path,
                "width": width,
                "height": height,
                "steps": eff_steps,
                "guidance": eff_guidance,
                "seed": seed,
                "transparent": transparent,
            }
        )

    t0 = time.monotonic()
    LOCK_PATH.parent.mkdir(parents=True, exist_ok=True)
    with LOCK_PATH.open("w") as lock_fh:
        fcntl.flock(lock_fh.fileno(), fcntl.LOCK_EX)
        if backend in ("sana-sprint-0.6b", "sana-sprint-1.6b", "sana-sprint"):
            for job in prepared:
                _run_sana_sprint(
                    cfg, job["prompt"], job["width"], job["height"], job["steps"], job["guidance"], job["seed"], job["out_path"]
                )
                if job["transparent"]:
                    _remove_background(job["out_path"])
        elif backend == "sd-cli":
            saved_ingine = _find_ingine_serve() if gpu_handoff else None
            ingine_launched = False
            try:
                if saved_ingine is not None:
                    _stop_ingine_serve(saved_ingine[0])
                for job in prepared:
                    _run_sd_cli(
                        cfg,
                        job["prompt"],
                        job["negative_prompt"],
                        job["width"],
                        job["height"],
                        job["steps"],
                        job["guidance"],
                        job["seed"],
                        job["out_path"],
                        use_cuda=gpu_handoff,
                    )
            finally:
                if saved_ingine is not None:
                    _, argv, port = saved_ingine
                    _launch_ingine_serve(argv)
                    ingine_launched = True
            # Overlap CPU background removal with GPU language-service reload.
            for job in prepared:
                if job["transparent"] and job["out_path"].is_file():
                    _remove_background(job["out_path"])
            if ingine_launched and saved_ingine is not None:
                _wait_ingine_health(saved_ingine[2])
        else:
            raise RuntimeError("Configured image generation backend is unavailable")

    elapsed = round(time.monotonic() - t0, 2)
    results = [
        {
            "ok": True,
            "path": str(job["out_path"]),
            "width": job["width"],
            "height": job["height"],
            "transparent": job["transparent"],
        }
        for job in prepared
    ]
    return {
        "ok": True,
        "count": len(results),
        "images": results,
        "elapsed_sec": elapsed,
    }


def generate_image(
    *,
    prompt: str,
    out: str,
    size: str | None = None,
    steps: int | None = None,
    guidance: float | None = None,
    seed: int | None = None,
    transparent: bool = False,
) -> dict[str, Any]:
    res = generate_batch(
        [
            {
                "prompt": prompt,
                "out": out,
                "size": size,
                "steps": steps,
                "guidance": guidance,
                "seed": seed,
                "transparent": transparent,
            }
        ]
    )
    single = dict(res["images"][0])
    single["elapsed_sec"] = res["elapsed_sec"]
    return single


def _try_host_daemon(payload_dict: dict[str, Any]) -> dict[str, Any] | None:
    if os.environ.get("INGINE_IMAGE_NO_DAEMON") == "1":
        return None
    port = int(os.environ.get("INGINE_IMAGE_PORT", "8091"))
    url = f"http://127.0.0.1:{port}/generate"
    payload = json.dumps(payload_dict).encode("utf-8")
    req = urllib.request.Request(url, data=payload, headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=600.0) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except urllib.error.URLError:
        return None


def main() -> int:
    parser = argparse.ArgumentParser(
        prog="gen-image",
        description="Generate one image or a batch of image assets from text prompts.",
    )
    parser.add_argument("--prompt", help="Detailed English prompt describing the image")
    parser.add_argument("--out", help="Output file path (.png, .jpg, .webp)")
    parser.add_argument("--batch", help="Path to JSON array of {prompt, out, ...} items, or '-' for stdin")
    parser.add_argument("--size", default=None, help="Output dimensions WxH (e.g. 512x512, 768x512)")
    parser.add_argument("--steps", type=int, default=None, help="Optional refinement steps")
    parser.add_argument("--guidance", type=float, default=None, help="Optional guidance scale")
    parser.add_argument("--seed", type=int, default=None, help="Optional random seed")
    parser.add_argument(
        "--transparent",
        action="store_true",
        help="Remove background and save with alpha transparency (.png or .webp)",
    )
    parser.add_argument("--quiet", action="store_true", help="Suppress JSON output")
    args = parser.parse_args()

    if args.batch:
        raw_batch = sys.stdin.read() if args.batch == "-" else Path(args.batch).read_text(encoding="utf-8")
        items = json.loads(raw_batch)
        if not isinstance(items, list):
            raise ValueError("--batch expects a JSON array of objects")
        for item in items:
            if isinstance(item, dict) and "out" in item:
                item["out"] = str(Path(str(item["out"])).expanduser().resolve())
        result = _try_host_daemon({"items": items})
        if result is None:
            result = generate_batch(items)
    else:
        if not args.prompt or not args.out:
            parser.error("--prompt and --out are required unless --batch is used")
        payload_dict = {
            "prompt": args.prompt,
            "out": str(Path(args.out).expanduser().resolve()),
            "size": args.size,
            "steps": args.steps,
            "guidance": args.guidance,
            "seed": args.seed,
            "transparent": bool(args.transparent),
        }
        result = _try_host_daemon(payload_dict)
        if result is None:
            result = generate_image(
                prompt=args.prompt,
                out=args.out,
                size=args.size,
                steps=args.steps,
                guidance=args.guidance,
                seed=args.seed,
                transparent=bool(args.transparent),
            )
    if not result.get("ok"):
        raise RuntimeError(str(result.get("error", "Image generation failed")))

    if not args.quiet:
        print(json.dumps(result))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(json.dumps({"ok": False, "error": str(exc)}), file=sys.stderr)
        raise SystemExit(1)
