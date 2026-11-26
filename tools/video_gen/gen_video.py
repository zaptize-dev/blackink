#!/usr/bin/env python3
"""Video generation CLI (Wan 2.2 TI2V 5B via sd-cli) with GPU handoff for BlackInk."""

from __future__ import annotations

import argparse
import fcntl
import json
import os
import re
import shutil
import signal
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

DEFAULT_CONFIG_PATH = Path.home() / ".config" / "ingine" / "video-gen.json"
LOCK_PATH = Path("/tmp/blackink-gen-image.lock")

DEFAULT_CONFIG: dict[str, Any] = {
    "gpu_handoff": True,
    "sd_cli_cuda_bin": "/home/zaptize/blackink/.tools/stable-diffusion.cpp/build-cuda/bin/sd-cli",
    "ld_library_path": "/home/zaptize/blackink/.cuda-toolkit/nvidia/cu13/lib",
    "diffusion_model": "/home/zaptize/blackink/models/video/wan22-ti2v-5b/Wan2.2-TI2V-5B-Q6_K.gguf",
    "t5xxl": "/home/zaptize/blackink/models/video/wan22-ti2v-5b/umt5-xxl-encoder-Q4_K_M.gguf",
    "vae_path": "/home/zaptize/blackink/models/video/wan22-ti2v-5b/wan2.2_vae.safetensors",
    "sampler": "euler",
    "threads": 8,
    "default_size": "832x480",
    "default_frames": 81,
    "default_fps": 16,
    "default_steps": 20,
    "default_guidance": 6.0,
    "default_flow_shift": 5.0,
    "negative_prompt": "blurry, distorted, jitter, flickering, low quality, text, watermark",
    "extra_args": [
        "--flash-attn",
        "off,diffusion=on",
        "--vae-tiling",
        "--temporal-tiling",
        "--extra-tiling-args",
        "temporal_decode_frames=1",
    ],
}


def load_config() -> dict[str, Any]:
    cfg = dict(DEFAULT_CONFIG)
    cfg_path = Path(os.environ.get("INGINE_VIDEO_CONFIG", str(DEFAULT_CONFIG_PATH)))
    if cfg_path.is_file():
        with cfg_path.open("r", encoding="utf-8") as fh:
            user_cfg = json.load(fh)
        if isinstance(user_cfg, dict):
            cfg.update(user_cfg)
    return cfg


def parse_size(raw: str) -> tuple[int, int]:
    m = re.fullmatch(r"(\d{2,4})[xX](\d{2,4})", raw.strip())
    if not m:
        raise ValueError(f"Invalid size '{raw}', expected WxH such as 832x480")
    w, h = int(m.group(1)), int(m.group(2))
    if w < 128 or h < 128 or w > 1920 or h > 1920 or (w % 16) != 0 or (h % 16) != 0:
        raise ValueError("Width and height must be multiples of 16 between 128 and 1920")
    return w, h


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
        if argv and Path(argv[0]).name == "ingine-serve":
            port = 8090
            for idx, arg in enumerate(argv[:-1]):
                if arg == "--port" and argv[idx + 1].isdigit():
                    port = int(argv[idx + 1])
                    break
            return int(entry.name), argv, port
    return None


def _wait_ingine_idle(port: int, timeout_sec: float = 30.0) -> None:
    deadline = time.monotonic() + timeout_sec
    health_url = f"http://127.0.0.1:{port}/health"
    while time.monotonic() < deadline:
        try:
            with urllib.request.urlopen(health_url, timeout=1.5) as resp:
                if resp.status == 200:
                    data = json.loads(resp.read().decode("utf-8"))
                    if not data.get("running", False) and int(data.get("waiting", 0)) == 0:
                        return
        except Exception:
            return
        time.sleep(0.25)


def _stop_ingine_serve(pid: int, port: int, timeout_sec: float = 15.0) -> None:
    _wait_ingine_idle(port)
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
    with Path("/tmp/ingine-serve.log").open("ab") as log_fh:
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
    raise RuntimeError("Timed out waiting for ingine-serve to recover after video generation")


def _postprocess_video(raw_webm: Path, out_path: Path, loop: bool) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if out_path.suffix.lower() == ".webm" and not loop:
        shutil.move(str(raw_webm), str(out_path))
        return

    cmd = ["ffmpeg", "-y", "-i", str(raw_webm)]
    if loop:
        cmd.extend(["-filter_complex", "[0:v]split[f][r];[r]reverse[rev];[f][rev]concat=n=2:v=1:a=0"])
    if out_path.suffix.lower() == ".mp4":
        cmd.extend(["-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "18", "-movflags", "+faststart"])
    cmd.append(str(out_path))
    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if proc.returncode != 0 or not out_path.is_file():
        raise RuntimeError(f"ffmpeg postprocess failed: {proc.stderr[-300:]}")


def generate_video(
    *,
    prompt: str,
    out: str,
    init_img: str | None = None,
    end_img: str | None = None,
    size: str | None = None,
    frames: int | None = None,
    fps: int | None = None,
    steps: int | None = None,
    guidance: float | None = None,
    flow_shift: float | None = None,
    seed: int | None = None,
    negative_prompt: str | None = None,
    loop: bool = False,
) -> dict[str, Any]:
    cfg = load_config()
    width, height = parse_size(str(size or cfg.get("default_size", "832x480")))
    eff_frames = int(frames if frames is not None else cfg.get("default_frames", 81))
    eff_fps = int(fps if fps is not None else cfg.get("default_fps", 16))
    eff_steps = int(steps if steps is not None else cfg.get("default_steps", 20))
    eff_guidance = float(guidance if guidance is not None else cfg.get("default_guidance", 6.0))
    eff_shift = float(flow_shift if flow_shift is not None else cfg.get("default_flow_shift", 5.0))
    eff_neg = negative_prompt if negative_prompt is not None else str(cfg.get("negative_prompt", ""))
    out_path = Path(out).expanduser().resolve()

    sd_bin = str(cfg.get("sd_cli_cuda_bin") or "sd-cli")
    gpu_handoff = bool(cfg.get("gpu_handoff", True))

    t0 = time.monotonic()
    LOCK_PATH.parent.mkdir(parents=True, exist_ok=True)
    with LOCK_PATH.open("w") as lock_fh, tempfile.TemporaryDirectory(prefix="blackink_vid_") as tmpdir:
        fcntl.flock(lock_fh.fileno(), fcntl.LOCK_EX)
        raw_webm = Path(tmpdir) / "raw.webm"
        cmd = [
            sd_bin,
            "-M",
            "vid_gen",
            "--diffusion-model",
            str(cfg["diffusion_model"]),
            "--t5xxl",
            str(cfg["t5xxl"]),
            "--vae",
            str(cfg["vae_path"]),
            "-p",
            prompt,
            "-W",
            str(width),
            "-H",
            str(height),
            "--video-frames",
            str(eff_frames),
            "--fps",
            str(eff_fps),
            "--steps",
            str(eff_steps),
            "--cfg-scale",
            str(eff_guidance),
            "--flow-shift",
            str(eff_shift),
            "--sampling-method",
            str(cfg.get("sampler", "euler")),
            "-t",
            str(int(cfg.get("threads", 8))),
            "-o",
            str(raw_webm),
        ]
        if eff_neg:
            cmd.extend(["-n", eff_neg])
        if init_img:
            cmd.extend(["-i", str(Path(init_img).expanduser().resolve())])
        if end_img:
            cmd.extend(["--end-img", str(Path(end_img).expanduser().resolve())])
        if seed is not None:
            cmd.extend(["-s", str(seed)])
        for extra in cfg.get("extra_args", []):
            cmd.append(str(extra))

        env = os.environ.copy()
        ld_extra = cfg.get("ld_library_path")
        if ld_extra:
            cur_ld = env.get("LD_LIBRARY_PATH", "")
            env["LD_LIBRARY_PATH"] = f"{ld_extra}:{cur_ld}" if cur_ld else str(ld_extra)

        saved_ingine = _find_ingine_serve() if gpu_handoff else None
        ingine_launched = False
        try:
            if saved_ingine is not None:
                _stop_ingine_serve(saved_ingine[0], saved_ingine[2])
            proc = subprocess.run(cmd, env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            if proc.returncode != 0 or not raw_webm.is_file():
                raise RuntimeError(f"Video generation failed (rc={proc.returncode}): {proc.stderr[-400:]}")
        finally:
            if saved_ingine is not None:
                _, argv, port = saved_ingine
                _launch_ingine_serve(argv)
                ingine_launched = True

        # Run ffmpeg conversion on CPU while ingine-serve reloads into VRAM.
        _postprocess_video(raw_webm, out_path, loop=loop)
        if ingine_launched and saved_ingine is not None:
            _wait_ingine_health(saved_ingine[2])

    elapsed = round(time.monotonic() - t0, 2)
    return {
        "ok": True,
        "path": str(out_path),
        "width": width,
        "height": height,
        "frames": eff_frames * (2 if loop else 1),
        "fps": eff_fps,
        "loop": loop,
        "elapsed_sec": elapsed,
    }


def _try_host_daemon(payload_dict: dict[str, Any]) -> dict[str, Any] | None:
    if os.environ.get("INGINE_VIDEO_NO_DAEMON") == "1":
        return None
    port = int(os.environ.get("INGINE_VIDEO_PORT", "8092"))
    url = f"http://127.0.0.1:{port}/generate"
    payload = json.dumps(payload_dict).encode("utf-8")
    req = urllib.request.Request(url, data=payload, headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=900.0) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except urllib.error.URLError:
        return None


def main() -> int:
    parser = argparse.ArgumentParser(prog="gen-video", description="Generate video assets with GPU handoff.")
    parser.add_argument("--prompt", required=True, help="Detailed English prompt describing motion and scene")
    parser.add_argument("--out", required=True, help="Output video path (.mp4 or .webm)")
    parser.add_argument("-i", "--init-img", default=None, help="Optional initial keyframe image")
    parser.add_argument("--end-img", default=None, help="Optional ending keyframe image (FLF2V)")
    parser.add_argument("--size", default=None, help="Output WxH (default: 832x480)")
    parser.add_argument("--frames", type=int, default=None, help="Frame count (default: 81 = ~5s @ 16fps)")
    parser.add_argument("--fps", type=int, default=None, help="Frames per second (default: 16)")
    parser.add_argument("--steps", type=int, default=None, help="Diffusion steps (default: 20)")
    parser.add_argument("--guidance", type=float, default=None, help="CFG scale (default: 6.0)")
    parser.add_argument("--flow-shift", type=float, default=None, help="Flow shift (default: 5.0)")
    parser.add_argument("--seed", type=int, default=None, help="Optional random seed")
    parser.add_argument("--negative-prompt", default=None, help="Optional negative prompt")
    parser.add_argument("--loop", action="store_true", help="Create seamless ping-pong loop (forward + reverse)")
    parser.add_argument("--quiet", action="store_true", help="Suppress JSON output")
    args = parser.parse_args()

    payload = {
        "prompt": args.prompt,
        "out": str(Path(args.out).expanduser().resolve()),
        "init_img": str(Path(args.init_img).expanduser().resolve()) if args.init_img else None,
        "end_img": str(Path(args.end_img).expanduser().resolve()) if args.end_img else None,
        "size": args.size,
        "frames": args.frames,
        "fps": args.fps,
        "steps": args.steps,
        "guidance": args.guidance,
        "flow_shift": args.flow_shift,
        "seed": args.seed,
        "negative_prompt": args.negative_prompt,
        "loop": bool(args.loop),
    }
    result = _try_host_daemon(payload)
    if result is None:
        result = generate_video(**payload)
    if not result.get("ok"):
        raise RuntimeError(str(result.get("error", "Video generation failed")))
    if not args.quiet:
        print(json.dumps(result))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(json.dumps({"ok": False, "error": str(exc)}), file=sys.stderr)
        raise SystemExit(1)
