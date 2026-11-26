#!/usr/bin/env python3
"""3D mesh generation CLI (Hunyuan3D-2mini -> .glb) with GPU handoff for BlackInk."""

from __future__ import annotations

import argparse
import fcntl
import json
import os
import signal
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

DEFAULT_CONFIG_PATH = Path.home() / ".config" / "ingine" / "3d-gen.json"
LOCK_PATH = Path("/tmp/blackink-gen-image.lock")

DEFAULT_CONFIG: dict[str, Any] = {
    "gpu_handoff": True,
    "model_id": "tencent/Hunyuan3D-2mini",
    "subfolder": "hunyuan3d-dit-v2-mini-turbo",
    "texgen_model_id": "tencent/Hunyuan3D-2",
    "default_steps": 15,
    "default_octree_res": 256,
    "default_chunks": 20000,
    "default_max_faces": 25000,
    "default_texture": True,
    "gen_image_bin": "/home/zaptize/.local/bin/gen-image",
}


def load_config() -> dict[str, Any]:
    cfg = dict(DEFAULT_CONFIG)
    cfg_path = Path(os.environ.get("INGINE_3D_CONFIG", str(DEFAULT_CONFIG_PATH)))
    if cfg_path.is_file():
        with cfg_path.open("r", encoding="utf-8") as fh:
            user_cfg = json.load(fh)
        if isinstance(user_cfg, dict):
            cfg.update(user_cfg)
    return cfg


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
    raise RuntimeError("Timed out waiting for ingine-serve to recover after 3D generation")


def _run_3d_pipeline(
    cfg: dict[str, Any],
    image_path: Path,
    out_path: Path,
    steps: int,
    octree_res: int,
    seed: int | None,
    max_faces: int,
    texture: bool,
) -> tuple[int, int, bool]:
    import gc
    import torch
    from PIL import Image
    from hy3dgen.rembg import BackgroundRemover
    from hy3dgen.shapegen import Hunyuan3DDiTFlowMatchingPipeline

    with Image.open(image_path) as raw_img:
        if raw_img.mode != "RGBA":
            rembg = BackgroundRemover()
            rgba = rembg(raw_img.convert("RGB"))
            del rembg
        else:
            rgba = raw_img.copy()

    pipe = Hunyuan3DDiTFlowMatchingPipeline.from_pretrained(
        str(cfg.get("model_id", "tencent/Hunyuan3D-2mini")),
        subfolder=str(cfg.get("subfolder", "hunyuan3d-dit-v2-mini-turbo")),
        use_safetensors=True,
        device="cuda",
    )
    if "turbo" in str(cfg.get("subfolder", "")).lower():
        pipe.enable_flashvdm()

    gen = torch.manual_seed(seed) if seed is not None else None
    mesh = pipe(
        image=rgba,
        num_inference_steps=steps,
        octree_resolution=octree_res,
        num_chunks=int(cfg.get("default_chunks", 20000)),
        generator=gen,
        output_type="trimesh",
    )[0]

    del pipe
    gc.collect()
    torch.cuda.empty_cache()

    if max_faces > 0 and len(mesh.faces) > max_faces:
        import fast_simplification
        import trimesh

        verts_out, faces_out = fast_simplification.simplify(
            mesh.vertices, mesh.faces, target_count=max_faces
        )
        mesh = trimesh.Trimesh(vertices=verts_out, faces=faces_out, process=True)

    textured = False
    if texture:
        from hy3dgen.texgen import Hunyuan3DPaintPipeline

        tex_pipe = Hunyuan3DPaintPipeline.from_pretrained(
            str(cfg.get("texgen_model_id", "tencent/Hunyuan3D-2"))
        )
        mesh = tex_pipe(mesh, image=rgba)
        textured = True
        del tex_pipe
        gc.collect()
        torch.cuda.empty_cache()

    out_path.parent.mkdir(parents=True, exist_ok=True)
    mesh.export(str(out_path))
    return len(mesh.vertices), len(mesh.faces), textured


def generate_3d(
    *,
    out: str,
    image: str | None = None,
    prompt: str | None = None,
    steps: int | None = None,
    octree_res: int | None = None,
    seed: int | None = None,
    max_faces: int | None = None,
    texture: bool | None = None,
) -> dict[str, Any]:
    if not image and not prompt:
        raise ValueError("Either --image or --prompt is required")
    cfg = load_config()
    eff_steps = int(steps if steps is not None else cfg.get("default_steps", 15))
    eff_octree = int(octree_res if octree_res is not None else cfg.get("default_octree_res", 380))
    eff_faces = int(max_faces if max_faces is not None else cfg.get("default_max_faces", 40000))
    eff_tex = bool(texture if texture is not None else cfg.get("default_texture", True))
    out_path = Path(out).expanduser().resolve()
    gpu_handoff = bool(cfg.get("gpu_handoff", True))

    t0 = time.monotonic()
    with tempfile.TemporaryDirectory(prefix="blackink_3d_") as tmpdir:
        if image:
            src_img = Path(image).expanduser().resolve()
        else:
            src_img = Path(tmpdir) / "ref.png"
            gen_img_bin = str(cfg.get("gen_image_bin", "gen-image"))
            img_prompt = (
                f"{prompt.strip()}, centered single 3D product asset isolated on a pure solid white background, "
                "3/4 studio view, clean sharp silhouette, no text"
            )
            proc = subprocess.run(
                [gen_img_bin, "--prompt", img_prompt, "--out", str(src_img), "--size", "768x768"],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )
            if proc.returncode != 0 or not src_img.is_file():
                raise RuntimeError(f"Reference image generation failed: {proc.stderr[-300:]}")

        LOCK_PATH.parent.mkdir(parents=True, exist_ok=True)
        with LOCK_PATH.open("w") as lock_fh:
            fcntl.flock(lock_fh.fileno(), fcntl.LOCK_EX)
            saved_ingine = _find_ingine_serve() if gpu_handoff else None
            ingine_launched = False
            try:
                if saved_ingine is not None:
                    _stop_ingine_serve(saved_ingine[0], saved_ingine[2])
                # Run CUDA 3D pipeline in a fresh subprocess so 100% of VRAM/CUDA context
                # is released before ingine-serve reloads.
                worker_meta = Path(tmpdir) / "worker_meta.json"
                worker_cmd = [
                    sys.executable,
                    str(Path(__file__).resolve()),
                    "--_worker",
                    "--image",
                    str(src_img),
                    "--out",
                    str(out_path),
                    "--steps",
                    str(eff_steps),
                    "--octree-res",
                    str(eff_octree),
                    "--max-faces",
                    str(eff_faces),
                    "--_meta",
                    str(worker_meta),
                ]
                if not eff_tex:
                    worker_cmd.append("--no-texture")
                if seed is not None:
                    worker_cmd.extend(["--seed", str(seed)])
                wproc = subprocess.run(worker_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
                if wproc.returncode != 0 or not worker_meta.is_file():
                    raise RuntimeError(f"3D worker failed (rc={wproc.returncode}): {wproc.stderr[-400:]}")
                meta = json.loads(worker_meta.read_text(encoding="utf-8"))
                verts, faces, textured = int(meta["vertices"]), int(meta["faces"]), bool(meta["textured"])
            finally:
                if saved_ingine is not None:
                    _, argv, port = saved_ingine
                    _launch_ingine_serve(argv)
                    ingine_launched = True
            if ingine_launched and saved_ingine is not None:
                _wait_ingine_health(saved_ingine[2])

    elapsed = round(time.monotonic() - t0, 2)
    return {
        "ok": True,
        "path": str(out_path),
        "vertices": verts,
        "faces": faces,
        "textured": textured,
        "steps": eff_steps,
        "octree_res": eff_octree,
        "elapsed_sec": elapsed,
    }


def _try_host_daemon(payload_dict: dict[str, Any]) -> dict[str, Any] | None:
    if os.environ.get("INGINE_3D_NO_DAEMON") == "1":
        return None
    port = int(os.environ.get("INGINE_3D_PORT", "8093"))
    url = f"http://127.0.0.1:{port}/generate"
    payload = json.dumps(payload_dict).encode("utf-8")
    req = urllib.request.Request(url, data=payload, headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=600.0) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except urllib.error.URLError:
        return None


def main() -> int:
    parser = argparse.ArgumentParser(prog="gen-3d", description="Generate 3D .glb meshes with Hunyuan3D-2mini.")
    parser.add_argument("--image", default=None, help="Input reference image path")
    parser.add_argument("--prompt", default=None, help="Optional text prompt (auto-generates reference image)")
    parser.add_argument("--out", required=True, help="Output 3D file path (.glb, .obj, .stl)")
    parser.add_argument("--steps", type=int, default=None, help="Diffusion steps (default: 15)")
    parser.add_argument("--octree-res", type=int, default=None, help="Octree resolution (default: 380)")
    parser.add_argument("--max-faces", type=int, default=None, help="Decimate mesh to max faces (default: 40000)")
    parser.add_argument("--no-texture", action="store_true", help="Skip PBR texture painting (geometry only)")
    parser.add_argument("--seed", type=int, default=None, help="Optional random seed")
    parser.add_argument("--quiet", action="store_true", help="Suppress JSON output")
    parser.add_argument("--_worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--_meta", default=None, help=argparse.SUPPRESS)
    args = parser.parse_args()

    if args._worker:
        cfg = load_config()
        verts, faces, textured = _run_3d_pipeline(
            cfg,
            Path(args.image),
            Path(args.out),
            int(args.steps or cfg.get("default_steps", 15)),
            int(args.octree_res or cfg.get("default_octree_res", 256)),
            args.seed,
            int(args.max_faces or cfg.get("default_max_faces", 40000)),
            not bool(args.no_texture),
        )
        Path(args._meta).write_text(
            json.dumps({"vertices": verts, "faces": faces, "textured": textured}),
            encoding="utf-8",
        )
        return 0

    if not args.image and not args.prompt:
        parser.error("--image or --prompt is required")

    payload = {
        "image": str(Path(args.image).expanduser().resolve()) if args.image else None,
        "prompt": args.prompt,
        "out": str(Path(args.out).expanduser().resolve()),
        "steps": args.steps,
        "octree_res": args.octree_res,
        "max_faces": args.max_faces,
        "texture": False if args.no_texture else None,
        "seed": args.seed,
    }
    result = _try_host_daemon(payload)
    if result is None:
        result = generate_3d(**payload)
    if not result.get("ok"):
        raise RuntimeError(str(result.get("error", "3D generation failed")))
    if not args.quiet:
        print(json.dumps(result))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(json.dumps({"ok": False, "error": str(exc)}), file=sys.stderr)
        raise SystemExit(1)
