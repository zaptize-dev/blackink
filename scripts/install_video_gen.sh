#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV_DIR="${INGINE_IMAGE_VENV:-$ROOT/.tools/image-gen-venv}"
SD_DIR="${INGINE_SD_CPP_DIR:-$ROOT/.tools/stable-diffusion.cpp}"
MODEL_DIR="${INGINE_VIDEO_MODEL_DIR:-$ROOT/models/video}/wan22-ti2v-5b"
DIFFUSION_FILE="$MODEL_DIR/Wan2.2-TI2V-5B-Q6_K.gguf"
T5_FILE="$MODEL_DIR/umt5-xxl-encoder-Q4_K_M.gguf"
VAE_FILE="$MODEL_DIR/wan2.2_vae.safetensors"
CUDA_ROOT="${CUDAToolkit_ROOT:-$ROOT/.cuda-toolkit/nvidia/cu13}"
CFG_DIR="${XDG_CONFIG_HOME:-$HOME/.config}/ingine"
CFG_FILE="$CFG_DIR/video-gen.json"
CLI_SRC="$ROOT/tools/video_gen/gen_video.py"
SERVER_SRC="$ROOT/tools/video_gen/video_server.py"

mkdir -p "$MODEL_DIR"
if [[ ! -f "$DIFFUSION_FILE" || ! -f "$T5_FILE" || ! -f "$VAE_FILE" ]]; then
  echo "Downloading Wan 2.2 TI2V 5B (Q6_K) video models..."
  "$VENV_DIR/bin/python" - "$MODEL_DIR" <<'PY'
import os, shutil, sys
from huggingface_hub import hf_hub_download

d = sys.argv[1]
files = [
    ("QuantStack/Wan2.2-TI2V-5B-GGUF", "Wan2.2-TI2V-5B-Q6_K.gguf", "Wan2.2-TI2V-5B-Q6_K.gguf"),
    ("city96/umt5-xxl-encoder-gguf", "umt5-xxl-encoder-Q4_K_M.gguf", "umt5-xxl-encoder-Q4_K_M.gguf"),
    ("Comfy-Org/Wan_2.2_ComfyUI_Repackaged", "split_files/vae/wan2.2_vae.safetensors", "wan2.2_vae.safetensors"),
]
for repo, src, dst in files:
    out = os.path.join(d, dst)
    if not os.path.exists(out):
        p = hf_hub_download(repo, src)
        shutil.copy2(p, out)
PY
fi

mkdir -p "$CFG_DIR"
if [[ ! -f "$CFG_FILE" ]]; then
  cat > "$CFG_FILE" <<EOF
{
  "gpu_handoff": true,
  "sd_cli_cuda_bin": "$SD_DIR/build-cuda/bin/sd-cli",
  "ld_library_path": "$CUDA_ROOT/lib",
  "diffusion_model": "$DIFFUSION_FILE",
  "t5xxl": "$T5_FILE",
  "vae_path": "$VAE_FILE",
  "sampler": "euler",
  "threads": 8,
  "default_size": "832x480",
  "default_frames": 81,
  "default_fps": 16,
  "default_steps": 20,
  "default_guidance": 6.0,
  "default_flow_shift": 5.0
}
EOF
  echo "Wrote default config -> $CFG_FILE"
fi

WRAPPER_LOCAL="$HOME/.local/bin/gen-video"
SERVER_LOCAL="$HOME/.local/bin/gen-video-server"
mkdir -p "$HOME/.local/bin"
cat > "$WRAPPER_LOCAL" <<EOF
#!/usr/bin/env bash
exec "$VENV_DIR/bin/python" "$CLI_SRC" "\$@"
EOF
chmod +x "$WRAPPER_LOCAL"

cat > "$SERVER_LOCAL" <<EOF
#!/usr/bin/env bash
exec "$VENV_DIR/bin/python" "$SERVER_SRC" "\$@"
EOF
chmod +x "$SERVER_LOCAL"

if ! curl -sf http://127.0.0.1:8092/health >/dev/null 2>&1; then
  nohup setsid "$SERVER_LOCAL" >/tmp/gen-video-server.log 2>&1 </dev/null &
  sleep 1
  echo "Started gen-video-server on 127.0.0.1:8092"
fi
