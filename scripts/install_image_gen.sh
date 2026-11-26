#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV_DIR="${INGINE_IMAGE_VENV:-$ROOT/.tools/image-gen-venv}"
SD_DIR="${INGINE_SD_CPP_DIR:-$ROOT/.tools/stable-diffusion.cpp}"
MODEL_DIR="${INGINE_IMAGE_MODEL_DIR:-$ROOT/models/image}/flux2"
DIFFUSION_FILE="$MODEL_DIR/flux-2-klein-4b-Q4_0.gguf"
LLM_FILE="$MODEL_DIR/Qwen3-4B-Q4_K_M.gguf"
VAE_FILE="$MODEL_DIR/flux2-vae.safetensors"
CUDA_ROOT="${CUDAToolkit_ROOT:-$ROOT/.cuda-toolkit/nvidia/cu13}"
CFG_DIR="${XDG_CONFIG_HOME:-$HOME/.config}/ingine"
CFG_FILE="$CFG_DIR/image-gen.json"
CLI_SRC="$ROOT/tools/image_gen/gen_image.py"

if [[ ! -d "$VENV_DIR" ]]; then
  echo "Creating image-gen helper venv at $VENV_DIR..."
  python3 -m venv "$VENV_DIR"
fi
"$VENV_DIR/bin/pip" install --upgrade pip wheel >/dev/null
"$VENV_DIR/bin/pip" install huggingface-hub pillow "rembg[cpu]" >/dev/null

if [[ ! -d "$SD_DIR" ]]; then
  echo "Cloning stable-diffusion.cpp..."
  git clone --recursive --depth 1 https://github.com/leejet/stable-diffusion.cpp.git "$SD_DIR"
fi

if [[ ! -x "$SD_DIR/build/bin/sd-cli" ]]; then
  echo "Building sd-cli (CPU AVX2/FMA/F16C)..."
  export PATH="$ROOT/.tools/venv/bin:$PATH"
  cmake -S "$SD_DIR" -B "$SD_DIR/build" -G Ninja \
    -DCMAKE_BUILD_TYPE=Release \
    -DGGML_AVX2=ON \
    -DGGML_FMA=ON \
    -DGGML_F16C=ON \
    -DSD_BUILD_EXAMPLES=ON
  cmake --build "$SD_DIR/build" -j"$(nproc)" --target sd-cli
fi

GPU_HANDOFF=false
CUDA_BIN=""
CUDA_LIB=""
if [[ -x "$CUDA_ROOT/bin/nvcc" ]]; then
  CUDA_LIB="$CUDA_ROOT/lib"
  if [[ -f "$CUDA_LIB/libcublas.so.13" && ! -e "$CUDA_LIB/libcublas.so" ]]; then
    ln -sf libcublas.so.13 "$CUDA_LIB/libcublas.so"
  fi
  if [[ ! -x "$SD_DIR/build-cuda/bin/sd-cli" ]]; then
    echo "Building sd-cli (CUDA sm_120a)..."
    export PATH="$ROOT/.tools/venv/bin:$CUDA_ROOT/bin:$PATH"
    export CUDAToolkit_ROOT="$CUDA_ROOT"
    cmake -S "$SD_DIR" -B "$SD_DIR/build-cuda" -G Ninja \
      -DCMAKE_BUILD_TYPE=Release \
      -DSD_CUDA=ON \
      -DCMAKE_CUDA_ARCHITECTURES=120 \
      -DCMAKE_CUDA_COMPILER="$CUDA_ROOT/bin/nvcc" \
      -DCUDAToolkit_ROOT="$CUDA_ROOT" \
      -DSD_BUILD_EXAMPLES=ON
    cmake --build "$SD_DIR/build-cuda" -j"$(nproc)" --target sd-cli
  fi
  if [[ -x "$SD_DIR/build-cuda/bin/sd-cli" ]]; then
    GPU_HANDOFF=true
    CUDA_BIN="$SD_DIR/build-cuda/bin/sd-cli"
  fi
fi

mkdir -p "$MODEL_DIR"
if [[ ! -f "$DIFFUSION_FILE" || ! -f "$LLM_FILE" || ! -f "$VAE_FILE" ]]; then
  echo "Downloading default image model..."
  "$VENV_DIR/bin/python" - "$MODEL_DIR" <<'PY'
import os, shutil, sys
from huggingface_hub import hf_hub_download

d = sys.argv[1]
files = [
    ("leejet/FLUX.2-klein-4B-GGUF", "flux-2-klein-4b-Q4_0.gguf", "flux-2-klein-4b-Q4_0.gguf"),
    ("Qwen/Qwen3-4B-GGUF", "Qwen3-4B-Q4_K_M.gguf", "Qwen3-4B-Q4_K_M.gguf"),
    ("Comfy-Org/flux2-dev", "split_files/vae/flux2-vae.safetensors", "flux2-vae.safetensors"),
]
for repo, src, dst in files:
    out = os.path.join(d, dst)
    if not os.path.exists(out):
        p = hf_hub_download(repo, src)
        shutil.copy2(p, out)
PY
  rm -rf "${HF_HOME:-$HOME/.cache/huggingface}/hub/models--leejet--FLUX.2-klein-4B-GGUF" \
         "${HF_HOME:-$HOME/.cache/huggingface}/hub/models--Qwen--Qwen3-4B-GGUF" \
         "${HF_HOME:-$HOME/.cache/huggingface}/hub/models--Comfy-Org--flux2-dev"
fi

mkdir -p "$CFG_DIR"
if [[ ! -f "$CFG_FILE" ]]; then
  cat > "$CFG_FILE" <<EOF
{
  "backend": "sd-cli",
  "gpu_handoff": ${GPU_HANDOFF},
  "sd_cli_bin": "$SD_DIR/build/bin/sd-cli",
  "sd_cli_cuda_bin": "$CUDA_BIN",
  "ld_library_path": "$CUDA_LIB",
  "diffusion_model": "$DIFFUSION_FILE",
  "llm_path": "$LLM_FILE",
  "vae_path": "$VAE_FILE",
  "sampler": "euler",
  "threads": 8,
  "default_size": "512x512",
  "default_steps": 4,
  "default_guidance": 1.0,
  "extra_args": ["--flash-attn", "off,diffusion=on"]
}
EOF
  echo "Wrote default config -> $CFG_FILE"
fi

WRAPPER_LOCAL="$HOME/.local/bin/gen-image"
SERVER_LOCAL="$HOME/.local/bin/gen-image-server"
SERVER_SRC="$ROOT/tools/image_gen/image_server.py"
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

if ! curl -sf http://127.0.0.1:8091/health >/dev/null 2>&1; then
  nohup setsid "$SERVER_LOCAL" >/tmp/blackink-image-server.log 2>&1 </dev/null &
  sleep 1
  echo "Started gen-image-server on 127.0.0.1:8091"
fi

if [[ -w /usr/local/bin ]]; then
  ln -sf "$WRAPPER_LOCAL" /usr/local/bin/gen-image
  echo "Installed gen-image -> /usr/local/bin/gen-image"
elif command -v sudo >/dev/null 2>&1 && sudo -n true 2>/dev/null; then
  sudo ln -sf "$WRAPPER_LOCAL" /usr/local/bin/gen-image
  echo "Installed gen-image -> /usr/local/bin/gen-image"
else
  echo "Installed gen-image -> $WRAPPER_LOCAL (ensure \$HOME/.local/bin is in PATH)"
fi
