#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV_DIR="${INGINE_IMAGE_VENV:-$ROOT/.tools/image-gen-venv}"
SD_DIR="${INGINE_SD_CPP_DIR:-$ROOT/.tools/stable-diffusion.cpp}"
MODEL_DIR="${INGINE_IMAGE_MODEL_DIR:-$ROOT/models/image}/flux2"
DIFFUSION_FILE="$MODEL_DIR/flux-2-klein-4b-Q4_0.gguf"
LLM_FILE="$MODEL_DIR/Qwen3-4B-Q4_K_M.gguf"
TAE_FILE="$MODEL_DIR/taef2.safetensors"
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

mkdir -p "$MODEL_DIR"
if [[ ! -f "$DIFFUSION_FILE" || ! -f "$LLM_FILE" || ! -f "$TAE_FILE" ]]; then
  echo "Downloading default CPU image model..."
  "$VENV_DIR/bin/python" - "$MODEL_DIR" <<'PY'
import os, shutil, sys
from huggingface_hub import hf_hub_download

d = sys.argv[1]
files = [
    ("leejet/FLUX.2-klein-4B-GGUF", "flux-2-klein-4b-Q4_0.gguf", "flux-2-klein-4b-Q4_0.gguf"),
    ("Qwen/Qwen3-4B-GGUF", "Qwen3-4B-Q4_K_M.gguf", "Qwen3-4B-Q4_K_M.gguf"),
    ("madebyollin/taef2", "taef2.safetensors", "taef2.safetensors"),
]
for repo, src, dst in files:
    out = os.path.join(d, dst)
    if not os.path.exists(out):
        p = hf_hub_download(repo, src)
        shutil.copy2(p, out)
PY
  rm -rf "${HF_HOME:-$HOME/.cache/huggingface}/hub/models--leejet--FLUX.2-klein-4B-GGUF" \
         "${HF_HOME:-$HOME/.cache/huggingface}/hub/models--Qwen--Qwen3-4B-GGUF" \
         "${HF_HOME:-$HOME/.cache/huggingface}/hub/models--madebyollin--taef2"
fi

mkdir -p "$CFG_DIR"
if [[ ! -f "$CFG_FILE" ]]; then
  cat > "$CFG_FILE" <<EOF
{
  "backend": "sd-cli",
  "sd_cli_bin": "$SD_DIR/build/bin/sd-cli",
  "diffusion_model": "$DIFFUSION_FILE",
  "llm_path": "$LLM_FILE",
  "taesd_path": "$TAE_FILE",
  "sampler": "euler",
  "threads": 8,
  "default_size": "512x512",
  "default_steps": 3,
  "default_guidance": 1.0,
  "extra_args": ["--flash-attn", "off,diffusion=on"]
}
EOF
  echo "Wrote default config -> $CFG_FILE"
fi

WRAPPER_LOCAL="$HOME/.local/bin/gen-image"
mkdir -p "$HOME/.local/bin"
cat > "$WRAPPER_LOCAL" <<EOF
#!/usr/bin/env bash
exec "$VENV_DIR/bin/python" "$CLI_SRC" "\$@"
EOF
chmod +x "$WRAPPER_LOCAL"

if [[ -w /usr/local/bin ]]; then
  ln -sf "$WRAPPER_LOCAL" /usr/local/bin/gen-image
  echo "Installed gen-image -> /usr/local/bin/gen-image"
elif command -v sudo >/dev/null 2>&1 && sudo -n true 2>/dev/null; then
  sudo ln -sf "$WRAPPER_LOCAL" /usr/local/bin/gen-image
  echo "Installed gen-image -> /usr/local/bin/gen-image"
else
  echo "Installed gen-image -> $WRAPPER_LOCAL (ensure \$HOME/.local/bin is in PATH)"
fi
