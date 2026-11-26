#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV_DIR="${INGINE_IMAGE_VENV:-$ROOT/.tools/image-gen-venv}"
SD_DIR="${INGINE_SD_CPP_DIR:-$ROOT/.tools/stable-diffusion.cpp}"
MODEL_DIR="${INGINE_IMAGE_MODEL_DIR:-$ROOT/models/image}"
MODEL_FILE="$MODEL_DIR/sdxl-lightning-4step-q4_0.gguf"
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
if [[ ! -f "$MODEL_FILE" ]]; then
  echo "Downloading and quantizing default CPU image model..."
  FP16_PATH="$("$VENV_DIR/bin/python" -c "from huggingface_hub import hf_hub_download; print(hf_hub_download('ByteDance/SDXL-Lightning', 'sdxl_lightning_4step.safetensors'))")"
  "$SD_DIR/build/bin/sd-cli" -M convert -m "$FP16_PATH" -o "$MODEL_FILE" --type q4_0
  rm -rf "${HF_HOME:-$HOME/.cache/huggingface}/hub/models--ByteDance--SDXL-Lightning"
fi

mkdir -p "$CFG_DIR"
if [[ ! -f "$CFG_FILE" ]]; then
  cat > "$CFG_FILE" <<EOF
{
  "backend": "sd-cli",
  "sd_cli_bin": "$SD_DIR/build/bin/sd-cli",
  "model_path": "$MODEL_FILE",
  "sampler": "euler",
  "threads": 8,
  "default_size": "512x512",
  "default_steps": 4,
  "default_guidance": 1.0
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
