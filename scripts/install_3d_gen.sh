#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV_DIR="${INGINE_3D_VENV:-$ROOT/.tools/gen3d-venv}"
HY3D_DIR="${INGINE_HY3D_DIR:-$ROOT/.tools/Hunyuan3D-2}"
CUDA_ROOT="${CUDAToolkit_ROOT:-$ROOT/.cuda-toolkit/nvidia/cu13}"
UV_BIN="${UV_BIN:-$ROOT/.tools/bin/uv}"
CFG_DIR="${XDG_CONFIG_HOME:-$HOME/.config}/ingine"
CFG_FILE="$CFG_DIR/3d-gen.json"
CLI_SRC="$ROOT/tools/gen_3d/gen_3d.py"
SERVER_SRC="$ROOT/tools/gen_3d/server_3d.py"

if [[ ! -d "$VENV_DIR" ]]; then
  echo "Creating gen3d-venv (Python 3.11 + PyTorch cu128) at $VENV_DIR..."
  "$UV_BIN" venv --python 3.11 "$VENV_DIR"
  "$UV_BIN" pip install --python "$VENV_DIR/bin/python" torch torchvision --index-url https://download.pytorch.org/whl/cu128
  "$UV_BIN" pip install --python "$VENV_DIR/bin/python" \
    "transformers==4.49.0" "huggingface_hub>=0.26,<1.0" "diffusers==0.32.2" \
    einops opencv-python omegaconf tqdm trimesh pymeshlab pygltflib xatlas \
    accelerate rembg onnxruntime ninja pybind11 fast-simplification
fi

if [[ ! -d "$HY3D_DIR" ]]; then
  git clone --depth 1 https://github.com/Tencent-Hunyuan/Hunyuan3D-2.git "$HY3D_DIR"
  "$UV_BIN" pip install --python "$VENV_DIR/bin/python" -e "$HY3D_DIR" --no-deps
fi

mkdir -p "$CFG_DIR"
if [[ ! -f "$CFG_FILE" ]]; then
  cat > "$CFG_FILE" <<EOF
{
  "gpu_handoff": true,
  "model_id": "tencent/Hunyuan3D-2mini",
  "subfolder": "hunyuan3d-dit-v2-mini-turbo",
  "texgen_model_id": "tencent/Hunyuan3D-2",
  "default_steps": 15,
  "default_octree_res": 256,
  "default_max_faces": 25000,
  "default_texture": true
}
EOF
  echo "Wrote default config -> $CFG_FILE"
fi

WRAPPER_LOCAL="$HOME/.local/bin/gen-3d"
SERVER_LOCAL="$HOME/.local/bin/gen-3d-server"
mkdir -p "$HOME/.local/bin"
cat > "$WRAPPER_LOCAL" <<EOF
#!/usr/bin/env bash
export LD_LIBRARY_PATH="$CUDA_ROOT/lib:\${LD_LIBRARY_PATH:-}"
exec "$VENV_DIR/bin/python" "$CLI_SRC" "\$@"
EOF
chmod +x "$WRAPPER_LOCAL"

cat > "$SERVER_LOCAL" <<EOF
#!/usr/bin/env bash
export LD_LIBRARY_PATH="$CUDA_ROOT/lib:\${LD_LIBRARY_PATH:-}"
exec "$VENV_DIR/bin/python" "$SERVER_SRC" "\$@"
EOF
chmod +x "$SERVER_LOCAL"

if ! curl -sf http://127.0.0.1:8093/health >/dev/null 2>&1; then
  nohup setsid "$SERVER_LOCAL" >/tmp/gen-3d-server.log 2>&1 </dev/null &
  sleep 1
  echo "Started gen-3d-server on 127.0.0.1:8093"
fi
