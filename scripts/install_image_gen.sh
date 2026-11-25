#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV_DIR="${INGINE_IMAGE_VENV:-$ROOT/.tools/image-gen-venv}"
CFG_DIR="${XDG_CONFIG_HOME:-$HOME/.config}/ingine"
CFG_FILE="$CFG_DIR/image-gen.json"
CLI_SRC="$ROOT/tools/image_gen/gen_image.py"

if [[ ! -d "$VENV_DIR" ]]; then
  echo "Creating image-gen CPU venv at $VENV_DIR..."
  python3 -m venv "$VENV_DIR"
fi

"$VENV_DIR/bin/pip" install --upgrade pip wheel >/dev/null
"$VENV_DIR/bin/pip" install --index-url https://download.pytorch.org/whl/cpu torch
"$VENV_DIR/bin/pip" install diffusers transformers accelerate sentencepiece pillow

mkdir -p "$CFG_DIR"
if [[ ! -f "$CFG_FILE" ]]; then
  cat > "$CFG_FILE" <<'EOF'
{
  "backend": "sana-sprint-0.6b",
  "model_id": "Efficient-Large-Model/Sana_Sprint_0.6B_1024px_diffusers",
  "dtype": "bfloat16",
  "threads": 8,
  "default_size": "512x512",
  "default_steps": 2,
  "default_guidance": 4.5
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
