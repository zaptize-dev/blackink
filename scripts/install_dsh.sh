#!/usr/bin/env bash
# Install and configure DeepSeek Harness (@deepseek-ai/dsh) pre-wired to BlackInk's local API.
set -euo pipefail

ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)"
DSH_APP="$ROOT/.dsh-app"
DSH_PKG="@deepseek-ai/dsh@0.2.2-rc.5"
DSH_HOME="${DSH_HOME:-$HOME/.dsh}"

if ! command -v node >/dev/null 2>&1 || ! command -v npm >/dev/null 2>&1; then
  echo "Warning: node/npm not found; skipping DeepSeek Harness (/editor) installation." >&2
  echo "Install Node.js >= 22 and re-run setup.sh --update to enable /editor." >&2
  exit 0
fi

mkdir -p "$DSH_APP"
if [[ ! -x "$DSH_APP/node_modules/.bin/dsh" ]]; then
  echo "Installing $DSH_PKG into $DSH_APP..."
  npm --prefix "$DSH_APP" install --no-audit --no-fund "$DSH_PKG"
else
  echo "DeepSeek Harness already installed in $DSH_APP."
fi

mkdir -p "$DSH_HOME/profiles/web" "$ROOT/workspace"
if [[ ! -f "$DSH_HOME/.env" ]] || ! grep -q '^BLACKINK_API_KEY=' "$DSH_HOME/.env" 2>/dev/null; then
  printf 'BLACKINK_API_KEY=none\n' >> "$DSH_HOME/.env"
fi

API_PORT="${INGINE_API_PORT:-18090}"
CTX_LEN="${INGINE_CTX:-106496}"
WORKSPACE_DIR="${DSH_WORKSPACE:-$HOME/workspace}"
LAN_IP="${INGINE_LAN_IP:-$(ip -4 route get 1.1.1.1 2>/dev/null | awk '{for(i=1;i<=NF;i++) if($i=="src"){print $(i+1); exit}}' || true)}"
if [[ -z "$LAN_IP" ]]; then
  LAN_IP="$(hostname -I 2>/dev/null | awk '{print $1}' || true)"
fi
LAN_IP="${LAN_IP:-127.0.0.1}"
mkdir -p "$WORKSPACE_DIR" "$DSH_HOME/storages"
PATCH_FILE="$DSH_HOME/profiles/web/cordis.patch.yml"
if [[ ! -f "$PATCH_FILE" ]] || ! grep -q 'blackink:' "$PATCH_FILE" 2>/dev/null; then
  cat > "$PATCH_FILE" <<EOF
- id: working-directory
  config:
    defaultDirectory: ${WORKSPACE_DIR}
- id: llm-pi-ai
  config:
    providers:
      blackink:
        api: openai-completions
        baseURL: http://${LAN_IP}:${API_PORT}/v1
        apiKeyEnv: BLACKINK_API_KEY
        reasoning: medium
        models:
          - id: inkling-2-mini-abliterated
            name: Inkling 2 Mini Abliterated (BlackInk)
            contextWindow: ${CTX_LEN}
            maxTokens: 24576
            input: [text, image]
            reasoningEfforts:
              off: none
              low: low
              medium: medium
              high: high
              max: max
- id: agent-default-model
  config:
    provider: blackink
    model: inkling-2-mini-abliterated
EOF
  echo "Configured DeepSeek Harness default model -> http://${LAN_IP}:${API_PORT}/v1 ($PATCH_FILE)."
elif grep -q "baseURL: http://127.0.0.1:${API_PORT}/v1" "$PATCH_FILE" 2>/dev/null && [[ "$LAN_IP" != "127.0.0.1" ]]; then
  sed -i "s|baseURL: http://127.0.0.1:${API_PORT}/v1|baseURL: http://${LAN_IP}:${API_PORT}/v1|g" "$PATCH_FILE"
  echo "Updated DeepSeek Harness BlackInk baseURL -> http://${LAN_IP}:${API_PORT}/v1."
fi
if grep -q "defaultDirectory: ${ROOT}/workspace" "$PATCH_FILE" 2>/dev/null; then
  sed -i "s|defaultDirectory: ${ROOT}/workspace|defaultDirectory: ${WORKSPACE_DIR}|g" "$PATCH_FILE"
  echo "Updated DeepSeek Harness defaultDirectory -> ${WORKSPACE_DIR}."
fi
if grep -qE "input:[[:space:]]*\[[[:space:]]*text[[:space:]]*\]" "$PATCH_FILE" 2>/dev/null; then
  sed -i 's/input:[[:space:]]*\[[[:space:]]*text[[:space:]]*\]/input: [text, image]/g' "$PATCH_FILE"
  echo "Enabled image input for BlackInk in DeepSeek Harness."
fi
if grep -qE "contextWindow: (98304|114688|131072)" "$PATCH_FILE" 2>/dev/null; then
  sed -i -E "s/contextWindow: (98304|114688|131072)/contextWindow: ${CTX_LEN}/g" "$PATCH_FILE"
fi
# Remove orphaned top-level compaction-basic entry if present; dsh-web-app mounts compaction-basic inside presets.
python3 - "$PATCH_FILE" "$DSH_APP/node_modules/@deepseek-ai/dsh-web-app/presets" <<'PY'
import glob, os, re, sys

patch_file, presets_dir = sys.argv[1], sys.argv[2]
if os.path.exists(patch_file):
    txt = open(patch_file).read()
    new = re.sub(
        r"\n?- name: '@deepseek-ai/dsh-compaction-basic'\n(?:  .*\n)*",
        "\n",
        txt,
    )
    if new != txt:
        open(patch_file, "w").write(new)

target = "                name: '@deepseek-ai/dsh-compaction-basic'\n"
cfg = (
    "                name: '@deepseek-ai/dsh-compaction-basic'\n"
    "                config:\n"
    "                  thresholdRatio: 0.78\n"
    "                  retainRatio: 0.16\n"
    "                  headroomTokens: 8192\n"
)
for p in sorted(glob.glob(os.path.join(presets_dir, "*.patch.yml"))):
    s = open(p).read()
    if "headroomTokens:" not in s and target in s:
        open(p, "w").write(s.replace(target, cfg, 1))
        print(f"Patched compaction-basic headroomTokens=8192 in {os.path.basename(p)}")
PY
if grep -q "maxTokens: 32768" "$PATCH_FILE" 2>/dev/null; then
  sed -i "s/maxTokens: 32768/maxTokens: 24576/g" "$PATCH_FILE"
fi

WS_FILE="$DSH_HOME/storages/workspace.json"
if [[ ! -f "$WS_FILE" ]]; then
  NOW_ISO="$(date -u '+%Y-%m-%dT%H:%M:%S.000Z')"
  cat > "$WS_FILE" <<EOF
{
  "unit": {
    "name": "workspace",
    "version": 2
  },
  "global": {
    "initialized": true,
    "workspaceIds": [
      "00000000-0000-4000-8000-000000000001"
    ],
    "archivedSessionIds": [],
    "pinnedSessionIds": []
  },
  "tables": {
    "workspaces": {
      "00000000-0000-4000-8000-000000000001": {
        "path": "${WORKSPACE_DIR}",
        "title": "workspace",
        "sessionIds": [],
        "createdAt": "${NOW_ISO}",
        "updatedAt": "${NOW_ISO}"
      }
    }
  }
}
EOF
  echo "Initialized DeepSeek Harness default workspace -> $WORKSPACE_DIR ($WS_FILE)."
fi

AGENTS_FILE="$WORKSPACE_DIR/AGENTS.md"
if [[ ! -f "$AGENTS_FILE" ]]; then
  cat > "$AGENTS_FILE" <<EOF
# Workspace Rules

1. **One folder per project:** Whenever the user asks to create a new project, create a dedicated subfolder \`./<project-slug>/\` inside this workspace and initialize its own \`git init\` repository inside that project folder (never at the workspace root).
2. **Strict directory isolation:** Work ONLY inside the active project folder (\`./<project-slug>/\`). Never list, read, modify, or run commands against sibling project folders, parent directories (\`../\`), \`$HOME\`, \`/home/zaptize/blackink\`, or system directories.
3. **Preview & access link:** To run or test any web app or API, pick an unused TCP port in \`4100-4999\` (check first with \`ss -ltn\`), bind the dev server to \`0.0.0.0\` (e.g. \`--host 0.0.0.0 --port <PORT>\`), detach it so it stays alive after the tool exits (\`nohup setsid <cmd> >/tmp/preview-<PORT>.log 2>&1 </dev/null &\`), verify it is listening, and always reply with the full network link \`http://${LAN_IP}:<PORT>\`.
4. **Deploy-ready structure:** When the user asks to prepare for Git / Coolify, include a clean \`Dockerfile\` (or \`docker-compose.yml\`) and \`.gitignore\` inside the project folder and stop any temporary dev server on that port if requested.
5. **Image assets:** For icons, logos, and simple graphics, prefer inline SVG. When a project needs raster photos, illustrations, or hero banners, generate them with \`~/.local/bin/gen-image --prompt "<detailed English prompt>" --out ./<project-slug>/assets/<name>.png --size 512x512\` (supports sizes like \`512x512\`, \`768x512\`, \`512x768\`, and \`--transparent\` for background-free PNG/WebP cutouts) and reference them from HTML/CSS.
EOF
  echo "Created default workspace rules -> $AGENTS_FILE."
elif ! grep -q "gen-image" "$AGENTS_FILE" 2>/dev/null; then
  cat >> "$AGENTS_FILE" <<EOF
5. **Image assets:** For icons, logos, and simple graphics, prefer inline SVG. When a project needs raster photos, illustrations, or hero banners, generate them with \`~/.local/bin/gen-image --prompt "<detailed English prompt>" --out ./<project-slug>/assets/<name>.png --size 512x512\` (supports sizes like \`512x512\`, \`768x512\`, \`512x768\`, and \`--transparent\` for background-free PNG/WebP cutouts) and reference them from HTML/CSS.
EOF
  echo "Added gen-image rule to $AGENTS_FILE."
elif ! grep -q -- "--transparent" "$AGENTS_FILE" 2>/dev/null; then
  sed -i 's|512x768`)|512x768`, and `--transparent` for background-free PNG/WebP cutouts)|g' "$AGENTS_FILE"
  echo "Updated gen-image rule with --transparent in $AGENTS_FILE."
fi
if grep -q '`gen-image --prompt' "$AGENTS_FILE" 2>/dev/null; then
  sed -i 's|`gen-image --prompt|`~/.local/bin/gen-image --prompt|g' "$AGENTS_FILE"
fi
