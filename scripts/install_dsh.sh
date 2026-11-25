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
CTX_LEN="${INGINE_CTX:-114688}"
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
if grep -qE "contextWindow: (98304|131072)" "$PATCH_FILE" 2>/dev/null; then
  sed -i -E "s/contextWindow: (98304|131072)/contextWindow: ${CTX_LEN}/g" "$PATCH_FILE"
fi
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
EOF
  echo "Created default workspace rules -> $AGENTS_FILE."
fi
