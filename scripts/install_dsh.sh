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
CTX_LEN="${INGINE_CTX:-98304}"
WORKSPACE_DIR="${DSH_WORKSPACE:-$ROOT/workspace}"
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
        baseURL: http://127.0.0.1:${API_PORT}/v1
        apiKeyEnv: BLACKINK_API_KEY
        reasoning: medium
        models:
          - id: inkling-2-mini-abliterated
            name: Inkling 2 Mini Abliterated (BlackInk)
            contextWindow: ${CTX_LEN}
            maxTokens: 32768
            input: [text]
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
  echo "Configured DeepSeek Harness default model -> blackink/inkling-2-mini-abliterated ($PATCH_FILE)."
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
