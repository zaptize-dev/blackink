#!/usr/bin/env bash
# Behaviour of the abliterated model against the original, both served by ingine: refusals (32 mild prompts,
# temperature 0, without and with reasoning) and capability (math 20, code 12, knowledge 12). One server at
# a time on port 8017. REASONING=1 adds reasoning stability (tools/reasoning_eval.py, GSM8K and AIME at
# temperature 1, about half an hour per model) served with the default profile (FP8 KV cache, DSpark, 262K).
#   ABLATION=ablation/inkling-2-mini-abliterated.safetensors tools/eval.sh
set -euo pipefail
HERE=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
ROOT=$(cd -- "$HERE/.." && pwd)
PY=$HERE/.venv/bin/python
SERVER=${SERVER:-$ROOT/native/build-install/apps/serve/ingine-serve}  # built by scripts/install.sh
PORT=8017
MODEL=${MODEL:-${INGINE_MODEL_DIR:-$ROOT/models/Inkling-2-Mini}}
ABLATION=${ABLATION:?set ABLATION to the ablation file}
OUT=${OUT:-$ROOT/results}
mkdir -p "$OUT"

serve() {
  local label=$1; shift
  "$SERVER" --engine --model-dir "$MODEL" --host 127.0.0.1 --port $PORT --dense-fp8 --max-ctx 16384 "$@" \
      >"$OUT/serve-$label.log" 2>&1 &
  PID=$!
  for _ in $(seq 1 300); do
    if curl -sf "http://127.0.0.1:$PORT/health" >/dev/null 2>&1; then return 0; fi
    if ! kill -0 $PID 2>/dev/null; then echo "server for $label exited"; tail -20 "$OUT/serve-$label.log"; return 1; fi
    sleep 1
  done
  echo "server for $label not ready"; return 1
}
stop() { kill -TERM $PID; wait $PID || true; }  # background jobs of a non-interactive shell ignore SIGINT

run() {
  local label=$1
  "$PY" "$HERE/refusal_eval.py" --base-url "http://127.0.0.1:$PORT/v1" --label "$label" --out "$OUT/refusals-$label.json" | tail -3
  "$PY" "$HERE/refusal_eval.py" --base-url "http://127.0.0.1:$PORT/v1" --label "$label-thinking" --thinking \
      --out "$OUT/refusals-$label-thinking.json" | tail -3
  "$PY" "$HERE/capability_eval.py" --base-url "http://127.0.0.1:$PORT/v1" --label "$label" --out "$OUT/capability-$label.json" | tail -1
}

serve abliterated --ablation "$ABLATION"
run abliterated
stop
serve original
run original
stop

for mode in "" "-thinking"; do
  "$PY" "$HERE/refusal_eval.py" --compare "$OUT/refusals-original$mode.json" "$OUT/refusals-abliterated$mode.json"
done
"$PY" "$HERE/capability_eval.py" --compare "$OUT/capability-original.json" "$OUT/capability-abliterated.json"

if [[ ${REASONING:-0} == 1 ]]; then
  profile=(--kv-fp8 --spec --spec-threshold 0.8 --max-ctx 262144)
  for label in abliterated original; do
    if [[ $label == abliterated ]]; then serve "reasoning-$label" "${profile[@]}" --ablation "$ABLATION"
    else serve "reasoning-$label" "${profile[@]}"; fi
    "$PY" "$HERE/reasoning_eval.py" --base-url "http://127.0.0.1:$PORT/v1" --label "$label" \
        --out "$OUT/reasoning-$label.json" | tail -1
    stop
  done
  "$PY" "$HERE/reasoning_eval.py" --compare "$OUT/reasoning-original.json" "$OUT/reasoning-abliterated.json"
fi
