# BlackInk — Inkling-2-Mini Abliterated (16 GB Blackwell Edition)

**BlackInk** is a 16 GB NVIDIA Blackwell-optimized fork of **ingine**, serving an abliterated
[thinkingmachines/Inkling-2-Mini](https://huggingface.co/thinkingmachines/Inkling-2-Mini) on a single **16 GB+ NVIDIA
Blackwell GPU (RTX 5060 Ti 16 GB, RTX 5070 Ti, RTX 5080, or RTX 5090)** with **98,304 (96K) tokens of Q4 context**
with speculative decoding enabled (up to **131,072 / 128K** without `--spec`), **Full-NVFP4 routed experts**, **FP4
Tensor Core MoE prefill**, **adaptive DSpark speculative decoding**, an **OpenAI-compatible API**, an **integrated Web
Chat UI** (`/` or `/chat`), and a **live GPU Monitor dashboard** (`/metrics`).

---

## Key Features Added Over Upstream `ingine`

* **On-the-fly `MXFP8 -> NVFP4` Expert Transcoding (`--full-nvfp4`)**: Layers 20–23 are transcoded on the GPU at startup
  from 8-bit `MXFP8` to 4-bit `NVFP4` (matching layers 0–19), `embed` is quantized to `FP8 E4M3` with per-row scales,
  and router/gate weights are deduplicated — saving **~2.32 GiB of VRAM** without modifying the checkpoint on disk.
* **4-bit Full-Attention KV Cache (`--kv-q4`)**: Compresses the 8 long-range full-attention visits to 4-bit symmetric
  non-linear quantization (`0.5 byte/element`), fitting **98,304 (96K) tokens of context** alongside `--spec` on
  16 GB GPUs (or **131,072 / 128K tokens in ~1.0 GiB** without `--spec`, while keeping the 40 sliding-window visits in
  exact `BF16`).
* **FP4 Block-Scaled Tensor Core MoE Prefill (`moe_fp4.cu`)**: Grouped `W4A4` two-term FP4 Tensor Core (`sm_120a`)
  MoE GEMMs for prefill (`prefill_chunk = 384`), doubling prompt-reading throughput compared to BF16 dequantization.
* **Adaptive DSpark Speculative Decoding (`--spec`)**: Combines the official 3-layer DSpark drafter with EMA acceptance
  cooldown (`ema_acc < 0.85` fallback to pipelined `width=1` steps) and a 4,096-token trigram suffix wake-up, plus
  `markov_w1` offloaded to pinned host RAM to preserve VRAM.
* **16 GB System RAM & VRAM Friendly**: Streams layer-by-layer loading through a lean 64 MiB pinned staging pool with
  CUDA Unified Memory fallback when VRAM is tight.
* **Integrated Web Chat UI, Live Monitor & Development Harness (`dsh`)**:
  * **`/` or `/chat`**: Official `llama-ui` SvelteKit interface with streaming `<think>` reasoning blocks,
    `reasoning_effort` controls, live `tok/s` telemetry, and multi-conversation history.
  * **`/metrics` & `/api/metrics`**: Live Strata-style GPU/engine telemetry dashboard.
  * **`/editor/`**: Built-in **DeepSeek Harness (`dsh`)** agentic workspace (installed by `setup.sh` when Node.js is
    available and pre-configured to use `inkling-2-mini-abliterated` as its default model).

---

## RTX 5060 Ti 16 GB Benchmarks

Measured on a single **NVIDIA GeForce RTX 5060 Ti 16 GB** (`sm_120a`) running:
`--max-ctx 98304 --dense-fp8 --full-nvfp4 --kv-q4 --spec --spec-threshold 0.70`

| Workload | Settings | Speed |
|---|---|---:|
| **PT-BR Conversation** | `T=1.0`, thinking enabled | **133.1 – 139.8 tok/s** |
| **Python QuickSort** | `T=1.0`, thinking enabled | **137.2 tok/s** |
| **HTML / CSS Generation** | `T=0.0`, no thinking | **137.9 – 172.5 tok/s** |
| **Prompt Prefill (~2.5K – 2.6K tokens)** | `prefill_chunk = 384`, FP4 MoE | **2,457.0 – 2,519.5 tok/s** |

* **VRAM footprint**: `15,792 MiB / 16,311 MiB` (`519 MiB` free) at `98,304` context with `--spec` enabled.

---

## Requirements & Installation

You need:
* **Linux x86_64**
* **NVIDIA Blackwell GPU (`sm_120a`, compute capability 12.0)** with **at least 16 GB VRAM** (RTX 5060 Ti 16 GB, RTX 5070 Ti, RTX 5080, RTX 5090; RTX PRO 4500/5000/6000 Blackwell untested)
* **NVIDIA driver for CUDA 13 (`580` or newer)**
* **C++20 compiler** (GCC 12 to 16)
* **~24 GB free disk space** (20.2 GB model files from [satellitedown/Inkling-2-Mini-abliterated-ingine](https://huggingface.co/satellitedown/Inkling-2-Mini-abliterated-ingine), ~1.2 GB private CUDA toolkit/build tools, ~1 GB download cache)

The server is a standalone C++/CUDA binary and needs no Python or PyTorch at runtime.

### Quick Install

```bash
curl -fsSL https://raw.githubusercontent.com/zaptize-dev/blackink/main/get.sh | bash
```

To install into a custom directory:

```bash
curl -fsSL https://raw.githubusercontent.com/zaptize-dev/blackink/main/get.sh | INGINE_DIR=/path/to/folder bash
```

Or clone and run the installer manually:

```bash
git clone https://github.com/zaptize-dev/blackink
cd blackink
bash setup.sh
```

In the setup menu:
1. **Install everything**: validates GPU/driver/compiler, installs CMake, Ninja, and a private CUDA 13.4 toolkit, compiles `ingine-serve`, and downloads the pinned model.
2. **Download / resume model**: downloads or verifies model files.
3. **Start the server (foreground)**: runs `ingine serve` in the current terminal (**Ctrl+C** stops it).
4. **Update**: pulls `origin/main` and runs an incremental rebuild (`bash setup.sh --update` does the same).
5. **Install the `ingine` command**: symlinks `~/.local/bin/ingine` to `scripts/ingine`.

### Directory Layout

| Path | Contents |
|---|---|
| `.tools/` | `uv` and a small Python environment (`.tools/venv/`) with CMake, Ninja, and `huggingface_hub` (installer/build only) |
| `.cuda-toolkit/` | Private CUDA 13.4 toolkit (`nvcc`, runtime, cuBLAS) isolated from system CUDA |
| `native/build-install/` | Compiled server binary (`apps/serve/ingine-serve`) and `profile.env` from [runtime-manifest.json](runtime-manifest.json) |
| `models/Inkling-2-Mini/` | Downloaded Inkling-2-Mini weights and abliteration tensor |
| `.run/` | PID file and server log when running without a systemd user session |

---

## Usage & Configuration

```bash
ingine start      # start the server in the background (systemd user unit inkling-ingine)
ingine stop       # stop the server and free the GPU
ingine status     # unit state, health, served model, GPU memory (0=ready, 1=starting, 3=stopped)
ingine links      # print active URLs: API, Chat (/chat), Monitor (/metrics), Editor (/editor/?token=...)
ingine logs       # view server logs (ingine logs -f to follow)
ingine serve      # run in the foreground; extra flags are passed to ingine-serve
ingine update     # pull latest commits and rebuild incrementally
```

Defaults come from [runtime-manifest.json](runtime-manifest.json) and can be overridden in `~/.config/ingine/env` or environment variables:

| Variable | Default | Meaning |
|---|---|---|
| `INGINE_MODEL_DIR` | `models/Inkling-2-Mini` | Model folder |
| `INGINE_ABLATION` | `$INGINE_MODEL_DIR/inkling-2-mini-abliterated.safetensors` | Abliteration tensor (empty serves original `inkling-2-mini`) |
| `INGINE_CTX` | `262144` (32 GB+) / `98304` (16 GB + spec) | Maximum KV context length |
| `HOST`, `PORT` | `127.0.0.1`, `8003` | Bind address and port |
| `INGINE_EXTRA_ARGS` | empty | Extra flags passed to `ingine-serve` |
| `INGINE_START_TIMEOUT` | `300` | Seconds `ingine start` waits for readiness |

Example `~/.config/ingine/env` for a **16 GB GPU (RTX 5060 Ti 16 GB)**:

```bash
HOST=0.0.0.0
PORT=8090
INGINE_CTX=98304
INGINE_EXTRA_ARGS="--dense-fp8 --full-nvfp4 --kv-q4 --spec --spec-threshold 0.70"
```

---

## Web UI, Live Monitor & API

* **Web Chat UI**: Open `http://<server-ip>:<port>/` (or `/chat`) in any browser.
* **Live Monitor**: Open `http://<server-ip>:<port>/metrics` (`GET /api/metrics` for JSON).
* **Health & Models**: `GET /health` and `GET /v1/models`.
* **OpenAI-Compatible Chat API**: `POST /v1/chat/completions` at `http://<server-ip>:<port>/v1`, model `inkling-2-mini-abliterated` (no API key required; keep bound to a trusted network).

Supported request parameters on `POST /v1/chat/completions`:
* `reasoning_effort`: `none`, `minimal`, `low`, `medium`, `high`, `xhigh`, `max` (or `0` to `0.99`; server default `medium` / `0.7`)
* `enable_thinking: false`: answer directly without `<think>` reasoning (also accepted in `chat_template_kwargs`)
* `tools`, `stop`, `max_tokens` / `max_completion_tokens`, `temperature`, `top_p`, `top_k`, `seed`
* `stream: true` (SSE streaming with `reasoning_content` and `content` deltas; `stream_options.include_usage` supported)

```bash
curl -s http://127.0.0.1:8090/v1/chat/completions -H 'Content-Type: application/json' -d '{
  "model": "inkling-2-mini-abliterated",
  "messages": [{"role": "user", "content": "Explain how hash maps handle collisions."}],
  "reasoning_effort": "low",
  "max_tokens": 2048
}'
```

```python
from openai import OpenAI

client = OpenAI(base_url="http://127.0.0.1:8090/v1", api_key="none")
stream = client.chat.completions.create(
    model="inkling-2-mini-abliterated",
    messages=[{"role": "user", "content": "Write a Python LRU cache with tests."}],
    reasoning_effort="medium",
    stream=True,
)
for chunk in stream:
    delta = chunk.choices[0].delta
    print(getattr(delta, "reasoning_content", None) or delta.content or "", end="", flush=True)
```

### Oh My Pi (OMP)

Merge into `~/.omp/agent/models.yml` (adjust `baseUrl` and `contextWindow` to match your `PORT` and `INGINE_CTX`):

```yaml
providers:
  ingine:
    baseUrl: http://127.0.0.1:8090/v1
    auth: none
    api: openai-completions
    models:
      - id: inkling-2-mini-abliterated
        name: Inkling 2 Mini abliterated (ingine)
        contextWindow: 98304
        maxTokens: 24576
        reasoning: true
        thinking:
          mode: effort
          efforts: [minimal, low, medium, high, xhigh, max]
          defaultLevel: medium
          requiresEffort: false
        input: [text]
        cost: {input: 0, output: 0, cacheRead: 0, cacheWrite: 0}
        compat:
          supportsDeveloperRole: true
          supportsStore: false
          supportsStrictMode: false
          supportsToolChoice: false
          supportsReasoningEffort: true
          reasoningEffortMap: {minimal: none}
          reasoningContentField: reasoning_content
          maxTokensField: max_tokens
          streamIdleTimeoutMs: 600000
```

---

## Upstream Reference Results (Abliteration & RTX 5090)

### Abliteration Quality

Refusals on held-out prompts (greedy decoding):

| Benchmark | Original | Abliterated |
|---|---:|---:|
| `harmful_test` (104), direct answer | 104/104 | 1/104 |
| JailbreakBench harmful behaviours (100), direct answer | 100/100 | 5/100 |
| `harmful_test` (48), reasoning first (effort low, 768-token budget) | 47/48 | 0/48 |

Capability & reasoning retention (greedy & `T=1.0`, raw data in [`results/`](results)):

| Benchmark | Original | Abliterated |
|---|---:|---:|
| Math (20, reasoning, effort low) | 20/20 | 20/20 |
| Code (12, tests executed) | 12/12 | 12/12 |
| Multiple choice (12) | 12/12 | 12/12 |
| GSM8K, effort low / high (`T=1.0`) | 40/40 / 40/40 | 39/40 / 40/40 |
| AIME 2024, effort high (`T=1.0`) | 13/15 | 13/15 |
| Reasoning loops (95 samples) | 0 | 0 |

### Upstream RTX 5090 Reference Speed (32 GB, 262K FP8 KV)

| | Chat | Code | Coding Agent (2K–39K) |
|---|---:|---:|---:|
| `temperature = 1` | 489 tok/s | 565 tok/s | 502 tok/s |
| `temperature = 0` (greedy) | 486 tok/s | 568 tok/s | 524 tok/s |

Prefill on RTX 5090: `10,900 tok/s` (20K), `9,600 tok/s` (64K), `8,100 tok/s` (131K), `7,000 tok/s` (200K).

---

## License and Credits

This repository (`ingine`, tools, installer, and abliteration tensor) is licensed under **Apache-2.0** (see [LICENSE](LICENSE) and [NOTICE](NOTICE)). The base model [Inkling-2-Mini](https://huggingface.co/thinkingmachines/Inkling-2-Mini) by Thinking Machines is also Apache-2.0. CUDA components retain NVIDIA's license terms.

The abliteration removes safety refusals; you are responsible for how you use the model. This is an unofficial community project, not endorsed by Thinking Machines.
