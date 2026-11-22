# BlackInk — Inkling-2-Mini abliterated (16 GB Blackwell Edition)

**BlackInk** is a 16 GB Blackwell-optimized fork of **ingine**, serving an abliterated
[thinkingmachines/Inkling-2-Mini](https://huggingface.co/thinkingmachines/Inkling-2-Mini) on a single **16 GB NVIDIA
Blackwell GPU (RTX 5060 Ti 16 GB, RTX 5070 Ti, RTX 5080, or RTX 5090)** with **131,072 (128K) tokens of Q4 context**,
**Full-NVFP4 routed experts**, **DSpark speculative decoding**, an **OpenAI-compatible API**, and an **integrated Web
Chat Studio** (`/` or `/chat`).

### What BlackInk adds over upstream `ingine`:
* **On-the-fly `MXFP8 -> NVFP4` Expert Transcoding (`--full-nvfp4`)**: Layers 20–23 are transcoded on the GPU at startup
  from 8-bit `MXFP8` to 4-bit `NVFP4` (matching layers 0–19) and the `embed` table is quantized to `FP8 E4M3` with
  per-row scales + deduplicated router/gate weights, saving **~2.32 GiB of VRAM** without altering the checkpoint on
  disk.
* **4-bit Full-Attention KV Cache (`--kv-q4`)**: Compresses the 8 long-range full-attention visits to 4-bit symmetric
  non-linear quantization (`0.5 byte/element`), fitting **131,072 (128K) tokens of context in just 1.0 GiB of VRAM**
  (while keeping the 40 sliding-window visits in exact `BF16`).
* **16 GB System RAM & VRAM Friendly**: Uses a lean 64 MiB pinned staging pool for streaming layer-by-layer loading and
  includes an automatic CUDA Unified Memory fallback if VRAM becomes tight.
* **Built-in Web Chat UI**: Open `http://<server-ip>:8003/` in any browser for a dark-mode chat interface with streaming
  `<think>` reasoning blocks, `reasoning_effort` selector, live `tok/s` telemetry, and multi-conversation history.

The installer sets everything up and downloads the ready-to-run model from Hugging Face,
[satellitedown/Inkling-2-Mini-abliterated-ingine](https://huggingface.co/satellitedown/Inkling-2-Mini-abliterated-ingine)
(20.2 GB, pinned to one revision and checked file by file).

## Run

You need **Linux x86_64, an NVIDIA GPU with compute capability 12.0 and at least 32 GB of memory (RTX 5090), an
NVIDIA driver for CUDA 13 (580 or newer), a C++20 compiler (GCC 12 to 16; the menu offers to install one), and
about 24 GB of free disk** (20.2 GB of model files; the private CUDA toolkit, the build tools and the build take
about 1.2 GB, uv's download cache about 1 GB more). The RTX PRO 6000, 5000 and 4500 Blackwell cards are also
compute capability 12.0 but untested. Other GPUs are not supported: the kernels are built for `sm_120a` and decode
the routed experts with Blackwell's FP4 conversion instructions. The installer does not install or change GPU
drivers. The server is a C++/CUDA program: it needs no Python or PyTorch to run.

Install with one command. It clones this repository to `~/blackink` and opens its menu:

```bash
curl -fsSL https://raw.githubusercontent.com/zaptize-dev/blackink/main/get.sh | bash
```

Choose **1** to install (build tools, CUDA toolkit, build of the C++ engine and model download), then **5** to add
the `blackink` / `ingine` command and `blackink start` to run the server. Run the same command again any time to get back to the
menu.
To install somewhere else, set the folder:
`curl -fsSL https://raw.githubusercontent.com/zaptize-dev/blackink/main/get.sh | INGINE_DIR=/path/to/folder bash`.

Or do the same by hand:

```bash
git clone https://github.com/zaptize-dev/blackink
cd blackink
bash setup.sh
```

The menu:

1. **Install everything:** check the GPU, driver, tools and disk space, install CMake, Ninja and a private CUDA
   13.4 toolkit (with cuBLAS), build the C++ engine (`ingine-serve`, a few minutes) and download the model.
2. **Download / resume model:** download the pinned model files, or resume an interrupted download, and verify them.
3. **Start the server (foreground):** run `ingine serve` in this terminal; **Ctrl+C** stops it.
4. **Update:** pull this repository and rerun the install steps (see below). `bash setup.sh --update` does the same
   without the menu.
5. **Install the `ingine` command:** link `~/.local/bin/ingine` to `scripts/ingine` (make sure `~/.local/bin` is on
   your `PATH`).

Choose **0** to exit. Every step can be repeated: an interrupted install continues with option 1, an interrupted
download with option 2. `bash setup.sh --help` lists the options without installing anything.

**Updating** (`ingine update`, menu option 4 or `bash setup.sh --update`) fast-forwards this repository from
GitHub, then reruns the idempotent install steps: build tools, CUDA toolkit, an incremental rebuild of the engine,
and the model check. The model is downloaded again only when the pinned revision in
[runtime-manifest.json](runtime-manifest.json) changed; files that no longer match are moved aside to
`models/Inkling-2-Mini.previous-<time>/`, never deleted. Delete that folder once the new version works. A running
server keeps the old code until you restart it: `ingine stop && ingine start`. Installs from before the C++ engine
kept a Python environment in `.venv/` and compiled kernels in `engine/`; the update leaves them in place and
prints the command that deletes them (about 6 GB).

Everything lives inside the install folder:

| Path | Contents |
|---|---|
| `.tools/` | uv (when it was not installed already) and a small Python environment (`.tools/venv/`) with CMake, Ninja and `huggingface_hub`, used only by the installer |
| `.cuda-toolkit/` | private CUDA 13.4 toolkit (nvcc, runtime, cuBLAS) used to build ingine; the server loads its CUDA libraries from here, and the system CUDA is not used or changed |
| `native/build-install/` | the built server, `apps/serve/ingine-serve`, and its serving profile (`profile.env`, from [runtime-manifest.json](runtime-manifest.json)) |
| `models/Inkling-2-Mini/` | the model from Hugging Face: Inkling-2-Mini's files and the abliteration file (set `INGINE_MODEL_DIR` to use a download elsewhere) |
| `.run/` | pid file and server log, used only when there is no systemd user session (`ingine start` then runs the server with `nohup`) |

## Use

```bash
ingine start      # start the server in the background (systemd user unit inkling-ingine), wait until it is ready
ingine stop       # stop it and free the GPU
ingine status     # unit state, health, served model, GPU memory; exit code 0 ready, 1 starting, 3 stopped
ingine logs       # server log; ingine logs -f follows it
ingine serve      # run in the foreground instead; extra arguments go to the server, e.g. --max-ctx 131072
ingine update     # same as bash setup.sh --update
```

Without the symlink from option 5, run `scripts/ingine` from the install folder. Only one copy of the model fits
on a 32 GB card; stop other programs that hold a lot of GPU memory first. `ingine start` and `ingine stop` are
non-interactive, return when the server is ready or stopped, and send a desktop notification when `notify-send`
is available; if you like, bind them to keys in your desktop environment.

Settings come from the profile in [runtime-manifest.json](runtime-manifest.json) and can be overridden in
`~/.config/ingine/env` (shell assignments, read by every `ingine` command) or in the environment:

| Variable | Default | Meaning |
|---|---|---|
| `INGINE_MODEL_DIR` | `models/Inkling-2-Mini` in the install folder | model folder; point it at an existing download of `satellitedown/Inkling-2-Mini-abliterated-ingine` to reuse it |
| `INGINE_ABLATION` | `$INGINE_MODEL_DIR/inkling-2-mini-abliterated.safetensors` | abliteration file; a relative path is taken relative to the install folder; empty serves the original model (model id `inkling-2-mini`) |
| `INGINE_CTX` | 262144 | context length; lower it to leave GPU memory to other programs |
| `HOST`, `PORT` | 127.0.0.1, 8003 | bind address and port |
| `INGINE_EXTRA_ARGS` | empty | further server arguments, e.g. `--top-p 0.95` (sampling for requests that set none, also `--temperature`, `--top-k`), `--park-mb 0` (see below), `--spec-threshold 0.9` |
| `INGINE_START_TIMEOUT` | 300 | seconds `ingine start` waits for the server to report ready |

```bash
# ~/.config/ingine/env
INGINE_MODEL_DIR=/data/models/Inkling-2-Mini
INGINE_CTX=131072
```

`INGINE_MODEL_DIR` must hold the pinned revision; this checks it without downloading:
`.tools/venv/bin/python scripts/download_model.py --model-dir "$INGINE_MODEL_DIR" --verify-only`.

## Connect an application

Use the OpenAI-compatible API at **`http://127.0.0.1:8003/v1`**, model **`inkling-2-mini-abliterated`**
(`inkling-2-mini` when serving without the ablation). No API key is required, and **there is no
authentication**: keep the server on loopback. Requests run one at a time; up to 4 more wait in a queue and
further ones get HTTP 503. `GET /health` and `GET /v1/models` report the state.

A follow-up request only reads what is new in its conversation. Conversations that take turns (subagents, a
second client) are kept warm in spare GPU memory, by default up to 3 GiB and never the last 4 GiB that are free
after start-up (`--park-mb N` sets the budget in MiB, `0` turns it off); they are dropped when memory runs short.

`POST /v1/chat/completions` takes:

* `reasoning_effort`: `none`, `minimal`, `low`, `medium`, `high`, `xhigh`, `max` (the chat template maps them to
  0, 0.1, 0.2, 0.7, 0.9, 0.99, 0.99) or a number from 0 to 0.99; the template's default is 0.9.
* `enable_thinking: false` to answer without reasoning (also accepted in `chat_template_kwargs`).
* `tools` (function calling through the model's chat template), `stop`, `max_tokens` / `max_completion_tokens`,
  `temperature`, `top_p`, `top_k`, `seed`. Requests that omit the sampling fields use the model's defaults
  (temperature 1, no cutoff; 0 is greedy) or the server's `--temperature` / `--top-p` / `--top-k`.
* `stream: true` for server-sent events; reasoning arrives as `reasoning_content` deltas before the answer's
  `content` (non-streaming responses carry `message.reasoning_content`). `stream_options.include_usage` adds usage.

```bash
curl -s http://127.0.0.1:8003/v1/chat/completions -H 'Content-Type: application/json' -d '{
  "model": "inkling-2-mini-abliterated",
  "messages": [{"role": "user", "content": "Explain how hash maps handle collisions."}],
  "reasoning_effort": "low",
  "max_tokens": 4096
}'
```

```python
from openai import OpenAI

client = OpenAI(base_url="http://127.0.0.1:8003/v1", api_key="none")
stream = client.chat.completions.create(
    model="inkling-2-mini-abliterated",
    messages=[{"role": "user", "content": "Write a Python LRU cache with tests."}],
    reasoning_effort="medium",
    stream=True,
    # extra_body={"enable_thinking": False},  # answer without reasoning
)
for chunk in stream:
    delta = chunk.choices[0].delta
    print(getattr(delta, "reasoning_content", None) or delta.content or "", end="", flush=True)
```

## Oh My Pi (OMP)

Merge this provider into `~/.omp/agent/models.yml`, keeping existing providers:

```yaml
providers:
  ingine:
    baseUrl: http://127.0.0.1:8003/v1
    auth: none
    api: openai-completions
    models:
      - id: inkling-2-mini-abliterated
        name: Inkling 2 Mini abliterated 256K (ingine)
        contextWindow: 262144
        maxTokens: 32768
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

Start the server first (`ingine start`), then open OMP:

```bash
omp --model ingine/inkling-2-mini-abliterated
```

OMP's `minimal` level is sent as `none` (thinking effort 0). If you lower `INGINE_CTX`, lower `contextWindow` to
match.

## Results

### Abliteration

Refusals on held-out prompts the abliteration search never saw (greedy decoding):

| | original | abliterated |
|---|---|---|
| `harmful_test` (104), direct answer | 104/104 | 1/104 |
| JailbreakBench harmful behaviours (100), direct answer | 100/100 | 5/100 |
| `harmful_test` (48), reasoning first (effort low, 768-token budget) | 47/48 | 0/48 |

Damage against the original model: KL divergence 0.043 on its own answers to 64 held-out harmless prompts; on
8,192 tokens of technical prose, mean KL 0.034 and 91.2% top-1 agreement.

Through ingine (32 borderline prompts and a small capability set, greedy):

| | original | abliterated |
|---|---|---|
| refusals, answering directly | 15/32 | 1/32 (the reply notes it cannot produce images, then writes the scene) |
| refusals, reasoning first | 8/32 | 0/32 |
| math (20, reasoning, effort low) | 20/20 | 20/20 |
| code (12, tests executed) | 12/12 | 12/12 |
| multiple choice (12) | 12/12 | 12/12 |

Reasoning stability with the default profile, sampled at temperature 1 (GSM8K: 40 problems at two reasoning
efforts; AIME 2024: 15 problems, effort high, up to 32,768 tokens):

| | original | abliterated |
|---|---|---|
| GSM8K, effort low: correct | 40/40 | 39/40 |
| GSM8K, effort high: correct | 40/40 | 40/40 |
| AIME 2024: correct | 13/15 | 13/15 |
| AIME 2024: ran out of tokens | 2/15 | 2/15 |
| reasoning that loops (95 samples) | 0 | 0 |

Repeating a request with the same seed returned the same text in 6 of 6 cases for each model. Raw data:
[`results/`](results).

### Speed

One RTX 5090, one request, default profile, tokens per second while generating. "chat" explains hash-map
collisions and "code" writes an LRU-cache module with tests (short prompts, up to 1024 tokens, median of 3 runs);
"coding agent" replays 24 recorded agent requests (2K-39K-token prompts with tools):

| | chat | code | coding agent |
|---|---:|---:|---:|
| temperature 1 (the default) | 489 | 565 | 502 |
| greedy (temperature 0) | 486 | 568 | 524 |

The default profile runs DSpark speculative decoding: a small drafter proposes tokens and the full model checks
every one, so it changes the speed, not the output distribution. Predictable text goes fastest: in the agent replay,
tool calls run at 688 tokens/s, reasoning and prose at about 440; text outside English and code gains the least.

A new prompt is read at 10,900 tokens/s (20K tokens), 9,600 (64K), 8,100 (131K) and 7,000 (200K); a short prompt
takes 0.09 s to the first token. The server holds 23.8 GiB of GPU memory after start-up and no more while reading
a 254K-token prompt; conversations kept warm add up to 3 GiB.

### Agent loops

Twelve recorded Oh My Pi sessions (230 requests, prompts of 2K-89K tokens, 6.9M prompt tokens in total), replayed
in order (greedy, up to 400 tokens per answer): 214 of the 230 requests continue a conversation the server still
holds and read only what is new, so all 230 together wait 50 s for their first tokens; decoding runs at 563
tokens/s. Two of these sessions taking turns, as an agent and its subagent do (67 requests), wait 8.5 s in total.

### Long context

The default profile serves **262,144 tokens** of context (the checkpoint supports 1M). The long-range attention
layers keep their cache in FP8, 16 KiB per token (4 GiB at 262,144 tokens, allocated at start-up), so that it fits
beside the model in 32 GB; the sliding-window layers keep their last 1,024 tokens in BF16. A code haystack with a
passphrase hidden in the middle:

| | 131,443-token prompt | 254,019-token prompt |
|---|---:|---:|
| reading the prompt | 16.4 s (8,000 tokens/s) | 40.9 s (6,200 tokens/s) |
| decoding afterwards | 354 tokens/s | 258 tokens/s |

The passphrase was found in both.

## License and credits

This repository (ingine, tools, installer and the abliteration file) is licensed under **Apache-2.0**, see
[LICENSE](LICENSE). The base model, [Inkling-2-Mini](https://huggingface.co/thinkingmachines/Inkling-2-Mini) by
Thinking Machines, is also Apache-2.0; its files are redistributed unchanged in the Hugging Face repository
[satellitedown/Inkling-2-Mini-abliterated-ingine](https://huggingface.co/satellitedown/Inkling-2-Mini-abliterated-ingine),
not in this Git repository. CUDA components keep NVIDIA's license terms.

The search follows [Heretic](https://github.com/p-e-w/heretic) by p-e-w and the abliteration literature; the
reference model is built on the transformers Inkling2Loop pull request. See [NOTICE](NOTICE). The abliteration
removes the model's safety refusals: you are responsible for how you use it. This is an unofficial community
project, not endorsed by Thinking Machines.
