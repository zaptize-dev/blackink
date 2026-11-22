---
license: apache-2.0
base_model: thinkingmachines/Inkling-2-Mini
base_model_relation: finetune
library_name: ingine
pipeline_tag: text-generation
tags:
- abliterated
- uncensored
- decensored
- inkling
- mixture-of-experts
- nvfp4
- fp8
- cuda
- ingine
---

# Inkling-2-Mini · abliterated · ingine

An **abliterated** [thinkingmachines/Inkling-2-Mini](https://huggingface.co/thinkingmachines/Inkling-2-Mini) for
**[ingine](https://github.com/satellitedown/inkling-2-mini-abliterated-ingine)**, a C++/CUDA inference engine
written for this model. It answers instead of refusing, whether it replies directly or reasons first, and keeps the
original's scores on the tested math, code and knowledge sets. On one RTX 5090 ingine decodes about 300 tokens/s,
490-570 tokens/s with DSpark speculative decoding, reads prompts at 6,200-10,900 tokens/s and serves 262,144
tokens of context through an OpenAI-compatible API.

This repository is ready to run: Inkling-2-Mini's files, unchanged, plus `inkling-2-mini-abliterated.safetensors`,
which ingine applies at run time. No weights were trained or modified.

> **Reduced safety filtering.** Refusal behaviour has been largely removed. The model can produce content
> that the original would decline. You are responsible for how you use it and for complying with
> the law and the license.

## Quick start

```sh
curl -fsSL https://raw.githubusercontent.com/satellitedown/inkling-2-mini-abliterated-ingine/main/get.sh | bash
```

In the menu, choose **1** to install (a private CUDA toolkit, a build of ingine of about a minute, and this
repository at a pinned revision with every file checked), then **5** for the `ingine` command, and run
`ingine start`. The API is at `http://127.0.0.1:8003/v1`, model `inkling-2-mini-abliterated`, with streaming,
tools, `reasoning_content`, `reasoning_effort` (`none` .. `max`) and `enable_thinking: false`:

```sh
curl -N http://127.0.0.1:8003/v1/chat/completions -H 'Content-Type: application/json' \
  -d '{"model":"inkling-2-mini-abliterated","stream":true,"messages":[{"role":"user","content":"Hello"}]}'
```

ingine runs one request at a time with a short queue; a follow-up only reads what is new in its conversation, and
conversations that take turns (subagents, a second client) stay warm in spare GPU memory. The
[GitHub README](https://github.com/satellitedown/inkling-2-mini-abliterated-ingine#readme) covers every option,
context and memory settings, updates and agent setup (including Oh My Pi).

## Requirements

* an NVIDIA GPU with compute capability 12.0 and at least 32 GB: RTX 5090 (RTX PRO 6000/5000/4500 Blackwell
  untested); ingine is built for `sm_120a` and uses Blackwell's FP4 instructions
* Linux x86_64 with an NVIDIA driver for CUDA 13 and a C++20 compiler (GCC 12-16)
* ~22 GB of disk: 20.2 GB for this repository, about 1.2 GB for the CUDA toolkit, build tools and the build

## Results

| | Original | **This release** |
|---|---:|---:|
| Refusals, `mlabonne/harmful_behaviors` test (104), direct answer | 104 | **1** |
| Refusals, JailbreakBench harmful behaviours (100), direct answer | 100 | **5** |
| Refusals, `harmful_behaviors` test (48), reasoning first | 47 | **0** |
| Refusals, 32 mild borderline prompts, direct / reasoning first | 15 / 8 | **1 / 0** (the one notes it cannot produce images, then writes the scene) |
| Math (20, reasoning) / Python (12, tests executed) / multiple choice (12) | 20/20, 12/12, 12/12 | **20/20, 12/12, 12/12** |
| Reasoning at temperature 1, correct: GSM8K (40) effort low / high, AIME 2024 (15) | 40/40, 40/40, 13/15 | **39/40, 40/40, 13/15** |
| Reasoning at temperature 1: samples that loop (95) | 0 | **0** |
| Agreement with the original on 8,192 tokens of technical docs: top-1 / mean KLD | 1 / 0 | **0.912 / 0.034** |
| KL divergence on the original's own answers to 64 held-out harmless prompts | 0 | **0.043** |

Greedy decoding on prompts the abliteration never saw, except the reasoning rows (sampled, through ingine's default
profile); raw data in `abliteration/results/`.

Speed on one RTX 5090 with the default profile (262,144-token context, FP8 KV cache), one request, tokens per
second while generating ("chat" explains hash-map collisions and "code" writes an LRU-cache module with tests:
short prompts, up to 1024 tokens; "agent" replays 24 recorded coding-agent requests with 2K-39K-token prompts):

| | chat | code | agent |
|---|---:|---:|---:|
| plain decoding, greedy | 306 | 301 | |
| DSpark, greedy | 486 | 568 | 524 |
| plain decoding, temperature 1 | 301 | 301 | 279 |
| DSpark, temperature 1 (the default) | 489 | 565 | 502 |

DSpark's drafts are checked by the full model, so it returns what plain decoding returns. A new prompt is read at
10,900 tokens/s (20K tokens), 8,000 (131K) and 6,200 (254K, 41 s); a follow-up request only reads what is new:
twelve recorded agent sessions (230 requests) need 50 s to first token in total. Decoding after 254K tokens of
context runs at 258 tokens/s, and a passphrase hidden in the middle of those 254K tokens of source code was found.

## Manual setup

With CMake 3.28 or newer, Ninja and a CUDA 13.4 toolkit with cuBLAS (e.g. in `/opt/cuda`):

```sh
hf download satellitedown/Inkling-2-Mini-abliterated-ingine --local-dir inkling-2-mini-abliterated
git clone https://github.com/satellitedown/inkling-2-mini-abliterated-ingine ingine
cd ingine/native && cmake --preset release && cmake --build --preset release
build/apps/serve/ingine-serve --engine --model-dir ../../inkling-2-mini-abliterated \
    --ablation ../../inkling-2-mini-abliterated/inkling-2-mini-abliterated.safetensors \
    --max-ctx 262144 --dense-fp8 --kv-fp8 --spec --host 127.0.0.1 --port 8003
```

Without `--ablation`, ingine serves the original model.

## Files

| Path | Contents | Size |
|---|---|---:|
| `model-00001-of-00001.safetensors`, `model.safetensors.index.json` | Inkling-2-Mini's weights (BF16 dense, NVFP4/MXFP8 routed experts), unchanged | 19.5 GB |
| `mtp.safetensors` | Inkling-2-Mini's DSpark drafter, unchanged | 636 MB |
| `inkling-2-mini-abliterated.safetensors` | the abliteration | 885 KB |
| `config.json`, `generation_config.json`, `hf_quant_config.json`, `processor_config.json`, tokenizer files, `tiktoken/`, `chat_template.jinja` | Inkling-2-Mini's configuration, tokenizer and chat template, unchanged | small |
| `README.md`, `LICENSE`, `NOTICE`, `SHA256SUMS` | this card, Apache-2.0, attributions, checksums of every file | small |
| `abliteration/` | the scripts and raw results behind the numbers above | small |

The abliteration file works only with an engine that applies it as ingine does, to every attention and MLP write
after its short convolution; folding it into the weights gives a different model, which was not measured.

## License

* Inkling-2-Mini: Thinking Machines, Apache-2.0
  ([thinkingmachines/Inkling-2-Mini](https://huggingface.co/thinkingmachines/Inkling-2-Mini), revision
  `6e7c8248f75265bdd4fbab5273304243b5d8c2cb`); its files are redistributed unchanged.
* The abliteration and ingine: Apache-2.0. The abliteration follows [Heretic](https://github.com/p-e-w/heretic) by
  p-e-w. Attributions are in `NOTICE`.

Unofficial community release, not endorsed by Thinking Machines; provided as is, without warranty. Not tested:
refusals with sampling or in other languages, broad benchmarks, other GPUs and other inference engines. Do not
expose the server unauthenticated to untrusted networks.
