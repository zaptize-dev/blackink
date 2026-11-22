"""Reference Inkling-2-Mini for one RTX 5090.

This is the transformers PR model (huggingface/transformers#51050) with one change: the routed experts
stay in their checkpoint encoding (NVFP4 in layers 0-19, MXFP8 in layers 20-23) and an expert is decoded
only when a token routes to it. The PR decodes every expert to BF16 at load time (~52 GB), which does not
fit in 32 GB. Decoding follows the PR's `Inkling2LoopDequantizeExperts` exactly (FP32 product, BF16 cast).

It also owns two research hooks used by the abliteration tools:

* `Visits` numbers the 48 decoder-layer visits of a forward pass (layers 0-5, three passes of 6-17,
  layers 18-23) so per-visit state can be keyed on them.
* `Ablation` removes a direction from the residual-stream writes, i.e. from the outputs of every
  `post_attention_convolution` and `post_mlp_convolution`: ``y <- y - w * r * (r . y)``. These outputs are
  the only writes into the attention-residual blocks besides the embedding, so with ``w = 1`` the residual
  stream never contains ``r``.
"""
from __future__ import annotations

import json
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from safetensors import safe_open

from transformers import AutoModelForMultimodalLM, AutoTokenizer
from transformers.models.inkling2_loop import modeling_inkling2_loop as M

E2M1 = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0)
NUM_VISITS = 48
LOOP_START, LOOP_END, LOOP_PASSES = 6, 18, 3


def visit_index(layer: int, loop_pass: int) -> int:
    """Execution-order index of decoder `layer` in loop pass `loop_pass` (0 outside the loop)."""
    if layer < LOOP_START:
        return layer
    if layer < LOOP_END:
        return LOOP_START + (LOOP_END - LOOP_START) * loop_pass + (layer - LOOP_START)
    return layer + (LOOP_PASSES - 1) * (LOOP_END - LOOP_START)


def visit_layer(visit: int) -> int:
    if visit < LOOP_START:
        return visit
    loop_visits = (LOOP_END - LOOP_START) * LOOP_PASSES
    if visit < LOOP_START + loop_visits:
        return LOOP_START + (visit - LOOP_START) % (LOOP_END - LOOP_START)
    return visit - (LOOP_PASSES - 1) * (LOOP_END - LOOP_START)


class PackedExperts(nn.Module):
    """Routed experts in their checkpoint encoding, decoded per routed expert.

    `w13` rows are interleaved as stored (row 2i is gate i, row 2i+1 is up i); the PR de-interleaves the
    decoded tensor instead, which gives the same products.
    """

    def __init__(self, config):
        super().__init__()
        self.num_experts = config.n_routed_experts
        self.hidden_dim = config.hidden_size
        self.intermediate_dim = config.moe_intermediate_size
        self.act_fn = M.Inkling2LoopSoftcappedSwiGLU(config)
        self.fmt: str | None = None

    def attach(self, fmt: str, tensors: dict[str, torch.Tensor]) -> None:
        self.fmt = fmt
        for name, tensor in tensors.items():
            self.register_buffer(name, tensor, persistent=False)
        if fmt == "nvfp4":
            self.register_buffer("lut", torch.tensor(E2M1, dtype=torch.float32, device=tensors["w13"].device),
                                 persistent=False)

    def decode(self, which: str, expert: int) -> torch.Tensor:
        weight, scale = getattr(self, which)[expert], getattr(self, which + "_scale")[expert]
        rows = weight.shape[0]
        if self.fmt == "nvfp4":
            codes = weight.long()
            values = torch.stack((self.lut[codes & 0xF], self.lut[codes >> 4]), dim=-1)
            values = values.view(rows, scale.shape[1], 16) * scale.float()[..., None]
            values = values * getattr(self, which + "_factor")[expert]
        else:
            values = weight.float().view(rows, scale.shape[1], 32) * torch.exp2(scale.float() - 127)[..., None]
        return values.reshape(rows, -1).to(torch.bfloat16)

    def forward(self, hidden_states: torch.Tensor, top_k_index: torch.Tensor, top_k_weights: torch.Tensor):
        final = torch.zeros_like(hidden_states, dtype=torch.float32)
        with torch.no_grad():
            expert_mask = F.one_hot(top_k_index, num_classes=self.num_experts).permute(2, 1, 0)
            expert_hit = torch.greater(expert_mask.sum(dim=(-1, -2)), 0).nonzero().flatten().tolist()
        for expert in expert_hit:
            top_k_pos, token_idx = torch.where(expert_mask[expert])
            gate_up = F.linear(hidden_states[token_idx], self.decode("w13", expert))
            current = self.act_fn(gate_up[:, 0::2], gate_up[:, 1::2])
            current = F.linear(current, self.decode("w2", expert))
            current = current * top_k_weights[token_idx, top_k_pos, None]
            final.index_add_(0, token_idx, current.to(final.dtype))
        return final.to(hidden_states.dtype)


def _skip_packed_init(original):
    def init_weights(self, module):
        if isinstance(module, PackedExperts):
            return
        original(self, module)
    return init_weights


_PATCHED = False


def _patch_transformers() -> None:
    global _PATCHED
    if _PATCHED:
        return
    # Expert checkpoint keys then match no model parameter, so the loader skips them without decoding.
    M.Inkling2LoopExperts = PackedExperts
    M.Inkling2LoopPreTrainedModel._init_weights = _skip_packed_init(M.Inkling2LoopPreTrainedModel._init_weights)
    _PATCHED = True


def load_experts(model, model_dir: Path, device: str) -> None:
    quant = json.loads((model_dir / "hf_quant_config.json").read_text())
    fmt_of = {}
    for group in quant["model"]["parameter_groups"]:
        for name in group["tensors"]:
            if ".mlp.experts." in name:
                fmt_of[name] = group["compute_dtype"]
    index = json.loads((model_dir / "model.safetensors.index.json").read_text())["weight_map"]
    layers = model.model.language_model.layers
    handles: dict[str, object] = {}
    for idx, layer in enumerate(layers):
        prefix = f"model.llm.layers.{idx}.mlp.experts."
        fmt = fmt_of[prefix + "w13_weight"]
        names = {"w13": "w13_weight", "w13_scale": "w13_weight_scale", "w2": "w2_weight", "w2_scale": "w2_weight_scale"}
        if fmt == "nvfp4":
            names |= {"w13_amax": "w13_weight_amax", "w13_den": "w13_weight_outer_scale_denominator",
                      "w2_amax": "w2_weight_amax", "w2_den": "w2_weight_outer_scale_denominator"}
        tensors = {}
        for key, suffix in names.items():
            file = index[prefix + suffix]
            if file not in handles:
                handles[file] = safe_open(str(model_dir / file), framework="pt", device=device)
            tensors[key] = handles[file].get_tensor(prefix + suffix)
        if fmt == "nvfp4":
            for which in ("w13", "w2"):
                tensors[which + "_factor"] = tensors.pop(which + "_amax").float() / tensors.pop(which + "_den").float()
        layer.mlp.experts.attach(fmt, tensors)


def load(model_dir: str | Path, device: str = "cuda"):
    """Returns (model, tokenizer). The model is in eval mode with BF16 dense weights; text-only use needs no
    image processor."""
    _patch_transformers()
    model_dir = Path(model_dir)
    model = AutoModelForMultimodalLM.from_pretrained(model_dir, dtype=torch.bfloat16, device_map=device)
    load_experts(model, model_dir, device)
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    return model, tokenizer


class Visits:
    """Tracks which of the 48 layer visits is executing."""

    def __init__(self, model):
        self.current = -1
        self.handles = []
        text = model.model.language_model
        for idx, layer in enumerate(text.layers):
            self.handles.append(layer.register_forward_pre_hook(self._make_pre_hook(idx), with_kwargs=True))

    def _make_pre_hook(self, layer_idx: int):
        def hook(module, args, kwargs):
            offset = kwargs.get("cache_offset", 0)
            loop_pass = 0 if offset == 0 else 1 + (offset - (24 - LOOP_START)) // (LOOP_END - LOOP_START)
            self.current = visit_index(layer_idx, loop_pass)
        return hook

    def remove(self):
        for h in self.handles:
            h.remove()


class Ablation:
    """Directional ablation of residual writes, per visit: ``y <- y - w * r * (r . y)``.

    `attn_dirs`/`mlp_dirs`: [48, hidden] unit vectors (FP32); `attn_w`/`mlp_w`: [48] weights. A weight of 0
    leaves that write untouched.
    """

    def __init__(self, model, visits: Visits):
        self.visits = visits
        hidden = model.config.text_config.hidden_size
        device = model.device
        self.attn_dirs = torch.zeros(NUM_VISITS, hidden, device=device)
        self.mlp_dirs = torch.zeros(NUM_VISITS, hidden, device=device)
        self.attn_w = torch.zeros(NUM_VISITS, device=device)
        self.mlp_w = torch.zeros(NUM_VISITS, device=device)
        self.enabled = False
        self.handles = []
        for layer in model.model.language_model.layers:
            self.handles.append(layer.post_attention_convolution.register_forward_hook(self._make_hook("attn")))
            self.handles.append(layer.post_mlp_convolution.register_forward_hook(self._make_hook("mlp")))

    def _make_hook(self, kind: str):
        def hook(module, args, output):
            if not self.enabled:
                return None
            v = self.visits.current
            w = (self.attn_w if kind == "attn" else self.mlp_w)[v]
            r = (self.attn_dirs if kind == "attn" else self.mlp_dirs)[v]
            y = output.float()
            y = y - w * (y @ r)[..., None] * r
            return y.to(output.dtype)
        return hook

    def set(self, attn_dirs, attn_w, mlp_dirs, mlp_w):
        self.attn_dirs.copy_(F.normalize(attn_dirs.float(), dim=-1))
        self.mlp_dirs.copy_(F.normalize(mlp_dirs.float(), dim=-1))
        self.attn_w.copy_(attn_w)
        self.mlp_w.copy_(mlp_w)
        self.enabled = True

    def remove(self):
        for h in self.handles:
            h.remove()


def chat_prompt(tokenizer, user: str, effort="none", system: str | None = None) -> str:
    messages = ([{"role": "system", "content": system}] if system else []) + [{"role": "user", "content": user}]
    return tokenizer.apply_chat_template(messages, tokenize=False, reasoning_effort=effort)
