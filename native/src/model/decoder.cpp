// fast.FastDecoder.decode with fast.KernelOps: the fixed kernel sequence of one decode step for M
// consecutive tokens (48 visits: attention prologue mix (+ message gates), proj, attn, proj_conv, MLP
// prologue mix (+ gate), router, moe_up, moe_down; then the final read, norm and head).
#include "model/runtime.h"

#include "core/device.h"
#include "kernels/attn/attn.h"
#include "kernels/dense/dense.h"
#include "kernels/moe/moe.h"

#include <algorithm>
#include <stdexcept>

namespace ingine::model {
namespace dense = kernels::dense;
namespace moe = kernels::moe;
namespace attn = kernels::attn;

namespace {

dense::Matrix matrix(const Dense& d) {
  return dense::Matrix{d.w.data, static_cast<const float*>(d.scale.data), d.rows(), d.bscale.data};
}

const float* f32p(const Tensor& t) { return static_cast<const float*>(t.data); }
float* f32m(const Tensor& t) { return static_cast<float*>(t.data); }
__nv_bfloat16* bf16m(const Tensor& t) { return static_cast<__nv_bfloat16*>(t.data); }

// fast.proj_layout(full, single_input) as an int32 [S, 3] table (row_begin, row_end, input)
std::vector<int> proj_layout(bool full, bool single_input) {
  std::vector<int> t;
  for (int s = 0; s < num_segs(full); ++s) {
    t.push_back(SEG_BEGIN[s]);
    t.push_back(SEG_BEGIN[s + 1]);
    t.push_back(single_input ? 0 : SEG_INPUT[s]);
  }
  return t;
}

DeviceBuffer upload_ints(const std::vector<int>& v) {
  DeviceBuffer b(v.size() * sizeof(int));
  b.copy_from_host(v.data(), v.size() * sizeof(int));
  return b;
}

}  // namespace

Decoder::Decoder(const Weights& w, const Ablation& abl, State& st, bool pdl)
    : w_(w), abl_(abl), st_(st), pdl_(pdl), attn_ws_(attn::workspace_bytes(8)) {
  attn_ws_.fill(0);  // zeroed once; the kernel resets its counter itself
  segments_[0] = upload_ints(proj_layout(false, true));
  segments_[1] = upload_ints(proj_layout(false, false));
  full_segments_ = upload_ints(proj_layout(true, false));
}

Decoder::~Decoder() = default;

const std::int64_t* Decoder::block_table(const std::vector<const float*>& blocks) {
  auto it = tables_.find(blocks);
  if (it == tables_.end()) {
    // built on first use, i.e. in the eager warm-up before capture (KernelOps._block_ptrs)
    DeviceBuffer b(std::max<std::size_t>(blocks.size(), 1) * sizeof(void*));
    if (!blocks.empty()) b.copy_from_host(blocks.data(), blocks.size() * sizeof(void*));
    it = tables_.emplace(blocks, std::move(b)).first;
  }
  return static_cast<const std::int64_t*>(it->second.p);
}

void Decoder::decode(Buffers& b, const std::int64_t* pos, bool history, cudaStream_t stream) {
  const int m = b.m;
  const int H = HIDDEN;
  const bool pdl = pdl_;
  const Weights& w = w_;
  const Ablation& abl = abl_;
  State& st = st_;
  const std::size_t slab = std::size_t(m) * H;  // one [m, H] slab of a [G, m, H] buffer

  auto mix_params = [&](const std::vector<const float*>& blocks, const float* base, const float* write,
                        const float* abl_r, const float* abl_w, float* partial_out, const Tensor* queries,
                        const float* norm_w, __nv_bfloat16* out_bf16, float* out_f32) {
    dense::MixParams p;
    p.blocks = blocks.empty() ? nullptr : reinterpret_cast<const float* const*>(block_table(blocks));
    p.nblocks = int(blocks.size());
    p.base = base;
    p.write = write;
    // KernelOps.mix: the ablation applies to `write` only
    p.abl_r = write ? abl_r : nullptr;
    p.abl_w = write ? abl_w : nullptr;
    p.partial_out = partial_out;
    p.queries = queries ? f32p(*queries) : nullptr;
    p.Q = queries ? queries->ne[1] : 1;
    p.norm_w = norm_w;
    p.out_bf16 = out_bf16;
    p.out_f32 = out_f32;
    p.M = m;
    return p;
  };

  dense::embed(static_cast<const std::int64_t*>(b.token.data), m, w.embed.data,
               w.embed_scale.data ? f32p(w.embed_scale) : nullptr, f32p(w.embed_norm), f32m(b.B0), pdl, stream);
  int prev_v = -1;
  int tap = 0;
  const float* attn_dirs = f32p(abl.attn_dirs);
  const float* attn_w = f32p(abl.attn_w);
  const float* mlp_dirs = f32p(abl.mlp_dirs);
  const float* mlp_w = f32p(abl.mlp_w);
  __nv_bfloat16* taps = bf16m(b.taps);

  for (int v = 0; v < NUM_VISITS; ++v) {
    const int l = VISITS[v].layer, p = VISITS[v].pass;
    const Layer& layer = w.layers[l];
    const bool loop = is_loop_layer(l);
    const bool msgs = loop && p > 0;
    const bool boundary = l % BLOCK == 0;
    const std::vector<const float*> blocks = b.blocks(l, p);

    // ---- hidden entering this visit (the previous MLP write applied), stored where it is a block
    float* hid;
    const float* base;
    const float* write;
    if (v == 0) {
      hid = f32m(b.B0), base = f32p(b.B0), write = nullptr;
    } else if (l == LOOP_START && p > 0) {
      hid = f32m(b.B1), base = f32p(b.B1), write = nullptr;
    } else {
      hid = l == 6 ? f32m(b.B1) : l == 12 ? f32m(b.B2[p]) : l == 18 ? f32m(b.E[2]) : f32m(b.hid);
      base = f32p(b.part), write = f32p(b.y2);
    }
    const int pv = prev_v >= 0 ? prev_v : 0;
    const float* r_prev = mlp_dirs + std::size_t(pv) * H;
    const float* w_prev = mlp_w + pv;
    dense::MixParams msg_mix;  // message visits: launched together with the attention read (independent of it)
    if (msgs) {
      const int i = l - LOOP_START;
      const float* src_base = i == 0 ? f32p(b.part) : f32p(b.E[p - 1]);
      const float* src_write = i == 0 ? f32p(b.y2) : nullptr;
      // messages from pass p-1: its layer-12 block and its final hidden (E[p-1], completed at i == 0)
      const Tensor mq = rm(f32m(w.message_queries) + std::size_t(i) * 4 * H, DType::FP32, {4, H});
      msg_mix = mix_params({f32p(b.B2[p - 1])}, src_base, src_write, r_prev, w_prev,
                           i == 0 ? f32m(b.E[p - 1]) : nullptr, &mq, nullptr, nullptr, f32m(b.msgs));
      if (i == 0) write = nullptr;  // pass p-1's last write went into E[p-1]; layer 6 restarts from B1
    }
    // ---- attention read(s); at a tap visit the same launch writes b.taps[:, tap] = bf16(hid)
    const Tensor* queries = layer.attn_query.data ? &layer.attn_query : nullptr;
    float* partial_out = hid != base ? hid : nullptr;
    dense::MixParams ap =
        msgs ? mix_params(blocks, base, write, r_prev, w_prev, partial_out, queries, nullptr, nullptr, f32m(b.mixf))
             : mix_params(blocks, base, write, r_prev, w_prev, partial_out, l > 0 ? queries : nullptr,
                          f32p(layer.attn_norm_mix), bf16m(b.xin), nullptr);
    if (tap < 3 && v == TAP_VISITS[tap]) {
      ap.new_bf16 = taps + std::size_t(tap) * H;
      ap.new_bf16_stride = std::int64_t(NUM_TAPS) * H;
      ++tap;
    }
    if (msgs) dense::mix_pair(ap, msg_mix, pdl, stream);
    else dense::mix(ap, pdl, stream);
    if (msgs)  // sliding layers mix one query: all three gates read b.mixf[0]
      dense::message_gates(f32p(b.mixf), !layer.full, f32p(b.msgs), 3, m,
                           static_cast<const __nv_bfloat16*>(layer.qkv_gate_down.data), layer.qkv_gate_forget16.data,
                           layer.qkv_gate_gain16.data, false, f32p(layer.attn_norm3), bf16m(b.xin), nullptr, pdl,
                           stream);
    // ---- attention
    const bool single = !layer.full && !msgs;
    const DeviceBuffer& seg = layer.full ? full_segments_ : segments_[single ? 0 : 1];
    const Tensor& proj_out = layer.full ? b.proj_full : b.proj_sliding;
    const int R = proj_rows(layer.full);
    dense::proj(bf16m(b.xin), m, matrix(layer.wproj), static_cast<const int*>(seg.p), num_segs(layer.full),
                f32m(proj_out), pdl, stream);
    // b.hist_k/hist_v[v] (K and V segments of proj_out) are copied by the MLP-side mix below
    attn::DecodeArgs a;
    a.proj = f32p(proj_out);
    a.M = m;
    a.q_scale = f32p(layer.q_scale);
    a.r_scale = f32p(layer.r_scale);
    a.rel_proj = f32p(layer.rel_proj);
    a.conv_k_state = st.conv_k_at(v);
    a.conv_v_state = st.conv_v_at(v);
    a.pos = pos;
    a.workspace = attn_ws_.p;
    a.workspace_size = attn_ws_.bytes;
    a.out = bf16m(b.attn_out);
    if (layer.full) {
      const int fi = FULL_INDEX[v];
      a.k_conv = f32p(layer.k_static);
      a.v_conv = f32p(layer.v_static);
      a.k_cache = st.k_full_at(fi).data;
      a.v_cache = st.v_full_at(fi).data;
      a.cache_fp8 = st.kv_fp8;
      a.cache_q4 = st.kv_q4;
      a.cache_rows = st.max_ctx;
      a.kpos = nullptr;
      a.window = 0;
      a.log_scaling = true;
    } else {
      const int ri = RING_INDEX[v];
      a.k_conv = f32p(layer.k_conv);
      a.v_conv = f32p(layer.v_conv);
      a.k_cache = st.k_ring_at(ri).data;
      a.v_cache = st.v_ring_at(ri).data;
      a.cache_fp8 = false;
      a.cache_q4 = false;
      a.cache_rows = WINDOW;
      a.kpos = st.ring_pos_at(ri);
      a.window = WINDOW;
      a.log_scaling = false;
    }
    attn::attn_decode(a, stream);
    dense::proj_conv(bf16m(b.attn_out), m, matrix(layer.wo), f32p(layer.attn_conv), st.conv_attn_at(v), f32m(b.y),
                     history ? f32m(b.hist_attn) + std::size_t(v) * m * H : nullptr, pdl, stream);
    // ---- MLP read: this visit's attention write, ablated, on top of hid (or a fresh block)
    std::vector<const float*> mlp_blocks = blocks;
    if (boundary) mlp_blocks.push_back(hid);
    const float* mbase = boundary ? nullptr : hid;
    dense::MixParams mp =
        msgs ? mix_params(mlp_blocks, mbase, f32p(b.y), attn_dirs + std::size_t(v) * H, attn_w + v, f32m(b.part),
                          &layer.mlp_query, nullptr, nullptr, f32m(b.mixm))
             : mix_params(mlp_blocks, mbase, f32p(b.y), attn_dirs + std::size_t(v) * H, attn_w + v, f32m(b.part),
                          &layer.mlp_query, f32p(layer.mlp_norm1), bf16m(b.xm), nullptr);
    if (history) {  // b.hist_k/hist_v[v] = proj_out[:, K/V segment]
      int c = 0;
      for (auto [dst, s] : {std::pair{&b.hist_k, SEG_K}, std::pair{&b.hist_v, SEG_V}})
        mp.copies[c++] = dense::MixRowCopy{f32p(proj_out) + SEG_BEGIN[s], R, f32m(*dst) + std::size_t(v) * m * KV_DIM,
                                           KV_DIM, KV_DIM};
    }
    dense::mix(mp, pdl, stream);
    if (msgs)
      dense::message_gates(f32p(b.mixm), false, f32p(b.msgs) + 3 * slab, 1, m,
                           static_cast<const __nv_bfloat16*>(layer.mlp1_gate_down.data), layer.mlp1_gate_forget16.data,
                           layer.mlp1_gate_gain16.data, false, f32p(layer.mlp_norm1), bf16m(b.xm), nullptr, pdl,
                           stream);
    dense::router(bf16m(b.xm), m, layer.router_bf16.data, false, f32p(layer.router_bias),
                  static_cast<int*>(b.idx.data), f32m(b.wts), pdl, stream);
    const Experts& ex = layer.experts;
    const moe::Format fmt = ex.fmt == ExpertFormat::NVFP4 ? moe::Format::NVFP4 : moe::Format::MXFP8;
    moe::moe_up(bf16m(b.xm), m, static_cast<const int*>(b.idx.data), ex.w13.data, ex.w13_scale.data,
                f32p(ex.w13_factor), fmt, bf16m(b.act), pdl, stream);
    // with PDL, moe_down prefetches weights before waiting: valid because moe_up always precedes it
    moe::moe_down(bf16m(b.act), m, static_cast<const int*>(b.idx.data), f32p(b.wts), ex.w2.data, ex.w2_scale.data,
                  f32p(ex.w2_factor), fmt, f32p(layer.mlp_conv), st.conv_mlp_at(v), f32m(b.y2),
                  history ? f32m(b.hist_mlp) + std::size_t(v) * m * H : nullptr, pdl, stream);
    prev_v = v;
  }
  // ---- final read; the same launch writes b.taps[:, 3] = b.xf
  const Tensor final_norm1 = rm(w.final_norm.data, DType::FP32, {1, H});
  dense::MixParams fp = mix_params({f32p(b.B0), f32p(b.B1), f32p(b.B2[2]), f32p(b.E[2])}, f32p(b.part), f32p(b.y2),
                                   mlp_dirs + std::size_t(prev_v) * H, mlp_w + prev_v, nullptr, &w.final_query,
                                   f32p(final_norm1), bf16m(b.xf), nullptr);
  fp.out_copy = taps + std::size_t(tap) * H;
  fp.out_copy_stride = std::int64_t(NUM_TAPS) * H;
  dense::mix(fp, pdl, stream);
  dense::head(bf16m(b.xf), m, matrix(w.unembed), f32m(b.logits), pdl, stream);
}

}  // namespace ingine::model
