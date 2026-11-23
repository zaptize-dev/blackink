// model.Model.prefill for one chunk of T tokens, wired as fast.FastDecoder.decode (block states B0, B1, B2[p],
// E[p], partial / write pairs, message mixes) with T rows instead of M:
//   embed -> per visit: [message mix] -> attention read (mix) [-> message gates] -> taps -> projections
//   (cuBLASLt over the BF16 rows, FP8 rows dequantized per visit as weights.dequant) -> attn_prep (q/rel post-
//   processing, key/value convs, cache or sliding rows) -> rel_logits -> attention (+ gate) -> wo -> attention
//   conv -> MLP read (mix) [-> gate] -> router GEMM + top-8 -> grouped experts -> combine -> MLP conv
//   -> final read (mix) -> taps -> head of the last row.
// The mixes and the embedding are ModelCore's decode kernels (per-row CTAs, any row count). Ablation of the
// attention / MLP writes happens inside the mixes that consume them, as in the decode step.
#include "model/prefill.h"

#include "core/device.h"
#include "kernels/dense/dense.h"
#include "kernels/prefill/prefill.h"

#include <algorithm>
#include <stdexcept>
#include <string>

namespace ingine::model {
namespace pk = kernels::prefill;
namespace dense = kernels::dense;

namespace {

constexpr int H = HIDDEN;

const float* f32p(const Tensor& t) { return static_cast<const float*>(t.data); }
const __nv_bfloat16* bf16p(const Tensor& t) { return static_cast<const __nv_bfloat16*>(t.data); }
const __nv_bfloat16* bf16p(const void* p) { return static_cast<const __nv_bfloat16*>(p); }

// fast.Buffers.blocks
std::vector<const float*> blocks_of(int l, int p, const float* B0, const float* B1, float* const* B2, float* const* E) {
  if (l < 1) return {};
  if (l < 7) return {B0};
  if (l < 13) return {B0, B1};
  if (l < 19) return {B0, B1, B2[l < LOOP_END ? p : 2]};
  return {B0, B1, B2[2], E[2]};
}

}  // namespace

Prefiller::Prefiller(const Weights& w, const Ablation& abl, int max_chunk, cudaStream_t stream)
    : w_(w), abl_(abl), max_chunk_(max_chunk), stream_(stream), gemm_(std::make_unique<pk::Gemm>()) {
  if (max_chunk < 1) throw std::invalid_argument("Prefiller: max_chunk must be positive");
  const int L = WINDOW + max_chunk;
  kpos_rows_ = (L + pk::kAttnKeyTile - 1) / pk::kAttnKeyTile * pk::kAttnKeyTile;
  if (kpos_rows_ / pk::kAttnKeyTile > pk::kAttnMaxTiles)
    throw std::invalid_argument("Prefiller: max_chunk above " +
                                std::to_string(pk::kAttnMaxTiles * pk::kAttnKeyTile - WINDOW));
  const std::size_t C = std::size_t(max_chunk);
  auto alloc = [&](std::size_t bytes) {
    storage_.emplace_back(bytes);
    return storage_.back().p;
  };
  auto f32 = [&](std::size_t n) { return static_cast<float*>(alloc(n * 4)); };
  auto bf16 = [&](std::size_t n) { return static_cast<__nv_bfloat16*>(alloc(n * 2)); };
  auto i32 = [&](std::size_t n) { return static_cast<int*>(alloc(n * 4)); };
  B0_ = f32(C * H), B1_ = f32(C * H);
  for (int p = 0; p < PASSES; ++p) B2_[p] = f32(C * H), E_[p] = f32(C * H);
  hid_ = f32(C * H), part_ = f32(C * H), y_ = f32(C * H), y2_ = f32(C * H);
  xin_ = bf16(3 * C * H), xm_ = bf16(C * H), xf_ = bf16(C * H);
  mixf_ = f32(3 * C * H), msgs_ = f32(4 * C * H), mixm_ = f32(C * H);
  proj_ = bf16(C * PROJ_ROWS_FULL);
  q_ = bf16(C * HEADS * HEAD_DIM), rel16_ = bf16(C * HEADS * D_REL), rel_ = bf16(C * HEADS * REL_EXTENT);
  krows_ = bf16(std::size_t(kpos_rows_) * KV_DIM), vrows_ = bf16(std::size_t(kpos_rows_) * KV_DIM);
  kpos_ = i32(kpos_rows_);
  bounds_ = static_cast<int4*>(alloc(std::size_t(kpos_rows_ / pk::kAttnKeyTile) * sizeof(int4)));
  attn_out_ = bf16(C * HEADS * HEAD_DIM), wo_out_ = bf16(C * H);
  g16_ = bf16(C * GATE_RANK);
  // message-gate scratch lives in buffers dead while the gates run: gin_ in rel_ (written by rel_logits after
  // the attention-side gates, dead before the MLP-side gate), fpre_/gpre_ in down_ (written by the experts after
  // both gates)
  static_assert(3 * 2 * HIDDEN * 2 <= HEADS * REL_EXTENT * 2 && 2 * HIDDEN * 4 <= TOP_K * HIDDEN * 2);
  router_logits_ = f32(C * EXPERTS), wts_ = f32(C * TOP_K);
  idx_ = i32(C * TOP_K);
  counts_ = i32(EXPERTS), tiles_n_ = i32(1), pair_token_ = i32(C * TOP_K), pair_dst_ = i32(C * TOP_K);
  tiles_ = static_cast<int4*>(alloc(std::size_t(pk::moe_tiles_capacity(max_chunk)) * sizeof(int4)));
  act_ = bf16(C * TOP_K * INTER), down_ = bf16(C * TOP_K * H), moe_sum_ = bf16(C * H);
  fp4_ws_ = alloc(pk::moe_fp4_workspace_bytes(max_chunk));
  gin_ = rel_;
  fpre_ = reinterpret_cast<float*>(down_), gpre_ = fpre_ + C * H;
  wscratch_ = w.dense_fp8 ? bf16(std::max(std::size_t(PROJ_ROWS_FULL) * H, std::size_t(H) * HEADS * HEAD_DIM)) : nullptr;
  ids_ = static_cast<std::int64_t*>(alloc(C * 8));
  for (auto& b : storage_) b.fill(0);  // e.g. gathered sliding rows past ring + chunk: never keys, kept finite
}

Prefiller::~Prefiller() = default;

std::size_t Prefiller::device_bytes() const {
  std::size_t n = 0;
  for (const auto& b : storage_) n += b.bytes;
  return n;
}

const float* const* Prefiller::block_table(const std::vector<const float*>& blocks) {
  auto it = tables_.find(blocks);
  if (it == tables_.end()) {
    DeviceBuffer b(std::max<std::size_t>(blocks.size(), 1) * sizeof(void*));
    if (!blocks.empty()) b.copy_from_host(blocks.data(), blocks.size() * sizeof(void*));
    it = tables_.emplace(blocks, std::move(b)).first;
  }
  return static_cast<const float* const*>(it->second.p);
}

void Prefiller::run(State& st, const std::int32_t* tokens, int T, int pos0, float* logits, Tensor* taps) {
  if (T < 1 || T > max_chunk_) throw std::invalid_argument("Prefiller::run: T outside 1..max_chunk");
  if (pos0 < 0 || pos0 + T > st.max_ctx)
    throw std::invalid_argument("context " + std::to_string(pos0 + T) + " exceeds max_ctx " +
                                std::to_string(st.max_ctx));
  const Weights& w = w_;
  const cudaStream_t s = stream_;
  const std::size_t slab = std::size_t(T) * H;  // one [T, H] slab of a [G, T, H] buffer
  const float* attn_dirs = f32p(abl_.attn_dirs);
  const float* attn_w = f32p(abl_.attn_w);
  const float* mlp_dirs = f32p(abl_.mlp_dirs);
  const float* mlp_w = f32p(abl_.mlp_w);
  __nv_bfloat16* tap_out = taps ? static_cast<__nv_bfloat16*>(taps->data) : nullptr;

  auto mix = [&](const std::vector<const float*>& blocks, const float* base, const float* write, const float* abl_r,
                 const float* abl_w, float* partial_out, const float* queries, int Q, const float* norm_w,
                 __nv_bfloat16* out_bf16, float* out_f32) {
    dense::MixParams p;
    p.blocks = blocks.empty() ? nullptr : block_table(blocks);
    p.nblocks = int(blocks.size());
    p.base = base;
    p.write = write;
    p.abl_r = write ? abl_r : nullptr;
    p.abl_w = write ? abl_w : nullptr;
    p.partial_out = partial_out;
    p.queries = queries;
    p.Q = Q;
    p.norm_w = norm_w;
    p.out_bf16 = out_bf16;
    p.out_f32 = out_f32;
    p.M = T;
    dense::mix(p, false, s);
  };
  // B rows of a Dense matrix as BF16: the matrix itself, or its FP8/NVFP4 rows dequantized (weights.dequant)
  auto bf16_rows = [&](const Dense& d) -> const __nv_bfloat16* {
    if (!d.fp8() && !d.nvfp4()) return bf16p(d.w);
    dense::dequant_rows(d.w.data, f32p(d.scale), d.rows(), d.cols(), wscratch_, s, d.bscale.data);
    return wscratch_;
  };
  // message_gate for G gates: h_g, msg_g fp32 [T, H] -> out bf16 [G, T, H] = bf16(rms_norm(gated) * norm_w[g])
  auto gates = [&](const pk::GateRows& rows, int G, const Tensor& down, const Tensor& forget16, const Tensor& gain16,
                   const float* norm_w, __nv_bfloat16* out) {
    pk::gate_inputs(rows, G, T, gin_, s);
    for (int g = 0; g < G; ++g) {
      const __nv_bfloat16* gd = bf16p(down) + std::size_t(g) * GATE_RANK * 2 * H;
      gemm_->run(gin_ + std::size_t(g) * T * 2 * H, 2 * H, gd, 2 * H, g16_, GATE_RANK, false, T, GATE_RANK, 2 * H, s);
      gemm_->run(g16_, GATE_RANK, bf16p(forget16) + std::size_t(g) * H * GATE_RANK, GATE_RANK, fpre_, H, true, T, H,
                 GATE_RANK, s);
      gemm_->run(g16_, GATE_RANK, bf16p(gain16) + std::size_t(g) * H * GATE_RANK, GATE_RANK, gpre_, H, true, T, H,
                 GATE_RANK, s);
      pk::gate_apply(rows.h[g], rows.msg[g], fpre_, gpre_, norm_w + std::size_t(g) * H, T, out + std::size_t(g) * slab,
                     s);
    }
  };

  pk::widen_ids(tokens, T, ids_, s);
  dense::embed(ids_, T, w.embed.data, w.embed_scale.data ? f32p(w.embed_scale) : nullptr, f32p(w.embed_norm), B0_,
               false, s);
  int prev_v = -1;
  int tap = 0;

  for (int v = 0; v < NUM_VISITS; ++v) {
    const int l = VISITS[v].layer, p = VISITS[v].pass;
    const Layer& layer = w.layers[l];
    const bool msgs = is_loop_layer(l) && p > 0;
    const bool boundary = l % BLOCK == 0;
    const std::vector<const float*> blocks = blocks_of(l, p, B0_, B1_, B2_, E_);

    // ---- hidden entering this visit (the previous MLP write applied), stored where it is a block
    float* hid;
    const float* base;
    const float* write;
    if (v == 0) {
      hid = B0_, base = B0_, write = nullptr;
    } else if (l == LOOP_START && p > 0) {
      hid = B1_, base = B1_, write = nullptr;
    } else {
      hid = l == 6 ? B1_ : l == 12 ? B2_[p] : l == 18 ? E_[2] : hid_;
      base = part_, write = y2_;
    }
    const int pv = prev_v >= 0 ? prev_v : 0;
    const float* r_prev = mlp_dirs + std::size_t(pv) * H;
    const float* w_prev = mlp_w + pv;
    if (msgs) {
      const int i = l - LOOP_START;
      const float* src_base = i == 0 ? part_ : E_[p - 1];
      const float* src_write = i == 0 ? y2_ : nullptr;
      mix({B2_[p - 1]}, src_base, src_write, r_prev, w_prev, i == 0 ? E_[p - 1] : nullptr,
          f32p(w.message_queries) + std::size_t(i) * 4 * H, 4, nullptr, nullptr, msgs_);
      if (i == 0) write = nullptr;
    }
    // ---- attention read(s)
    const float* queries = layer.attn_query.data ? f32p(layer.attn_query) : nullptr;
    const int nq = layer.attn_query.data ? layer.attn_query.ne[1] : 1;
    float* partial_out = hid != base ? hid : nullptr;
    if (msgs) {
      mix(blocks, base, write, r_prev, w_prev, partial_out, queries, nq, nullptr, nullptr, mixf_);
      pk::GateRows rows;
      for (int g = 0; g < 3; ++g) rows.h[g] = mixf_ + (layer.full ? g : 0) * slab, rows.msg[g] = msgs_ + g * slab;
      gates(rows, 3, layer.qkv_gate_down, layer.qkv_gate_forget16, layer.qkv_gate_gain16, f32p(layer.attn_norm3),
            xin_);
    } else {
      mix(blocks, base, write, r_prev, w_prev, partial_out, l > 0 ? queries : nullptr, l > 0 ? nq : 1,
          f32p(layer.attn_norm_mix), xin_, nullptr);
    }
    if (tap < 3 && v == TAP_VISITS[tap]) {
      if (tap_out) pk::store_tap(hid, T, tap_out + std::size_t(tap) * H, std::int64_t(NUM_TAPS) * H, s);
      ++tap;
    }

    // ---- projections (fast.proj_layout): one GEMM for a single input, else per input segment
    const int R = proj_rows(layer.full);
    const __nv_bfloat16* wp = bf16_rows(layer.wproj);
    const bool single = !layer.full && !msgs;
    if (single) {
      gemm_->run(xin_, H, wp, H, proj_, R, false, T, R, H, s);
    } else {
      struct Part {
        int begin, end, input;
      };
      const Part parts[] = {{SEG_BEGIN[SEG_Q], SEG_BEGIN[SEG_K], 0},
                            {SEG_BEGIN[SEG_K], SEG_BEGIN[SEG_V], 1},
                            {SEG_BEGIN[SEG_V], SEG_BEGIN[SEG_K_DYN], 2},
                            {SEG_BEGIN[SEG_K_DYN], SEG_BEGIN[SEG_V_DYN], 1},
                            {SEG_BEGIN[SEG_V_DYN], SEG_BEGIN[NUM_SEGS], 2}};
      for (const Part& part : parts) {
        if (part.begin >= R) break;
        gemm_->run(xin_ + std::size_t(part.input) * slab, H, wp + std::size_t(part.begin) * H, H, proj_ + part.begin,
                   R, false, T, part.end - part.begin, H, s);
      }
    }

    // ---- attention
    pk::AttnPrepParams ap;
    ap.proj = proj_, ap.ld = R, ap.T = T, ap.pos0 = pos0, ap.full = layer.full;
    ap.q_scale = f32p(layer.q_scale), ap.r_scale = f32p(layer.r_scale);
    ap.k_w = f32p(layer.full ? layer.k_static : layer.k_conv);
    ap.v_w = f32p(layer.full ? layer.v_static : layer.v_conv);
    ap.conv_k = st.conv_k_at(v), ap.conv_v = st.conv_v_at(v);
    ap.q = q_, ap.rel = rel16_;
    pk::AttnParams at;
    at.q = q_, at.T = T, at.pos0 = pos0, at.rel = rel_, at.proj = proj_, at.ld = R, at.out = attn_out_;
    if (layer.full) {
      const int fi = FULL_INDEX[v];
      const std::size_t off = st.kv_q4 ? (std::size_t(pos0) * KV_DIM / 2)
                                       : (std::size_t(pos0) * KV_DIM * (st.kv_fp8 ? 1 : 2));
      void* kc = st.k_full_at(fi).data;
      void* vc = st.v_full_at(fi).data;
      ap.k_out = static_cast<char*>(kc) + off, ap.v_out = static_cast<char*>(vc) + off;
      ap.out_fp8 = st.kv_fp8, ap.out_q4 = st.kv_q4;
      pk::attn_prep(ap, s);
      at.k = kc, at.v = vc, at.kv_fp8 = st.kv_fp8, at.kv_q4 = st.kv_q4;
      at.kpos = nullptr, at.bounds = nullptr, at.L = pos0 + T;
      at.window = 0;
    } else {
      const int ri = RING_INDEX[v];
      auto* k_ring = static_cast<__nv_bfloat16*>(st.k_ring_at(ri).data);
      auto* v_ring = static_cast<__nv_bfloat16*>(st.v_ring_at(ri).data);
      std::int64_t* ring_pos = st.ring_pos_at(ri);
      // keys = cat(ring, chunk) as model.prefill reads them, before the ring takes the chunk's last rows
      pk::ring_gather(k_ring, v_ring, ring_pos, pos0, T, krows_, vrows_, kpos_, kpos_rows_, s);
      ap.k_out = krows_ + std::size_t(WINDOW) * KV_DIM, ap.v_out = vrows_ + std::size_t(WINDOW) * KV_DIM;
      ap.out_fp8 = false, ap.out_q4 = false;
      pk::attn_prep(ap, s);
      pk::ring_update(krows_, vrows_, pos0, T, k_ring, v_ring, ring_pos, s);
      pk::key_tile_bounds(kpos_, WINDOW + T, bounds_, s);
      at.k = krows_, at.v = vrows_, at.kv_fp8 = false, at.kv_q4 = false;
      at.kpos = kpos_, at.bounds = bounds_, at.L = WINDOW + T;
      at.window = WINDOW;
    }
    pk::conv_state_update(st.conv_k_at(v), proj_ + SEG_BEGIN[SEG_K], R, T, KV_DIM, s);
    pk::conv_state_update(st.conv_v_at(v), proj_ + SEG_BEGIN[SEG_V], R, T, KV_DIM, s);
    pk::rel_logits(rel16_, f32p(layer.rel_proj), T, rel_, s);
    pk::attention(at, s);

    // ---- output projection + post-attention conv
    const __nv_bfloat16* wo = bf16_rows(layer.wo);
    gemm_->run(attn_out_, HEADS * HEAD_DIM, wo, HEADS * HEAD_DIM, wo_out_, H, false, T, H, HEADS * HEAD_DIM, s);
    pk::static_conv(wo_out_, H, T, H, f32p(layer.attn_conv), st.conv_attn_at(v), y_, s);
    pk::conv_state_update(st.conv_attn_at(v), wo_out_, H, T, H, s);

    // ---- MLP read: this visit's attention write, ablated, on top of hid (or a fresh block)
    std::vector<const float*> mlp_blocks = blocks;
    if (boundary) mlp_blocks.push_back(hid);
    const float* mbase = boundary ? nullptr : hid;
    if (msgs) {
      mix(mlp_blocks, mbase, y_, attn_dirs + std::size_t(v) * H, attn_w + v, part_, f32p(layer.mlp_query), 1, nullptr,
          nullptr, mixm_);
      pk::GateRows rows;
      rows.h[0] = mixm_, rows.msg[0] = msgs_ + 3 * slab;
      gates(rows, 1, layer.mlp1_gate_down, layer.mlp1_gate_forget16, layer.mlp1_gate_gain16, f32p(layer.mlp_norm1),
            xm_);
    } else {
      mix(mlp_blocks, mbase, y_, attn_dirs + std::size_t(v) * H, attn_w + v, part_, f32p(layer.mlp_query), 1,
          f32p(layer.mlp_norm1), xm_, nullptr);
    }

    // ---- routed experts (model._moe_grouped) + post-MLP conv
    gemm_->run(xm_, H, bf16p(layer.router_bf16), H, router_logits_, EXPERTS, true, T, EXPERTS, H, s);
    CUDA_CHECK(cudaMemsetAsync(counts_, 0, EXPERTS * sizeof(int), s));
    pk::moe_topk(router_logits_, f32p(layer.router_bias), T, idx_, wts_, counts_, s);
    const pk::MoeWork work{counts_, tiles_n_, tiles_, pair_token_, pair_dst_};
    pk::moe_route(idx_, T, work, s);
    const Experts& ex = layer.experts;
    const pk::MoeExperts me{ex.fmt == ExpertFormat::NVFP4 ? 0 : 1, ex.w13.data, ex.w13_scale.data,
                            f32p(ex.w13_factor), ex.w2.data, ex.w2_scale.data, f32p(ex.w2_factor)};
    if (me.fmt == 0 && fp4_ws_ != nullptr) {
      pk::moe_fp4_gate_up(me, xm_, T, work, fp4_ws_, act_, s);
      pk::moe_fp4_down(me, act_, T, work, fp4_ws_, down_, s);
    } else {
      pk::moe_gate_up(me, xm_, T, work, act_, s);
      pk::moe_down(me, act_, T, work, down_, s);
    }
    pk::moe_combine(down_, wts_, T, moe_sum_, s);
    pk::static_conv(moe_sum_, H, T, H, f32p(layer.mlp_conv), st.conv_mlp_at(v), y2_, s);
    pk::conv_state_update(st.conv_mlp_at(v), moe_sum_, H, T, H, s);
    prev_v = v;
  }

  // ---- final read, taps, head of the last row
  mix({B0_, B1_, B2_[2], E_[2]}, part_, y2_, mlp_dirs + std::size_t(prev_v) * H, mlp_w + prev_v, nullptr,
      f32p(w.final_query), 1, f32p(w.final_norm), xf_, nullptr);
  if (tap_out)
    CUDA_CHECK(cudaMemcpy2DAsync(tap_out + std::size_t(NUM_TAPS - 1) * H, std::size_t(NUM_TAPS) * H * 2, xf_,
                                 std::size_t(H) * 2, std::size_t(H) * 2, T, cudaMemcpyDeviceToDevice, s));
  if (w.unembed.nvfp4()) {
    dense::head(xf_ + std::size_t(T - 1) * H, 1,
                dense::Matrix{w.unembed.w.data, f32p(w.unembed.scale), UNPADDED_VOCAB, w.unembed.bscale.data}, logits,
                false, s);
  } else {
    pk::head_logits(xf_ + std::size_t(T - 1) * H, w.unembed.w.data, f32p(w.unembed.scale), UNPADDED_VOCAB, logits, s);
  }
  CUDA_CHECK(cudaGetLastError());
  st.length = pos0 + T;
}

}  // namespace ingine::model
