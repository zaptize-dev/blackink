// model.State and fast.Buffers allocation.
#include "model/runtime.h"

#include "core/device.h"

#include <stdexcept>

namespace ingine::model {
namespace {

Tensor zeros(std::vector<DeviceBuffer>& storage, DType dt, std::initializer_list<std::int32_t> shape) {
  std::size_t n = 1;
  for (std::int32_t d : shape) n *= std::size_t(d);
  storage.emplace_back(n * dtype_size(dt));
  CUDA_CHECK(cudaMemset(storage.back().p, 0, storage.back().bytes));
  return rm(storage.back().p, dt, shape);
}

}  // namespace

Tensor State::k_ring_at(int ring) const {
  return rm(ptr<char>(k_ring) + std::size_t(ring) * WINDOW * KV_DIM * 2, DType::BF16, {WINDOW, KV_HEADS, HEAD_DIM});
}
Tensor State::v_ring_at(int ring) const {
  return rm(ptr<char>(v_ring) + std::size_t(ring) * WINDOW * KV_DIM * 2, DType::BF16, {WINDOW, KV_HEADS, HEAD_DIM});
}
std::int64_t* State::ring_pos_at(int ring) const { return ptr<std::int64_t>(ring_pos) + std::size_t(ring) * WINDOW; }
Tensor State::k_full_at(int full) const {
  if (kv_q4) {
    return rm(ptr<char>(k_full) + std::size_t(full) * max_ctx * (KV_DIM / 2), DType::U8,
              {max_ctx, KV_HEADS, HEAD_DIM / 2});
  }
  const std::size_t es = kv_fp8 ? 1 : 2;
  return rm(ptr<char>(k_full) + std::size_t(full) * max_ctx * KV_DIM * es, k_full.dtype, {max_ctx, KV_HEADS, HEAD_DIM});
}
Tensor State::v_full_at(int full) const {
  if (kv_q4) {
    return rm(ptr<char>(v_full) + std::size_t(full) * max_ctx * (KV_DIM / 2), DType::U8,
              {max_ctx, KV_HEADS, HEAD_DIM / 2});
  }
  const std::size_t es = kv_fp8 ? 1 : 2;
  return rm(ptr<char>(v_full) + std::size_t(full) * max_ctx * KV_DIM * es, v_full.dtype, {max_ctx, KV_HEADS, HEAD_DIM});
}

State make_state(int max_ctx, bool kv_fp8, bool kv_q4) {
  // the attention kernel keeps positions in int32 and caps full caches at 2^30 rows
  if (max_ctx < 1 || max_ctx > (1 << 30)) throw std::invalid_argument("max_ctx must be in 1..2^30");
  State st;
  st.max_ctx = max_ctx;
  st.kv_fp8 = kv_fp8 && !kv_q4;
  st.kv_q4 = kv_q4;
  const DType full = kv_q4 ? DType::U8 : (kv_fp8 ? DType::FP8_E4M3FN : DType::BF16);
  const int full_dim = kv_q4 ? (HEAD_DIM / 2) : HEAD_DIM;
  st.k_ring = zeros(st.storage, DType::BF16, {NUM_SLIDING_VISITS, WINDOW, KV_HEADS, HEAD_DIM});
  st.v_ring = zeros(st.storage, DType::BF16, {NUM_SLIDING_VISITS, WINDOW, KV_HEADS, HEAD_DIM});
  st.ring_pos = zeros(st.storage, DType::I64, {NUM_SLIDING_VISITS, WINDOW});
  CUDA_CHECK(cudaMemset(st.ring_pos.data, 0xff, st.storage.back().bytes));  // -1
  st.conv_k = zeros(st.storage, DType::FP32, {NUM_VISITS, CONV_K - 1, KV_DIM});
  st.conv_v = zeros(st.storage, DType::FP32, {NUM_VISITS, CONV_K - 1, KV_DIM});
  st.conv_attn = zeros(st.storage, DType::FP32, {NUM_VISITS, CONV_K - 1, HIDDEN});
  st.conv_mlp = zeros(st.storage, DType::FP32, {NUM_VISITS, CONV_K - 1, HIDDEN});
  st.k_full = zeros(st.storage, full, {NUM_FULL_VISITS, max_ctx, KV_HEADS, full_dim});
  st.v_full = zeros(st.storage, full, {NUM_FULL_VISITS, max_ctx, KV_HEADS, full_dim});
  st.length = 0;
  return st;
}

void reset_state(State& st, cudaStream_t stream) {
  CUDA_CHECK(cudaMemsetAsync(st.ring_pos.data, 0xff, st.ring_pos.bytes(), stream));
  for (Tensor* t : {&st.conv_k, &st.conv_v, &st.conv_attn, &st.conv_mlp})
    CUDA_CHECK(cudaMemsetAsync(t->data, 0, t->bytes(), stream));
  st.length = 0;
}

Buffers make_buffers(int m) {
  if (m < 1 || m > 8) throw std::invalid_argument("call width must be in 1..8");
  Buffers b;
  b.m = m;
  auto& s = b.storage;
  const int H = HIDDEN;
  b.token = zeros(s, DType::I64, {m});
  b.B0 = zeros(s, DType::FP32, {m, H});
  b.B1 = zeros(s, DType::FP32, {m, H});
  for (int p = 0; p < PASSES; ++p) {
    b.B2[p] = zeros(s, DType::FP32, {m, H});
    b.E[p] = zeros(s, DType::FP32, {m, H});
  }
  b.hid = zeros(s, DType::FP32, {m, H});
  b.part = zeros(s, DType::FP32, {m, H});
  b.y = zeros(s, DType::FP32, {m, H});
  b.y2 = zeros(s, DType::FP32, {m, H});
  b.xin = zeros(s, DType::BF16, {3, m, H});
  b.mixf = zeros(s, DType::FP32, {3, m, H});
  b.msgs = zeros(s, DType::FP32, {4, m, H});
  b.xm = zeros(s, DType::BF16, {1, m, H});
  b.mixm = zeros(s, DType::FP32, {1, m, H});
  b.proj_sliding = zeros(s, DType::FP32, {m, PROJ_ROWS_SLIDING});
  b.proj_full = zeros(s, DType::FP32, {m, PROJ_ROWS_FULL});
  b.attn_out = zeros(s, DType::BF16, {m, HEADS * HEAD_DIM});
  b.idx = zeros(s, DType::I32, {m, TOP_K});
  b.wts = zeros(s, DType::FP32, {m, TOP_K});
  b.act = zeros(s, DType::BF16, {m, TOP_K, INTER});
  b.xf = zeros(s, DType::BF16, {1, m, H});
  b.logits = zeros(s, DType::FP32, {m, UNPADDED_VOCAB});
  b.taps = zeros(s, DType::BF16, {m, NUM_TAPS, H});
  b.hist_attn = zeros(s, DType::FP32, {NUM_VISITS, m, H});
  b.hist_mlp = zeros(s, DType::FP32, {NUM_VISITS, m, H});
  b.hist_k = zeros(s, DType::FP32, {NUM_VISITS, m, KV_DIM});
  b.hist_v = zeros(s, DType::FP32, {NUM_VISITS, m, KV_DIM});
  return b;
}

std::vector<const float*> Buffers::blocks(int l, int p) const {
  auto f = [](const Tensor& t) { return static_cast<const float*>(t.data); };
  if (l < 1) return {};
  if (l < 7) return {f(B0)};
  if (l < 13) return {f(B0), f(B1)};
  if (l < 19) return {f(B0), f(B1), f(B2[l < LOOP_END ? p : 2])};
  return {f(B0), f(B1), f(B2[2]), f(E[2])};
}

}  // namespace ingine::model
