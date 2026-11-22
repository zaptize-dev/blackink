// Dense decode kernels (the PyTorch extension's kernels ported to raw-pointer launchers; contracts in
// native/KERNELS.md, area `dense`). Every launcher runs on `stream`, never allocates or synchronises, reads token
// ids, positions and routing from device memory, and is CUDA-graph capturable. pdl = programmatic dependent
// launch: the kernel prefetches weights/constants, then waits for its predecessor before reading activations or
// state.
// M (tokens per call) is 1..8 (mix and embed: any M >= 1, one CTA per row). Shapes are row-major (Python
// order). Invalid arguments throw std::invalid_argument.
#pragma once

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <cstdint>

namespace ingine::kernels::dense {

// A weight matrix [rows, K]: BF16 rows (scale == nullptr) or float8_e4m3fn rows with fp32 per-row scales.
struct Matrix {
  const void* w = nullptr;
  const float* scale = nullptr;
  int rows = 0;
};

// proj: multi-segment GEMV. xin bf16 [3, M, H]; segments int32 device [nseg, 3] = (row_begin, row_end,
// input index); out fp32 [M, R] = bf16(w[r] . xin[seg(r).input, m]), R = w.rows.
void proj(const __nv_bfloat16* xin, int M, Matrix w, const int* segments, int nseg, float* out, bool pdl,
          cudaStream_t stream);

// proj_conv: attention output projection (w [H, 3072]) + post-attention conv with state update.
// x bf16 [M, 3072]; conv_w fp32 [H, 4]; conv_state fp32 [3, H]; out fp32 [M, H]; conv_hist fp32 [M, H]
// (BF16-rounded conv inputs, for speculative rollback) or nullptr.
void proj_conv(const __nv_bfloat16* x, int M, Matrix w, const float* conv_w, float* conv_state, float* out,
               float* conv_hist, bool pdl, cudaStream_t stream);

// head: out[m, v] = 30 tanh(bf16(w_v . x[m]) / 30), x bf16 [M, H], out fp32 [M, V], V = w.rows.
void head(const __nv_bfloat16* x, int M, Matrix w, float* out, bool pdl, cudaStream_t stream);

// gemv: plain GEMV (drafter o_proj / down_proj). x bf16 [M, K] (row stride K), out fp32 [M, R] =
// bf16(w[r] . x[m]), R = w.rows; K = 3072 or 9216; same PDL contract as proj.
void gemv(const __nv_bfloat16* x, int M, int K, Matrix w, float* out, bool pdl, cudaStream_t stream);

// fp8_rows: bf16 [rows, K] -> float8_e4m3fn [rows, K] (q) + fp32 scale [rows] = amax / 448.
void fp8_rows(const __nv_bfloat16* w, int rows, int K, void* q, float* scale, cudaStream_t stream);

// dequant_rows: weights.dequant: out = bf16(float(q) * scale[row]) for FP8 rows [rows, K].
void dequant_rows(const void* q, const float* scale, int rows, int K, __nv_bfloat16* out, cudaStream_t stream);

// router: x bf16 [M, H]; w [128, H] fp32 (w_fp32) or bf16 holding BF16 values; bias fp32 [128];
// idx int32 [M, 8] (descending score), wts fp32 [M, 8] = 8 s / (sum s + 1e-20).
void router(const __nv_bfloat16* x, int M, const void* w, bool w_fp32, const float* bias, int* idx, float* wts,
            bool pdl, cudaStream_t stream);

// mix: ablated residual update + attention residual over block states + RMS norm (KERNELS.md).
// Side copies: FP32 rows copied by the same launch (dst + m * dst_stride <- src + m * src_stride, `cols`
// floats; 16-byte aligned), so the decode step needs no separate memcpy node.
struct MixRowCopy {
  const float* src = nullptr;
  long long src_stride = 0;
  float* dst = nullptr;  // null: no copy
  long long dst_stride = 0;
  int cols = 0;
};
struct MixParams {
  const float* const* blocks = nullptr;  // device table of nblocks device pointers, each fp32 [M, H]
  int nblocks = 0;                       // 0..5
  const float* base = nullptr;           // [M, H] or null
  const float* write = nullptr;          // [M, H] or null
  const float* abl_r = nullptr;          // [H] or null (no ablation of `write`)
  const float* abl_w = nullptr;          // [1] (device), required with abl_r
  float* partial_out = nullptr;          // [M, H] or null
  const float* queries = nullptr;        // [Q, H]; null only for the layer-0 path (no blocks, Q = 1)
  int Q = 1;                             // 1..4
  const float* norm_w = nullptr;         // [Q, H], required with out_bf16
  __nv_bfloat16* out_bf16 = nullptr;     // [Q, M, H] or null
  float* out_f32 = nullptr;              // [Q, M, H] or null
  __nv_bfloat16* new_bf16 = nullptr;     // bf16(new), row m at new_bf16 + m * new_bf16_stride, or null
  long long new_bf16_stride = 0;
  __nv_bfloat16* out_copy = nullptr;     // Q == 1: copy of out_bf16, row m at out_copy + m * out_copy_stride
  long long out_copy_stride = 0;
  MixRowCopy copies[2];
  int M = 1;
};
void mix(const MixParams& p, bool pdl, cudaStream_t stream);
// mix_pair: `p` (1..4 blocks, Q = 1 or 3) and the independent message mix `msg` (one block, Q = 4) of a
// message visit in one launch; same results as two mix calls. Neither may read what the other writes.
void mix_pair(const MixParams& p, const MixParams& msg, bool pdl, cudaStream_t stream);

// message_gates (loop passes 1-2): h, msg fp32 [G, M, H] (G <= 3; h_shared: every gate reads h[0], i.e. h is
// [M, H]); w_down bf16 [G, 64, 2H]; w_forget/w_gain [G, H, 64] fp32 (fg_fp32) or bf16; norm_w fp32 [G, H];
// out_bf16 [G, M, H]; out_f32 [G, M, H] (gated, before the norm) or null.
void message_gates(const float* h, bool h_shared, const float* msg, int G, int M, const __nv_bfloat16* w_down,
                   const void* w_forget, const void* w_gain, bool fg_fp32, const float* norm_w,
                   __nv_bfloat16* out_bf16, float* out_f32, bool pdl, cudaStream_t stream);

// embed: out[m] = float(bf16(rms_norm(float(table[token[m]])) * norm_w)), bit-exact with model.Model.embed.
// token int64 [M] (device), table bf16 or fp8_e4m3 [V, H], table_scale fp32 [V] (null for BF16), norm_w fp32 [H], out fp32 [M, H].
void embed(const std::int64_t* token, int M, const void* table, const float* table_scale, const float* norm_w,
           float* out, bool pdl, cudaStream_t stream);
inline void embed(const std::int64_t* token, int M, const __nv_bfloat16* table, const float* norm_w, float* out,
                  bool pdl, cudaStream_t stream) {
  embed(token, M, static_cast<const void*>(table), nullptr, norm_w, out, pdl, stream);
}

}  // namespace ingine::kernels::dense
