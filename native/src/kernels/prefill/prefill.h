// Prefill kernels (new for ingine): the pieces of model.Model.prefill for a chunk of T tokens that the decode
// kernels do not cover. Launchers are raw-pointer, run on `stream`, never allocate or synchronise. Shapes are
// row-major (Python order). H = 2304; positions of chunk row t are pos0 + t.
#pragma once

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>

namespace ingine::kernels::prefill {

inline constexpr int kHidden = 2304;
inline constexpr int kHeads = 24, kKvHeads = 8, kHeadDim = 128, kRel = 16, kRelExtent = 1024;
inline constexpr int kKvDim = kKvHeads * kHeadDim;  // 1024
inline constexpr int kInter = 1152, kExperts = 128, kTopK = 8;
inline constexpr int kAttnKeyTile = 64;    // keys per attention tile (tile bounds granularity)
inline constexpr int kAttnMaxTiles = 64;   // explicit-position key tiles per call (ring + chunk)

// ------------------------------------------------------------------------------------------------- GEMM

// cuBLASLt BF16 GEMM: C[M, N] (row-major, ldc) = A[M, K] (bf16, lda) . B[N, K]^T (bf16, ldb), FP32 accumulation
// and FP32 split-K reductions; C is bf16 (rounded once) or fp32.
class Gemm {
 public:
  Gemm();
  ~Gemm();
  Gemm(const Gemm&) = delete;
  Gemm& operator=(const Gemm&) = delete;
  void run(const __nv_bfloat16* A, int lda, const __nv_bfloat16* B, int ldb, void* C, int ldc, bool c_fp32, int M,
           int N, int K, cudaStream_t stream);

 private:
  struct Impl;
  Impl* impl_;
};

// ------------------------------------------------------------------------------------------- attention

// Keys/values of one sliding visit for the chunk: rows [0, 1024) are the ring before the chunk, rotated so
// row i holds slot (pos0 + i) % 1024 (position order for a filled ring), rows [1024, 1024 + T) the chunk.
// ring_gather fills rows [0, 1024) and the positions of every row (kpos int32 [kpos_rows], -1 = empty, padding
// rows -1); attn_prep writes the chunk rows.
void ring_gather(const __nv_bfloat16* k_ring, const __nv_bfloat16* v_ring, const int64_t* ring_pos, int pos0, int T,
                 __nv_bfloat16* k_rows, __nv_bfloat16* v_rows, int* kpos, int kpos_rows, cudaStream_t stream);

// model.prefill's ring update: the last min(T, 1024) chunk rows (rows 1024.. of k_rows/v_rows) into their slots
// (pos % 1024), ring_pos = pos.
void ring_update(const __nv_bfloat16* k_rows, const __nv_bfloat16* v_rows, int pos0, int T, __nv_bfloat16* k_ring,
                 __nv_bfloat16* v_ring, int64_t* ring_pos, cudaStream_t stream);

// Per key tile of kAttnKeyTile rows: int4 {min valid position, max valid position, all rows valid, any valid}.
void key_tile_bounds(const int* kpos, int L, int4* bounds, cudaStream_t stream);

struct AttnPrepParams {
  const __nv_bfloat16* proj = nullptr;  // [T, ld] BF16-rounded projections in SEG order (fast.proj_layout)
  int ld = 0;
  int T = 0, pos0 = 0;
  bool full = false;
  const float* q_scale = nullptr;  // [24, 128]
  const float* r_scale = nullptr;  // [24, 16]
  const float* k_w = nullptr;      // full: k_static [4, 1024] by lag; sliding: k_conv [1024, 4] conv1d order
  const float* v_w = nullptr;
  const float* conv_k = nullptr;   // [3, 1024] previous conv inputs (read only; see conv_state_update)
  const float* conv_v = nullptr;
  __nv_bfloat16* q = nullptr;      // out [T, 24, 128]
  __nv_bfloat16* rel = nullptr;    // out [T, 24, 16]
  void* k_out = nullptr;           // out [T, 8, 128] rows: bf16, fp8 e4m3 (out_fp8), or q4 (out_q4)
  void* v_out = nullptr;
  bool out_fp8 = false;
  bool out_q4 = false;
};
// model._qkv per token: q (scale, clip, tau, BF16), rel (scale, clip, tau, BF16), keys/values through the
// static (sliding) or dynamic (full) conv, BF16, key clip / value RMS norm, stored at the cache precision.
void attn_prep(const AttnPrepParams& p, cudaStream_t stream);

// rel_logits[t, h, d] = bf16(sum_j rel[t, h, j] * bf16(rel_proj[j, d])), rel bf16 [T, 24, 16], rel_proj fp32
// [16, 1024], out bf16 [T, 24, 1024].
void rel_logits(const __nv_bfloat16* rel, const float* rel_proj, int T, __nv_bfloat16* out, cudaStream_t stream);

struct AttnParams {
  const __nv_bfloat16* q = nullptr;  // [T, 24, 128]
  const void* k = nullptr;           // key rows [L, 8, 128]: bf16, fp8 e4m3 (kv_fp8), or q4 (kv_q4)
  const void* v = nullptr;
  bool kv_fp8 = false;
  bool kv_q4 = false;
  const int* kpos = nullptr;         // positions of the L keys (padded to tiles with -1), or null: key j at j
  const int4* bounds = nullptr;      // key_tile_bounds of kpos (with kpos)
  int L = 0;
  int T = 0, pos0 = 0;
  int window = 0;                    // 0: causal; else keys with 0 <= qpos - kpos < window
  const __nv_bfloat16* rel = nullptr;  // rel_logits [T, 24, 1024]
  const __nv_bfloat16* proj = nullptr; // [T, ld]: gate (sigmoid input) and sink logits in SEG order
  int ld = 0;
  __nv_bfloat16* out = nullptr;        // [T, 3072] = bf16(bf16(attention * sigmoid(lse - sink)) * sigmoid(gate))
};
// flex.attend + the output gate of model._visit: scores (q.k)/sqrt(128) + rel bias (distance < 1024), masked,
// softmax with the per-head sink folded in through the log-sum-exp. Q.K in BF16 MMA with FP32 accumulation
// (exact products), probabilities in FP16 (scaled by 2^15) times FP16 values (exact for BF16/E4M3 values in
// range), FP32 accumulation; FP32 softmax statistics.
void attention(const AttnParams& p, cudaStream_t stream);

// ----------------------------------------------------------------------------------------- convolutions

// model.static_conv on BF16-valued inputs: out[t, c] = (sum_j w[c, j] * ext[t + j] + x[t, c]) / sqrt(5),
// ext = [state (3 rows), x]; x bf16 [T, ld], w fp32 [C, 4], state fp32 [3, C], out fp32 [T, C].
void static_conv(const __nv_bfloat16* x, int ld, int T, int C, const float* w, const float* state, float* out,
                 cudaStream_t stream);

// state <- the last three rows of [state, x] (x bf16 [T, ld], C channels). Run after every reader of state.
void conv_state_update(float* state, const __nv_bfloat16* x, int ld, int T, int C, cudaStream_t stream);

// ---------------------------------------------------------------------------------------- message gates

// gin[g, t] = bf16([rms_norm(h[g, t]), rms_norm(msg[g, t])]) (bf16 [G, T, 2H]); h_g / msg_g fp32 [T, H].
struct GateRows {
  const float* h[3] = {};
  const float* msg[3] = {};
};
void gate_inputs(const GateRows& rows, int G, int T, __nv_bfloat16* gin, cudaStream_t stream);

// out[t] = bf16(rms_norm((1 - tanh(f)) * h + tanh(g) * msg) * norm_w); f, g fp32 [T, H] (the forget / gain
// products of the BF16 down projection).
void gate_apply(const float* h, const float* msg, const float* f, const float* g, const float* norm_w, int T,
                __nv_bfloat16* out, cudaStream_t stream);

// ------------------------------------------------------------------------------------------------- MoE

struct MoeWork {
  int* counts = nullptr;      // [128] (zeroed by moe_topk's caller)
  int* tiles_n = nullptr;     // [1]
  int4* tiles = nullptr;      // [moe_max_tiles(T)] {expert, first pair, pairs, 0}
  int* pair_token = nullptr;  // [T * 8] token of each expert-sorted pair
  int* pair_dst = nullptr;    // [T * 8] t * 8 + slot of each expert-sorted pair
};
// (token, expert) pairs per grouped-GEMM tile: 32 for small chunks (few pairs per expert: deeper weight
// prefetch), 128 otherwise. INGINE_MOE_TILE_ROWS (32 or 128) overrides it (benchmarks).
int moe_tile_rows(int T);
inline int moe_max_tiles(int T) { return (T * kTopK + moe_tile_rows(T) - 1) / moe_tile_rows(T) + kExperts; }
// tile-list capacity for every chunk size up to max_T
int moe_tiles_capacity(int max_T);

// Router selection on FP32 logits [T, 128] (x . router^T): s = sigmoid, top-8 of s + bias (descending),
// wts = 8 s / (sum s + 1e-20); counts[e] += hits.
void moe_topk(const float* logits, const float* bias, int T, int* idx, float* wts, int* counts, cudaStream_t stream);
// Expert-sorted pair lists and moe_tile_rows(T)-pair tiles from idx and counts.
void moe_route(const int* idx, int T, const MoeWork& w, cudaStream_t stream);

struct MoeExperts {
  int fmt = 0;  // 0 NVFP4, 1 MXFP8 (weights.py layouts)
  const void* w13 = nullptr;
  const void* w13_scale = nullptr;
  const float* w13_factor = nullptr;
  const void* w2 = nullptr;
  const void* w2_scale = nullptr;
  const float* w2_factor = nullptr;
};
// act[pair] = bf16(softcap(g, 4) * sigmoid(g) * softcap(u, 25)), g/u = bf16(W13_e . x[token]) (experts decoded
// exactly as model.decode_experts). x bf16 [T, H], act bf16 [T * 8, 1152] by sorted pair.
void moe_gate_up(const MoeExperts& e, const __nv_bfloat16* x, int T, const MoeWork& w, __nv_bfloat16* act,
                 cudaStream_t stream);
// down[t * 8 + slot] = bf16(W2_e . act[pair]), bf16 [T * 8, H].
void moe_down(const MoeExperts& e, const __nv_bfloat16* act, int T, const MoeWork& w, __nv_bfloat16* down,
              cudaStream_t stream);
// out[t] = bf16(sum over slots in order of down[t, slot] * wts[t, slot]), bf16 [T, H].
void moe_combine(const __nv_bfloat16* down, const float* wts, int T, __nv_bfloat16* out, cudaStream_t stream);

// ------------------------------------------------------------------------------------------------ misc

// int32 -> int64 token ids.
void widen_ids(const int32_t* in, int T, int64_t* out, cudaStream_t stream);
// out[t * out_stride] = bf16(x[t]) per row, x fp32 [T, H] (DSpark taps).
void store_tap(const float* x, int T, __nv_bfloat16* out, int64_t out_stride, cudaStream_t stream);
// model.logits for one row: out[v] = 30 tanh(bf16(w_v . x) / 30), w bf16 [V, H] or FP8 rows dequantized
// as weights.dequant (bf16(fp8 * scale[v])); x bf16 [H].
void head_logits(const __nv_bfloat16* x, const void* w, const float* scale, int V, float* out, cudaStream_t stream);

}  // namespace ingine::kernels::prefill
