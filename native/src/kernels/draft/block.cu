// DSpark draft block: parameters, norms, RoPE and the context commit (dspark.py mix_rmsnorm / rms / rope /
// Drafter._commit / the residual stream of _draft). The block attention is in attention.cu.
//
// Kernels of the commit/draft chains are PDL launches (launch_pdl): pdl_trigger() first, pdl_wait() before
// any read of data another kernel writes. Pointers to such data are not __restrict__ (a const __restrict__
// load may be scheduled above the wait). Norm weights are cold in DRAM when a round starts (a verify call
// streams gigabytes in between): the row kernels load them before the wait (ld_weight).
#include "kernels/draft/draft.h"
#include "kernels/draft/draft_device.cuh"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <stdexcept>

namespace ingine::kernels::draft {
namespace {

// Row kernels: one CTA per row of H values, a thread owns 8 consecutive elements (one 16-byte BF16 load).
constexpr int ROW_THREADS = H / 8;  // 288
static_assert(H % 8 == 0 && ROW_THREADS % 32 == 0);

// 8 floats -> 8 BF16 (one uint4), round to nearest.
__device__ __forceinline__ uint4 pack8(const float* f) {
  uint32_t w[4];
#pragma unroll
  for (int i = 0; i < 4; ++i) {
    const __nv_bfloat162 b = __floats2bfloat162_rn(f[2 * i], f[2 * i + 1]);
    w[i] = *reinterpret_cast<const uint32_t*>(&b);
  }
  return make_uint4(w[0], w[1], w[2], w[3]);
}

// rsqrt(mean(x^2) + eps) over the CTA's row (dspark.rms), from this thread's 8 values.
__device__ __forceinline__ float rms_scale(const float (&x)[8], float* scratch) {
  float ss = 0.f;
#pragma unroll
  for (int j = 0; j < 8; ++j) ss += x[j] * x[j];
  const float total = block_sum(ss, scratch);
  return rsqrtf(total / float(H) + EPS);
}

// y = bf16(rms(x) * w) for this thread's 8 values (separately rounded FP32 products).
__device__ __forceinline__ uint4 rms_norm(const float (&x)[8], const uint4& wv, float* scratch) {
  const float rs = rms_scale(x, scratch);
  float w[8], y[8];
  unpack8(wv, w);
#pragma unroll
  for (int j = 0; j < 8; ++j) y[j] = __fmul_rn(__fmul_rn(x[j], rs), w[j]);
  return pack8(y);
}

// mix_rmsnorm of this thread's 8 columns of one feats row [4, H]: m = sum_j f[j] * a[j], bf16(rms(m) * w).
__device__ __forceinline__ uint4 mix_row(const __nv_bfloat16* f, int c, const float (&a)[4], const uint4& wv,
                                         float* scratch) {
  uint4 fv[4];
#pragma unroll
  for (int j = 0; j < 4; ++j) fv[j] = *reinterpret_cast<const uint4*>(f + j * H + c);
  float m[8];
#pragma unroll
  for (int j = 0; j < 4; ++j) {
    float x[8];
    unpack8(fv[j], x);
#pragma unroll
    for (int i = 0; i < 8; ++i) m[i] = j == 0 ? __fmul_rn(x[i], a[0]) : __fadd_rn(m[i], __fmul_rn(x[i], a[j]));
  }
  return rms_norm(m, wv, scratch);
}

// alpha row `part` of bf16 [4, 4] as floats, loaded before a PDL wait (16-byte load of two rows).
__device__ __forceinline__ void load_alpha(const __nv_bfloat16* alpha, int part, float (&a)[4]) {
  const uint4 v = ld_weight(alpha + 8 * (part >> 1));
  float f[8];
  unpack8(v, f);
#pragma unroll
  for (int j = 0; j < 4; ++j) a[j] = f[4 * (part & 1) + j];
}

// ---- parameters (ordinary launches) ------------------------------------------------------------------

__global__ void set_draft_inputs_kernel(DraftParams* p, int anchor, int P, float temperature) {
  p->anchor = anchor;
  p->P = P;
  p->round_start = P;
  p->temperature = temperature;
}
__global__ void set_commit_start_kernel(DraftParams* p, int start) { p->start = start; }
__global__ void set_seed_kernel(DraftParams* p, std::uint64_t seed) {
  p->seed = seed;
  p->draws = 0;
}
__global__ void set_commit_kernel(DraftParams* p, const __nv_bfloat16* feats, int n, int start) {
  p->commit_feats = feats;
  p->ncommit = n;
  p->start = start;
}

// ---- context -------------------------------------------------------------------------------------------

__global__ void __launch_bounds__(ROW_THREADS) mix_rmsnorm_kernel(const __nv_bfloat16* feats,
                                                                  const __nv_bfloat16* __restrict__ alpha,
                                                                  const __nv_bfloat16* __restrict__ w,
                                                                  __nv_bfloat16* out) {
  __shared__ float scratch[33];
  pdl_trigger();
  const int r = blockIdx.x, c = 8 * threadIdx.x;
  const uint4 wv = ld_weight(w + c);
  const float a[4] = {bf(alpha[0]), bf(alpha[1]), bf(alpha[2]), bf(alpha[3])};
  pdl_wait();
  *reinterpret_cast<uint4*>(out + size_t(r) * H + c) = mix_row(feats + size_t(r) * 4 * H, c, a, wv, scratch);
}

// One head row of 128 values, one per thread (blockDim 128): bf16(rms(x) * w), then neox RoPE at `pos`
// over the full head dim, rounded to BF16. `x` is this thread's value; returns the thread's output.
__device__ __forceinline__ float norm_rope_head(float x, const __nv_bfloat16* __restrict__ w, int pos,
                                                float* scratch, float* pair) {
  const int d = threadIdx.x;
  const float ss = block_sum(x * x, scratch);
  const float rs = rsqrtf(ss / float(HEAD_DIM) + EPS);
  const float y = bf_round(__fmul_rn(__fmul_rn(x, rs), bf(w[d])));
  pair[d] = y;
  __syncthreads();
  const int half = HEAD_DIM / 2, i = d & (half - 1);
  // inv = theta ** (-(2 i) / 128) and the angle in FP32, as dspark.rope
  const float inv = powf(ROPE_THETA, (-float(i) * 2.f) / float(HEAD_DIM));
  const float ang = __fmul_rn(float(pos), inv);
  const float c = cosf(ang), s = sinf(ang);
  const float x1 = pair[i], x2 = pair[i + half];
  const float o = d < half ? __fsub_rn(__fmul_rn(x1, c), __fmul_rn(x2, s))
                           : __fadd_rn(__fmul_rn(x2, c), __fmul_rn(x1, s));
  __syncthreads();  // pair is reused by the caller's next head
  return bf_round(o);
}

__global__ void __launch_bounds__(HEAD_DIM) ctx_kv_store_kernel(const __nv_bfloat16* kv, const DraftParams* p,
                                                               int row0, int keep_from,
                                                               const __nv_bfloat16* __restrict__ k_norm,
                                                               void* k_cache, void* v_cache, bool fp8, int size,
                                                               std::int32_t* ctx_pos) {
  __shared__ float scratch[33];
  __shared__ float pair[HEAD_DIM];
  pdl_trigger();
  pdl_wait();
  const int r = keep_from + blockIdx.x, g = blockIdx.y, d = threadIdx.x;
  const int pos = p->start + row0 + r;
  const int slot = pos % size;
  const __nv_bfloat16* row = kv + size_t(r) * 2 * KV_DIM;
  const float k = norm_rope_head(bf(row[g * HEAD_DIM + d]), k_norm, pos, scratch, pair);
  const size_t at = (size_t(slot) * KV_HEADS + g) * HEAD_DIM + d;
  store_cache(k_cache, at, k, fp8);
  store_cache(v_cache, at, bf(row[KV_DIM + g * HEAD_DIM + d]), fp8);
  if (ctx_pos != nullptr && g == 0 && d == 0) ctx_pos[slot] = pos;
}

// The row work at a round's start, grid (BLOCK rows, parts) with part = part0 + blockIdx.y:
//   parts 0..2  the layers' context inputs xin[part, r] of the small commit (zero rows from n on)
//   part 3      f_rnn (row n - 1)
//   part 4      the draft block's input rows (embed_block)
// ROUND: n, the taps and the anchor come from the round inputs, and one thread publishes them in *p (no CTA
// reads the fields it writes); otherwise they are read from *p.
struct RowWeights {
  const __nv_bfloat16* alpha;       // bf16 [4, 4]
  const __nv_bfloat16* f_norm;      // bf16 [H]
  const void* embed;                // bf16 or fp8 [V, H]
  const float* embed_scale;         // fp32 [V] or null
  const float* embed_norm;          // fp32 [H]
  const __nv_bfloat16* mask_embed;  // bf16 [H]
  const __nv_bfloat16* in_norm;     // bf16 [H]
};
struct RowOutputs {
  __nv_bfloat16 *xin, *f_rnn, *res, *xn;
  unsigned long long* picks;
};

template <bool ROUND>
__global__ void __launch_bounds__(ROW_THREADS) round_rows_kernel(DraftParams* p, RoundInputs in, RowWeights W,
                                                                 CtxLayers L, RowOutputs O, int part0) {
  __shared__ float scratch[33];
  pdl_trigger();
  const int r = blockIdx.x, part = part0 + blockIdx.y, c = 8 * threadIdx.x;
  // weights first: in flight across the wait
  const __nv_bfloat16* norm = part < 3 ? L.ctx_norm[part] : (part == 3 ? W.f_norm : W.in_norm);
  const uint4 wv = ld_weight(norm + c);
  uint4 aux[2] = {make_uint4(0, 0, 0, 0), make_uint4(0, 0, 0, 0)};  // embed_norm (row 0) or mask_embed
  float a[4] = {0.f, 0.f, 0.f, 0.f};
  if (part == 4) {
    if (r == 0) {
      aux[0] = ld_weight(W.embed_norm + c);
      aux[1] = ld_weight(W.embed_norm + c + 4);
    } else {
      aux[0] = ld_weight(W.mask_embed + c);
    }
  } else {
    load_alpha(W.alpha, part, a);
  }
  pdl_wait();
  int n, anchor;
  const __nv_bfloat16* feats;
  if constexpr (ROUND) {
    const int m = *in.m, start = p->round_start;
    n = *in.accepted + 1;
    feats = in.taps[0];
    for (int i = 0; i < in.nwidths; ++i)
      if (in.widths[i] == m) feats = in.taps[i];
    anchor = p->temperature > 0.f ? *in.next_sampled : *in.next_greedy;
    if (r == 0 && blockIdx.y == 0 && threadIdx.x == 0) {
      p->commit_feats = feats;
      p->ncommit = n;
      p->start = start;
      p->anchor = anchor;
      p->P = start + n;
    }
  } else {
    n = p->ncommit;
    feats = p->commit_feats;
    anchor = p->anchor;
  }

  if (part == 4) {
    // res[0] = bf16(rms(embed[anchor]) * embed_norm), res[1..7] = mask_embed; xn = bf16(rms(res) * in_norm)
    float x[8];
    if (r == 0) {
      float v[8];
      if (W.embed_scale != nullptr) {
        const uint2 raw8 = *reinterpret_cast<const uint2*>(static_cast<const uint8_t*>(W.embed) + size_t(anchor) * H + c);
        const uint32_t u0 = e4m3x2_to_bf16x2(raw8.x), u1 = e4m3x2_to_bf16x2(raw8.x >> 16);
        const uint32_t u2 = e4m3x2_to_bf16x2(raw8.y), u3 = e4m3x2_to_bf16x2(raw8.y >> 16);
        unpack8(make_uint4(u0, u1, u2, u3), v);
        const float sc = W.embed_scale[anchor];
#pragma unroll
        for (int j = 0; j < 8; ++j) v[j] = bf_round(v[j] * sc);
      } else {
        unpack8(*reinterpret_cast<const uint4*>(static_cast<const __nv_bfloat16*>(W.embed) + size_t(anchor) * H + c), v);
      }
      const float rs = rms_scale(v, scratch);
      const float en[8] = {__uint_as_float(aux[0].x), __uint_as_float(aux[0].y), __uint_as_float(aux[0].z),
                           __uint_as_float(aux[0].w), __uint_as_float(aux[1].x), __uint_as_float(aux[1].y),
                           __uint_as_float(aux[1].z), __uint_as_float(aux[1].w)};
#pragma unroll
      for (int j = 0; j < 8; ++j) x[j] = bf_round(__fmul_rn(__fmul_rn(v[j], rs), en[j]));
    } else {
      unpack8(aux[0], x);
    }
    *reinterpret_cast<uint4*>(O.res + size_t(r) * H + c) = pack8(x);
    *reinterpret_cast<uint4*>(O.xn + size_t(r) * H + c) = rms_norm(x, wv, scratch);
    if (r == 0 && threadIdx.x < GAMMA) O.picks[threadIdx.x] = 0ull;
    return;
  }
  const __nv_bfloat16* f = feats + size_t(r) * 4 * H;
  if (part == 3) {
    if (r == n - 1) *reinterpret_cast<uint4*>(O.f_rnn + c) = mix_row(f, c, a, wv, scratch);
    return;
  }
  uint4* out = reinterpret_cast<uint4*>(O.xin + (size_t(part) * BLOCK + r) * H + c);
  if (r >= n) {
    *out = make_uint4(0, 0, 0, 0);
    return;
  }
  *out = mix_row(f, c, a, wv, scratch);
}

// grid (BLOCK rows, KV_HEADS, 3 layers), block 128
__global__ void __launch_bounds__(HEAD_DIM) commit_store_kernel(const float* kv, const DraftParams* p, CtxLayers L) {
  __shared__ float scratch[33];
  __shared__ float pair[HEAD_DIM];
  pdl_trigger();
  pdl_wait();
  const int r = blockIdx.x, g = blockIdx.y, l = blockIdx.z, d = threadIdx.x;
  if (r >= p->ncommit) return;
  const int pos = p->start + r;
  const int slot = pos % L.size[l];
  const float* row = kv + size_t(r) * 3 * 2 * KV_DIM + size_t(l) * 2 * KV_DIM;
  const float k = norm_rope_head(row[g * HEAD_DIM + d], L.k_norm_ctx[l], pos, scratch, pair);
  const size_t at = (size_t(slot) * KV_HEADS + g) * HEAD_DIM + d;
  store_cache(L.k_cache[l], at, k, L.fp8[l]);
  store_cache(L.v_cache[l], at, row[KV_DIM + g * HEAD_DIM + d], L.fp8[l]);
  if (L.ctx_pos[l] != nullptr && g == 0 && d == 0) L.ctx_pos[l][slot] = pos;
}

// ---- draft block ---------------------------------------------------------------------------------------

// grid (8 rows, 24 q + 8 k + 8 v heads), block 128
__global__ void __launch_bounds__(HEAD_DIM) qkv_rope_kernel(const float* qkv, const __nv_bfloat16* __restrict__ q_norm,
                                                           const __nv_bfloat16* __restrict__ k_norm,
                                                           const DraftParams* p, __nv_bfloat16* q_out,
                                                           void* k_cache, void* v_cache, bool fp8, bool local,
                                                           int ctx) {
  __shared__ float scratch[33];
  __shared__ float pair[HEAD_DIM];
  pdl_trigger();
  pdl_wait();
  const int r = blockIdx.x, h = blockIdx.y, d = threadIdx.x;
  const int pos = p->P + r;
  const float* row = qkv + size_t(r) * QKV_ROWS;
  const int slot = local ? WINDOW + r : pos % ctx;
  if (h < HEADS) {
    const float q = norm_rope_head(row[h * HEAD_DIM + d], q_norm, pos, scratch, pair);
    q_out[(size_t(r) * HEADS + h) * HEAD_DIM + d] = __float2bfloat16_rn(q);
  } else if (h < HEADS + KV_HEADS) {
    const int g = h - HEADS;
    const float k = norm_rope_head(row[Q_DIM + g * HEAD_DIM + d], k_norm, pos, scratch, pair);
    store_cache(k_cache, (size_t(slot) * KV_HEADS + g) * HEAD_DIM + d, k, fp8);
  } else {
    const int g = h - HEADS - KV_HEADS;
    store_cache(v_cache, (size_t(slot) * KV_HEADS + g) * HEAD_DIM + d, row[Q_DIM + KV_DIM + g * HEAD_DIM + d], fp8);
  }
}

// ---- residual stream -----------------------------------------------------------------------------------

__global__ void __launch_bounds__(ROW_THREADS) add_rmsnorm_kernel(__nv_bfloat16* res, const float* delta,
                                                                  const __nv_bfloat16* __restrict__ w,
                                                                  __nv_bfloat16* xn) {
  __shared__ float scratch[33];
  pdl_trigger();
  const size_t at = size_t(blockIdx.x) * H + 8 * threadIdx.x;
  const uint4 wv = ld_weight(w + 8 * threadIdx.x);
  pdl_wait();
  float x[8];
  unpack8(*reinterpret_cast<const uint4*>(res + at), x);
  const float4 d0 = *reinterpret_cast<const float4*>(delta + at), d1 = *reinterpret_cast<const float4*>(delta + at + 4);
  const float d[8] = {d0.x, d0.y, d0.z, d0.w, d1.x, d1.y, d1.z, d1.w};
#pragma unroll
  for (int j = 0; j < 8; ++j) x[j] = bf_round(__fadd_rn(x[j], d[j]));
  *reinterpret_cast<uint4*>(res + at) = pack8(x);
  *reinterpret_cast<uint4*>(xn + at) = rms_norm(x, wv, scratch);
}

__global__ void silu_mul_kernel(const float* gu, int rows, int inter, __nv_bfloat16* out) {
  pdl_trigger();
  pdl_wait();
  const size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i >= size_t(rows) * inter) return;
  const size_t r = i / inter, j = i % inter;
  const float g = gu[r * 2 * inter + j], u = gu[r * 2 * inter + inter + j];
  const float s = __fdiv_rn(g, __fadd_rn(1.f, expf(-g)));  // torch silu: x / (1 + exp(-x))
  out[i] = __float2bfloat16_rn(__fmul_rn(s, u));
}

__global__ void fill_kernel(float* x, size_t n, float value) {
  for (size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x; i < n; i += size_t(gridDim.x) * blockDim.x)
    x[i] = value;
}

void check_launch() { CUDA_CHECK(cudaGetLastError()); }

// rows are read and written 16 bytes at a time
void require_aligned(const void* ptr, const char* what) {
  if (reinterpret_cast<std::uintptr_t>(ptr) % 16 != 0) throw std::invalid_argument(what);
}

}  // namespace

void set_draft_inputs(DraftParams* p, int anchor, int P, float temperature, cudaStream_t stream) {
  set_draft_inputs_kernel<<<1, 1, 0, stream>>>(p, anchor, P, temperature);
  check_launch();
}
void set_commit_start(DraftParams* p, int start, cudaStream_t stream) {
  set_commit_start_kernel<<<1, 1, 0, stream>>>(p, start);
  check_launch();
}
void set_seed(DraftParams* p, std::uint64_t seed, cudaStream_t stream) {
  set_seed_kernel<<<1, 1, 0, stream>>>(p, seed);
  check_launch();
}
void set_commit(DraftParams* p, const __nv_bfloat16* feats, int n, int start, cudaStream_t stream) {
  if (n < 1 || n > BLOCK) throw std::invalid_argument("set_commit: n must be in 1..8");
  require_aligned(feats, "set_commit: feats must be 16-byte aligned");
  set_commit_kernel<<<1, 1, 0, stream>>>(p, feats, n, start);
  check_launch();
}

void mix_rmsnorm(const __nv_bfloat16* feats, int n, const __nv_bfloat16* alpha, const __nv_bfloat16* w,
                 __nv_bfloat16* out, cudaStream_t stream) {
  if (n <= 0) return;
  require_aligned(feats, "mix_rmsnorm: feats must be 16-byte aligned");
  require_aligned(out, "mix_rmsnorm: out must be 16-byte aligned");
  launch_pdl(mix_rmsnorm_kernel, dim3(n), dim3(ROW_THREADS), stream, feats, alpha, w, out);
}

void ctx_kv_store(const __nv_bfloat16* kv, int n, const DraftParams* p, int row0, int keep_from,
                  const __nv_bfloat16* k_norm, void* k_cache, void* v_cache, bool fp8, int size,
                  std::int32_t* ctx_pos, cudaStream_t stream) {
  if (keep_from < 0 || size <= 0) throw std::invalid_argument("ctx_kv_store: bad keep_from/size");
  if (n - keep_from <= 0) return;
  launch_pdl(ctx_kv_store_kernel, dim3(n - keep_from, KV_HEADS), dim3(HEAD_DIM), stream, kv, p, row0, keep_from,
             k_norm, k_cache, v_cache, fp8, size, ctx_pos);
}

void commit_mix(const DraftParams* p, const __nv_bfloat16* alpha, const CtxLayers& L, const __nv_bfloat16* f_norm,
                __nv_bfloat16* xin, __nv_bfloat16* f_rnn, cudaStream_t stream) {
  const RowWeights W{alpha, f_norm, nullptr, nullptr, nullptr, nullptr, nullptr};
  const RowOutputs O{xin, f_rnn, nullptr, nullptr, nullptr};
  launch_pdl(round_rows_kernel<false>, dim3(BLOCK, 4), dim3(ROW_THREADS), stream, const_cast<DraftParams*>(p),
             RoundInputs{}, W, L, O, 0);
}

void commit_store(const float* kv, const DraftParams* p, const CtxLayers& L, cudaStream_t stream) {
  for (int l = 0; l < 3; ++l)
    if (L.size[l] < BLOCK) throw std::invalid_argument("commit_store: rings must hold a block");
  launch_pdl(commit_store_kernel, dim3(BLOCK, KV_HEADS, 3), dim3(HEAD_DIM), stream, kv, p, L);
}

void embed_block(const void* embed, const float* embed_scale, const float* embed_norm, const __nv_bfloat16* mask_embed,
                 const __nv_bfloat16* in_norm, const DraftParams* p, __nv_bfloat16* res, __nv_bfloat16* xn,
                 unsigned long long* picks, cudaStream_t stream) {
  require_aligned(embed, "embed_block: embed must be 16-byte aligned");
  const RowWeights W{nullptr, nullptr, embed, embed_scale, embed_norm, mask_embed, in_norm};
  const RowOutputs O{nullptr, nullptr, res, xn, picks};
  launch_pdl(round_rows_kernel<false>, dim3(BLOCK, 1), dim3(ROW_THREADS), stream, const_cast<DraftParams*>(p),
             RoundInputs{}, W, CtxLayers{}, O, 4);
}

void round_begin(DraftParams* p, const RoundInputs& in, const __nv_bfloat16* alpha, const CtxLayers& L,
                 const __nv_bfloat16* f_norm, const void* embed, const float* embed_scale, const float* embed_norm,
                 const __nv_bfloat16* mask_embed, const __nv_bfloat16* in_norm, __nv_bfloat16* xin,
                 __nv_bfloat16* f_rnn, __nv_bfloat16* res, __nv_bfloat16* xn, unsigned long long* picks,
                 cudaStream_t stream) {
  if (in.nwidths < 1 || in.nwidths > 3 || !in.m || !in.accepted || !in.next_greedy || !in.next_sampled)
    throw std::invalid_argument("round_begin: incomplete round inputs");
  for (int i = 0; i < in.nwidths; ++i) require_aligned(in.taps[i], "round_begin: taps must be 16-byte aligned");
  require_aligned(embed, "round_begin: embed must be 16-byte aligned");
  const RowWeights W{alpha, f_norm, embed, embed_scale, embed_norm, mask_embed, in_norm};
  const RowOutputs O{xin, f_rnn, res, xn, picks};
  launch_pdl(round_rows_kernel<true>, dim3(BLOCK, 5), dim3(ROW_THREADS), stream, p, in, W, L, O, 0);
}

void qkv_rope(const float* qkv, const __nv_bfloat16* q_norm, const __nv_bfloat16* k_norm, const DraftParams* p,
              __nv_bfloat16* q_out, void* k_cache, void* v_cache, bool fp8, bool local, int ctx,
              cudaStream_t stream) {
  launch_pdl(qkv_rope_kernel, dim3(BLOCK, HEADS + 2 * KV_HEADS), dim3(HEAD_DIM), stream, qkv, q_norm, k_norm, p, q_out,
             k_cache, v_cache, fp8, local, ctx);
}

void add_rmsnorm(__nv_bfloat16* res, const float* delta, const __nv_bfloat16* w, __nv_bfloat16* xn, int rows,
                 cudaStream_t stream) {
  if (rows <= 0) return;
  launch_pdl(add_rmsnorm_kernel, dim3(rows), dim3(ROW_THREADS), stream, res, delta, w, xn);
}

void silu_mul(const float* gu, int rows, int inter, __nv_bfloat16* out, cudaStream_t stream) {
  const size_t n = size_t(rows) * inter;
  if (n == 0) return;
  launch_pdl(silu_mul_kernel, dim3(unsigned((n + 255) / 256)), dim3(256), stream, gu, rows, inter, out);
}

void fill(float* x, std::size_t n, float value, cudaStream_t stream) {
  if (n == 0) return;
  const unsigned blocks = unsigned(std::min<std::size_t>((n + 255) / 256, 4096));
  fill_kernel<<<blocks, 256, 0, stream>>>(x, n, value);
  check_launch();
}

}  // namespace ingine::kernels::draft
