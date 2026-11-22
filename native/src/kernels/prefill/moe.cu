// Prefill MoE routing (model._moe / _moe_grouped for T tokens): router selection, expert-sorted pair lists and
// tiles, and the deterministic combine. The grouped expert GEMMs are in wgemm.cu.
#include "kernels/prefill/common.cuh"
#include "kernels/prefill/prefill.h"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <stdexcept>

namespace ingine::kernels::prefill {
namespace {

// Warp per token: sigmoid scores, 8 rounds of arg-max over s + bias (lowest expert on ties).
__global__ void topk_kernel(const float* __restrict__ logits, const float* __restrict__ bias, int T,
                            int* __restrict__ idx, float* __restrict__ wts, int* __restrict__ counts) {
  const int t = blockIdx.x * (blockDim.x / 32) + (threadIdx.x >> 5), lane = threadIdx.x & 31;
  if (t >= T) return;
  float sc[4], key[4];
#pragma unroll
  for (int i = 0; i < 4; ++i) {
    const int e = lane + 32 * i;
    sc[i] = 1.f / (1.f + expf(-logits[size_t(t) * kExperts + e]));
    key[i] = sc[i] + bias[e];
  }
  int sel_e[kTopK];
  float sel_s[kTopK];
#pragma unroll
  for (int k = 0; k < kTopK; ++k) {
    float best = -INFINITY;
    int be = kExperts;
#pragma unroll
    for (int i = 0; i < 4; ++i)
      if (key[i] > best) best = key[i], be = lane + 32 * i;
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) {
      const float ob = __shfl_xor_sync(0xffffffffu, best, o);
      const int oe = __shfl_xor_sync(0xffffffffu, be, o);
      if (ob > best || (ob == best && oe < be)) best = ob, be = oe;
    }
    sel_e[k] = be;
    float s_be = 0.f;
#pragma unroll
    for (int i = 0; i < 4; ++i)
      if (lane + 32 * i == be) s_be = sc[i], key[i] = -INFINITY;
    sel_s[k] = __shfl_sync(0xffffffffu, s_be, be & 31);
  }
  if (lane == 0) {
    float sum = sel_s[0];
#pragma unroll
    for (int k = 1; k < kTopK; ++k) sum += sel_s[k];
    const float den = sum + 1e-20f;
#pragma unroll
    for (int k = 0; k < kTopK; ++k) {
      idx[size_t(t) * kTopK + k] = sel_e[k];
      wts[size_t(t) * kTopK + k] = sel_s[k] / den * 8.f;
      atomicAdd(counts + sel_e[k], 1);
    }
  }
}

// One CTA: expert offsets, tiles, scatter of the pairs into expert order (order within an expert does not
// affect any result: every pair row is computed independently).
__global__ void __launch_bounds__(1024) route_kernel(const int* __restrict__ idx, int T, int TR, MoeWork w) {
  __shared__ int offs[kExperts], tile0[kExperts + 1], cursor[kExperts];
  const int tid = threadIdx.x;
  if (tid == 0) {
    int o = 0, tl = 0;
    for (int e = 0; e < kExperts; ++e) {
      const int c = w.counts[e];
      offs[e] = o;
      tile0[e] = tl;
      o += c;
      tl += (c + TR - 1) / TR;
    }
    tile0[kExperts] = tl;
    *w.tiles_n = tl;
  }
  if (tid < kExperts) cursor[tid] = 0;
  __syncthreads();
  if (tid < kExperts) {
    const int c = w.counts[tid];
    for (int i = 0; i * TR < c; ++i) w.tiles[tile0[tid] + i] = make_int4(tid, offs[tid] + i * TR, min(TR, c - i * TR), 0);
  }
  for (int p = tid; p < T * kTopK; p += blockDim.x) {
    const int e = idx[p];
    const int pos = offs[e] + atomicAdd(&cursor[e], 1);
    w.pair_token[pos] = p / kTopK;
    w.pair_dst[pos] = p;
  }
}

// out[t] = bf16(sum_slot down[t, slot] * wts[t, slot]), 4 channels per thread.
__global__ void __launch_bounds__(kHidden / 4) combine_kernel(const uint2* __restrict__ down, const float* __restrict__ wts,
                                                            uint2* __restrict__ out) {
  const int t = blockIdx.x, c = threadIdx.x;
  float acc[4];
#pragma unroll
  for (int k = 0; k < kTopK; ++k) {
    const uint2 d = down[(size_t(t) * kTopK + k) * (kHidden / 4) + c];
    const float wk = wts[size_t(t) * kTopK + k];
    const float p[4] = {__fmul_rn(bf_lo(d.x), wk), __fmul_rn(bf_hi(d.x), wk), __fmul_rn(bf_lo(d.y), wk),
                        __fmul_rn(bf_hi(d.y), wk)};
#pragma unroll
    for (int i = 0; i < 4; ++i) acc[i] = k == 0 ? p[i] : __fadd_rn(acc[i], p[i]);
  }
  out[size_t(t) * (kHidden / 4) + c] = make_uint2(pack_bf16x2(acc[0], acc[1]), pack_bf16x2(acc[2], acc[3]));
}

}  // namespace

int moe_tile_rows(int T) {
  static const int forced = [] {
    const char* e = std::getenv("INGINE_MOE_TILE_ROWS");
    const int v = e ? std::atoi(e) : 0;
    return v == 32 || v == 128 ? v : 0;
  }();
  return forced ? forced : T <= 512 ? 32 : 128;
}

int moe_tiles_capacity(int max_T) { return std::max(moe_max_tiles(std::min(max_T, 512)), moe_max_tiles(max_T)); }

void moe_topk(const float* logits, const float* bias, int T, int* idx, float* wts, int* counts, cudaStream_t stream) {
  topk_kernel<<<(T + 7) / 8, 256, 0, stream>>>(logits, bias, T, idx, wts, counts);
}

void moe_route(const int* idx, int T, const MoeWork& w, cudaStream_t stream) {
  route_kernel<<<1, 1024, 0, stream>>>(idx, T, moe_tile_rows(T), w);
}

void moe_combine(const __nv_bfloat16* down, const float* wts, int T, __nv_bfloat16* out, cudaStream_t stream) {
  combine_kernel<<<T, kHidden / 4, 0, stream>>>(reinterpret_cast<const uint2*>(down), wts,
                                                reinterpret_cast<uint2*>(out));
}

}  // namespace ingine::kernels::prefill
