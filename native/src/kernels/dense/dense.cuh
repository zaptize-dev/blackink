// Shared device helpers of the `dense` kernels (vector loads, BF16/FP8 unpacking, reductions).
// Ported from engine/csrc/dense/dense.cuh.
#pragma once

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>
#include <stdint.h>

namespace ingine::kernels::dense {

constexpr int H = 2304;
constexpr float EPS = 1e-6f;

// 16-byte load of activations (L1 cached: every row of a block re-reads the same vector). A coherent,
// volatile load, not ld.global.nc: activations are written by the preceding kernel, and with programmatic
// dependent launch they must not be read (or hoisted by the compiler) before griddepcontrol.wait.
__device__ __forceinline__ uint4 ld_act(const void* p) {
  uint4 r;
  asm volatile("ld.global.v4.u32 {%0,%1,%2,%3}, [%4];"
               : "=r"(r.x), "=r"(r.y), "=r"(r.z), "=r"(r.w)
               : "l"(p)
               : "memory");
  return r;
}

// 16-byte asynchronous global -> shared copy (L2 only); complete with cp.async.commit_group / wait.
__device__ __forceinline__ void cp_async16(void* smem, const void* gmem) {
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16;" ::"r"(static_cast<unsigned>(__cvta_generic_to_shared(smem))),
               "l"(gmem)
               : "memory");
}

__device__ __forceinline__ float bf_lo(uint32_t u) { return __uint_as_float(u << 16); }
__device__ __forceinline__ float bf_hi(uint32_t u) { return __uint_as_float(u & 0xffff0000u); }

// 8 BF16 values (one uint4) -> 8 floats.
__device__ __forceinline__ void unpack_bf16x8(const uint4& v, float* f) {
  f[0] = bf_lo(v.x); f[1] = bf_hi(v.x);
  f[2] = bf_lo(v.y); f[3] = bf_hi(v.y);
  f[4] = bf_lo(v.z); f[5] = bf_hi(v.z);
  f[6] = bf_lo(v.w); f[7] = bf_hi(v.w);
}

// 4 E4M3 values (one uint32, element 0 in the low byte) -> 4 floats (exact via FP16).
__device__ __forceinline__ void unpack_e4m3x4(uint32_t u, float* f) {
  __half2_raw lo = __nv_cvt_fp8x2_to_halfraw2(static_cast<__nv_fp8x2_storage_t>(u & 0xffffu), __NV_E4M3);
  __half2_raw hi = __nv_cvt_fp8x2_to_halfraw2(static_cast<__nv_fp8x2_storage_t>(u >> 16), __NV_E4M3);
  float2 a = __half22float2(*reinterpret_cast<__half2*>(&lo));
  float2 b = __half22float2(*reinterpret_cast<__half2*>(&hi));
  f[0] = a.x; f[1] = a.y; f[2] = b.x; f[3] = b.y;
}

__device__ __forceinline__ void unpack_e4m3x16(const uint4& v, float* f) {
  unpack_e4m3x4(v.x, f); unpack_e4m3x4(v.y, f + 4); unpack_e4m3x4(v.z, f + 8); unpack_e4m3x4(v.w, f + 12);
}

__device__ __forceinline__ float round_bf16(float x) { return __bfloat162float(__float2bfloat16_rn(x)); }

// Sum over groups of LANES consecutive lanes (LANES = 16 or 32); every lane of the group gets the sum. Only
// the calling group's lanes take part, so the other half warp may have exited.
template <int LANES>
__device__ __forceinline__ unsigned group_mask() {
  return LANES == 32 ? 0xffffffffu : (0xffffu << (threadIdx.x & 16));
}
template <int LANES>
__device__ __forceinline__ float group_sum(float v) {
  const unsigned mask = group_mask<LANES>();
#pragma unroll
  for (int o = LANES / 2; o > 0; o >>= 1) v += __shfl_xor_sync(mask, v, o);
  return v;
}

constexpr int pow2_ceil(int n) { return n <= 1 ? 1 : 2 * pow2_ceil((n + 1) / 2); }

// Sums N values over a group of LANES lanes and scatters the totals: returns, in every lane l of the
// group, the total of value (l % P), P = pow2_ceil(N) (lanes with l % P >= N get 0). Plain butterflies
// down to P lanes, then a transposing butterfly (each lane ships the half of its partials its partner
// keeps): P * log2(LANES / P) + P - 1 shuffles instead of N * log2(LANES).
template <int LANES, int N>
__device__ __forceinline__ float reduce_scatter(const float (&in)[N]) {
  constexpr int P = pow2_ceil(N);
  static_assert(P <= LANES, "more values than lanes");
  const unsigned mask = group_mask<LANES>();
  const int lane = threadIdx.x & 31;
  float t[P];
#pragma unroll
  for (int i = 0; i < P; ++i) t[i] = i < N ? in[i] : 0.f;
#pragma unroll
  for (int o = LANES / 2; o >= P; o >>= 1)
#pragma unroll
    for (int i = 0; i < P; ++i) t[i] += __shfl_xor_sync(mask, t[i], o);
#pragma unroll
  for (int o = P / 2; o >= 1; o >>= 1) {
    const bool up = lane & o;
#pragma unroll
    for (int i = 0; i < o; ++i) {
      const float send = up ? t[i] : t[i + o];
      const float keep = up ? t[i + o] : t[i];
      t[i] = keep + __shfl_xor_sync(mask, send, o);
    }
  }
  return t[0];
}

// Slots per warp used by block_sum<N>: N values (plain butterflies), or 32 (reduce_scatter).
template <int N>
constexpr int block_sum_slots() { return N < 6 ? N : 32; }

// Block-wide sum of N <= 32 values per thread; `scratch` holds (blockDim/32 + 1) * block_sum_slots<N>()
// floats and must not be reused by the next block_sum without a barrier. Every thread gets the totals in
// `v`. Contains two __syncthreads. For N >= 6 the warp stage is reduce_scatter (31 or fewer shuffles
// instead of 5N): shuffle throughput is what bounds these small reductions.
template <int N>
__device__ __forceinline__ void block_sum(float (&v)[N], float* scratch) {
  static_assert(N <= 32, "at most 32 values");
  constexpr int S = block_sum_slots<N>();
  const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5, nw = blockDim.x >> 5;
  if constexpr (S == N) {
#pragma unroll
    for (int i = 0; i < N; ++i) v[i] = group_sum<32>(v[i]);
    if (lane == 0) {
#pragma unroll
      for (int i = 0; i < N; ++i) scratch[warp * N + i] = v[i];
    }
  } else {
    scratch[warp * 32 + lane] = reduce_scatter<32, N>(v);  // lane l: value l % P
  }
  __syncthreads();
  float* tot = scratch + nw * S;
  if (threadIdx.x < N) {
    float s = 0.f;
    for (int w = 0; w < nw; ++w) s += scratch[w * S + threadIdx.x];
    tot[threadIdx.x] = s;
  }
  __syncthreads();
#pragma unroll
  for (int i = 0; i < N; ++i) v[i] = tot[i];
}

}  // namespace ingine::kernels::dense
