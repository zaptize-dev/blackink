// Single-barrier block sum for the latency-bound dense kernels (mix, message_gates).
#pragma once

#include "kernels/dense/dense.cuh"

namespace ingine::kernels::dense {

// Block sum of N <= 32 values per thread over NW warps, every thread gets the totals in `v`: warp stage
// (butterflies; for N >= 6 the transposing reduce_scatter), one shared round, then lane i sums the NW warp
// partials of value i (warp order, so every thread and every CTA running the same code gets identical
// totals) and the totals are broadcast by shuffles. One __syncthreads instead of block_sum's two and no
// serial per-value tail. `scratch` ([NW * 32] floats) must not be reused by another call without a barrier.
template <int N, int NW>
__device__ __forceinline__ void block_sum_bcast(float (&v)[N], float* scratch) {
  static_assert(N <= 32, "at most 32 values");
  const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
  if constexpr (N < 6) {
#pragma unroll
    for (int i = 0; i < N; ++i) v[i] = group_sum<32>(v[i]);
    if (lane < N) {
      float mine = v[0];
#pragma unroll
      for (int i = 1; i < N; ++i) mine = lane == i ? v[i] : mine;
      scratch[warp * 32 + lane] = mine;
    }
  } else {
    const float part = reduce_scatter<32, N>(v);  // lane l: value l % pow2_ceil(N)
    if (lane < pow2_ceil(N)) scratch[warp * 32 + lane] = part;
  }
  __syncthreads();
  float s = 0.f;
  if (lane < N) {
#pragma unroll
    for (int w = 0; w < NW; ++w) s += scratch[w * 32 + lane];
  }
#pragma unroll
  for (int i = 0; i < N; ++i) v[i] = __shfl_sync(0xffffffffu, s, i);
}

}  // namespace ingine::kernels::dense
