// Device helpers shared by the DSpark draft and speculative-acceptance kernels.
#pragma once

#include "core/device.h"
#include "kernels/common/pdl.cuh"
#include "kernels/draft/rng.h"

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>

#include <cstdint>
#include <utility>

namespace ingine::kernels::draft {

__device__ __forceinline__ float bf_round(float x) { return __bfloat162float(__float2bfloat16_rn(x)); }
__device__ __forceinline__ float bf(const __nv_bfloat16& x) { return __bfloat162float(x); }

// One element of a key/value cache: BF16, or E4M3 without a scale (nearest even, saturating at +-448).
__device__ __forceinline__ void store_cache(void* cache, size_t at, float x, bool fp8) {
  if (fp8) static_cast<__nv_fp8_storage_t*>(cache)[at] = __nv_cvt_float_to_fp8(x, __NV_SATFINITE, __NV_E4M3);
  else static_cast<__nv_bfloat16*>(cache)[at] = __float2bfloat16_rn(x);
}
// 2 E4M3 (low 16 bits of u) -> BF16x2, exact: one cvt.rn.bf16x2.e4m3x2 on sm_120a (cuda_fp8.hpp)
__device__ __forceinline__ uint32_t e4m3x2_to_bf16x2(uint32_t u) {
  const __nv_bfloat162_raw b = __nv_cvt_fp8x2_to_bf162raw(static_cast<__nv_fp8x2_storage_t>(u & 0xffffu), __NV_E4M3);
  return uint32_t(b.x) | (uint32_t(b.y) << 16);
}

// 8 BF16 values (one uint4) -> 8 floats.
__device__ __forceinline__ void unpack8(const uint4& v, float* f) {
  const uint32_t w[4] = {v.x, v.y, v.z, v.w};
#pragma unroll
  for (int i = 0; i < 4; ++i) {
    f[2 * i] = __uint_as_float(w[i] << 16);
    f[2 * i + 1] = __uint_as_float(w[i] & 0xffff0000u);
  }
}

__device__ __forceinline__ float warp_sum(float v) {
#pragma unroll
  for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
  return v;
}
__device__ __forceinline__ float warp_max(float v) {
#pragma unroll
  for (int o = 16; o > 0; o >>= 1) v = fmaxf(v, __shfl_xor_sync(0xffffffffu, v, o));
  return v;
}

// Block-wide sum (blockDim.x a multiple of 32, <= 1024); every thread gets the total. `scratch` holds 33
// floats; contains two barriers, so consecutive calls may reuse it.
__device__ __forceinline__ float block_sum(float v, float* scratch) {
  const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5, nw = blockDim.x >> 5;
  v = warp_sum(v);
  __syncthreads();  // a previous call's readers are done with scratch
  if (lane == 0) scratch[warp] = v;
  __syncthreads();
  float t = 0.f;
  for (int w = 0; w < nw; ++w) t += scratch[w];
  return t;
}
__device__ __forceinline__ float block_max(float v, float* scratch) {
  const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5, nw = blockDim.x >> 5;
  v = warp_max(v);
  __syncthreads();
  if (lane == 0) scratch[warp] = v;
  __syncthreads();
  float t = -INFINITY;
  for (int w = 0; w < nw; ++w) t = fmaxf(t, scratch[w]);
  return t;
}

// Order-preserving map of a float to uint32 (larger float -> larger key; -0 sorts below +0).
__device__ __forceinline__ uint32_t float_key(float f) {
  const uint32_t u = __float_as_uint(f);
  return (u & 0x80000000u) ? ~u : (u | 0x80000000u);
}

// (score, index) packed so that atomicMax / max picks the largest score and, among equal scores, the
// lowest index (torch.argmax returns the first maximum).
__device__ __forceinline__ unsigned long long pack_argmax(float score, uint32_t index) {
  return (static_cast<unsigned long long>(float_key(score)) << 32) | (0xffffffffu - index);
}
__device__ __forceinline__ uint32_t unpack_argmax_index(unsigned long long p) {
  return 0xffffffffu - static_cast<uint32_t>(p & 0xffffffffu);
}

// 16-byte weight load issued ahead of pdl_wait(): ld.global.cg in asm volatile, which ptxas keeps above the
// wait (it may sink .nc loads below it, see kernels/common/pdl.cuh), so the weights are in flight while the
// predecessor finishes.
__device__ __forceinline__ uint4 ld_weight(const void* p) {
  uint4 r;
  asm volatile("ld.global.cg.v4.u32 {%0,%1,%2,%3}, [%4];" : "=r"(r.x), "=r"(r.y), "=r"(r.z), "=r"(r.w) : "l"(p));
  return r;
}

// Launches `kernel` as a programmatic dependent of its stream predecessor (kernels/common/pdl.cuh). The
// kernel calls pdl_trigger() first and pdl_wait() before it touches data another kernel writes.
template <class... Params, class... Args>
void launch_pdl(void (*kernel)(Params...), dim3 grid, dim3 block, cudaStream_t stream, Args&&... args) {
  CUDA_CHECK(pdl_launch(kernel, grid, block, 0, stream, std::forward<Args>(args)...));
}

}  // namespace ingine::kernels::draft
