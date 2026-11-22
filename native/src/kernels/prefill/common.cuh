// Adapted from fngine (github.com/satellitedown/fngine, Apache-2.0) for ingine.
// Device helpers of the prefill kernels: ldmatrix / mma.sync wrappers and cp.async (from fngine
// src/kernels/common/mma.cuh and memory.cuh), BF16/FP16/FP8 packing, swizzled shared-memory tiles.
#pragma once

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>

#include <cstdint>

namespace ingine::kernels::prefill {

__device__ __forceinline__ unsigned smem_addr(const void* ptr) {
  return static_cast<unsigned>(__cvta_generic_to_shared(ptr));
}

__device__ __forceinline__ void ldmatrix_x4(unsigned& r0, unsigned& r1, unsigned& r2, unsigned& r3,
                                            unsigned addr) {
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
               : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3)
               : "r"(addr));
}

__device__ __forceinline__ void ldmatrix_x4_t(unsigned& r0, unsigned& r1, unsigned& r2, unsigned& r3,
                                              unsigned addr) {
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0,%1,%2,%3}, [%4];\n"
               : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3)
               : "r"(addr));
}

// D += A * B, m16n8k16, BF16 inputs, FP32 accumulation.
__device__ __forceinline__ void mma_bf16(float (&c)[4], const unsigned (&a)[4], unsigned b0, unsigned b1) {
  asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
               "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
               : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
               : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}

// D += A * B, m16n8k16, FP16 inputs, FP32 accumulation.
__device__ __forceinline__ void mma_f16(float (&c)[4], const unsigned (&a)[4], unsigned b0, unsigned b1) {
  asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
               "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
               : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
               : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}

// 16-byte cp.async (L2 only); src_bytes < 16 zero-fills the rest (0: nothing is read).
__device__ __forceinline__ void cp_async16(void* smem_dst, const void* gmem_src, int src_bytes = 16) {
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n"
               :
               : "r"(smem_addr(smem_dst)), "l"(gmem_src), "r"(src_bytes));
}

__device__ __forceinline__ void cp_commit() { asm volatile("cp.async.commit_group;\n"); }

template <int Groups>
__device__ __forceinline__ void cp_wait() {
  asm volatile("cp.async.wait_group %0;\n" : : "n"(Groups));
}

__device__ __forceinline__ float bf_lo(uint32_t u) { return __uint_as_float(u << 16); }
__device__ __forceinline__ float bf_hi(uint32_t u) { return __uint_as_float(u & 0xffff0000u); }
__device__ __forceinline__ float bf2f(uint16_t u) { return __uint_as_float(uint32_t(u) << 16); }
__device__ __forceinline__ float round_bf16(float x) { return __bfloat162float(__float2bfloat16_rn(x)); }

__device__ __forceinline__ uint16_t f2bf_bits(float x) {
  return __bfloat16_as_ushort(__float2bfloat16_rn(x));
}

__device__ __forceinline__ uint32_t pack_bf16x2(float lo, float hi) {
  return uint32_t(f2bf_bits(lo)) | (uint32_t(f2bf_bits(hi)) << 16);
}

__device__ __forceinline__ uint32_t pack_half2(float lo, float hi) {
  const __half2 h = __floats2half2_rn(lo, hi);
  return *reinterpret_cast<const uint32_t*>(&h);
}

// 2 E4M3 codes (low byte first) -> 2 floats (exact via FP16).
__device__ __forceinline__ float2 e4m3x2_to_float2(uint16_t u) {
  __half2_raw h = __nv_cvt_fp8x2_to_halfraw2(static_cast<__nv_fp8x2_storage_t>(u), __NV_E4M3);
  return __half22float2(*reinterpret_cast<__half2*>(&h));
}

// 2 E4M3 codes -> packed FP16 pair (exact).
__device__ __forceinline__ uint32_t e4m3x2_to_half2(uint16_t u) {
  __half2_raw h = __nv_cvt_fp8x2_to_halfraw2(static_cast<__nv_fp8x2_storage_t>(u), __NV_E4M3);
  return uint32_t(h.x) | (uint32_t(h.y) << 16);
}

// float -> E4M3 code, round to nearest even after saturating to +-448 (torch clamp + .to(float8_e4m3fn)).
__device__ __forceinline__ uint8_t f2e4m3(float x) {
  return static_cast<uint8_t>(__nv_cvt_float_to_fp8(fminf(fmaxf(x, -448.f), 448.f), __NV_SATFINITE, __NV_E4M3));
}

// Q4 KV cache codebook (matches attn.cu kQ4Lut / quant_q4_nibble).
__constant__ float kQ4Lut[16] = {
  0.0f, 0.125f, 0.25f, 0.5f, 0.75f, 1.125f, 1.75f, 2.75f,
  -0.0f, -0.125f, -0.25f, -0.5f, -0.75f, -1.125f, -1.75f, -2.75f
};

__device__ __forceinline__ uint8_t quant_q4_nibble(float x) {
  const uint8_t s = x < 0.f ? 8u : 0u;
  const float a = fabsf(x);
  uint8_t m = 0u;
  if (a >= 2.25f) m = 7u;
  else if (a >= 1.4375f) m = 6u;
  else if (a >= 0.9375f) m = 5u;
  else if (a >= 0.625f) m = 4u;
  else if (a >= 0.375f) m = 3u;
  else if (a >= 0.1875f) m = 2u;
  else if (a >= 0.0625f) m = 1u;
  return s | m;
}

__device__ __forceinline__ uint32_t q4_byte_to_bf16x2(uint8_t b) {
  return pack_bf16x2(kQ4Lut[b & 0xfu], kQ4Lut[b >> 4]);
}

__device__ __forceinline__ uint32_t q4_byte_to_half2(uint8_t b) {
  return pack_half2(kQ4Lut[b & 0xfu], kQ4Lut[b >> 4]);
}

// Swizzled [rows][Chunks x 16 B] tile: element offset (in 16-bit units) of 16-byte chunk `chunk` of `row`.
// XOR with row & 7 spreads the 8 rows of an ldmatrix 8x8 matrix over distinct banks.
template <int Chunks>
__device__ __forceinline__ int swz(int row, int chunk) {
  return row * (Chunks * 8) + ((chunk ^ (row & 7)) << 3);
}

}  // namespace ingine::kernels::prefill
