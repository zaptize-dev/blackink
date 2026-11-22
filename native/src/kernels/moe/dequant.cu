// Routed-expert decode to BF16 (ported from engine/csrc/prefill/dequant.cu; kernel bodies unchanged).
//
// A prefill that multiplies many tokens per expert can decode the selected experts' matrices once into a
// BF16 scratch buffer and run grouped GEMMs on it. Decoding follows the checkpoint semantics exactly
// (FP32 product, one BF16 rounding), matching model.decode_experts:
//   NVFP4: value = e2m1(code) * e4m3(block scale per 16) * factor[expert]
//   MXFP8: value = e4m3(code) * 2^(e8m0 per 32 - 127)
// One thread decodes one scale block (16 or 32 values) and writes it with 16-byte stores.
#include "kernels/moe/moe.h"

#include "core/device.h"

#include <cuda_bf16.h>
#include <cuda_fp8.h>

#include <stdexcept>

namespace ingine::kernels::moe {
namespace {

__constant__ float kE2M1[16] = {0.f, .5f, 1.f, 1.5f, 2.f, 3.f, 4.f, 6.f, -0.f, -.5f, -1.f, -1.5f, -2.f, -3.f, -4.f, -6.f};

__global__ void dequant_nvfp4(const uint8_t* __restrict__ codes, const __nv_fp8_e4m3* __restrict__ scales,
                              const float* __restrict__ factor, const int32_t* __restrict__ experts,
                              __nv_bfloat16* __restrict__ out, int64_t blocks_per_expert, int64_t total) {
    int64_t b = blockIdx.x * (int64_t)blockDim.x + threadIdx.x;
    if (b >= total) return;
    int64_t slot = b / blocks_per_expert, within = b % blocks_per_expert;
    int64_t e = experts[slot];
    int64_t src = e * blocks_per_expert + within;
    // the reference forms e2m1 * block_scale, then multiplies by the expert factor (both FP32)
    const float scale = float(scales[src]), f = factor[e];
    uint2 c = *reinterpret_cast<const uint2*>(codes + src * 8);
    uint32_t w[2] = {c.x, c.y};
    __align__(16) __nv_bfloat16 v[16];
#pragma unroll
    for (int i = 0; i < 8; ++i) {
        uint32_t byte = (w[i / 4] >> (8 * (i % 4))) & 0xFF;
        v[2 * i] = __float2bfloat16((kE2M1[byte & 0xF] * scale) * f);
        v[2 * i + 1] = __float2bfloat16((kE2M1[byte >> 4] * scale) * f);
    }
    uint4* dst = reinterpret_cast<uint4*>(out + (slot * blocks_per_expert + within) * 16);
    dst[0] = reinterpret_cast<uint4*>(v)[0];
    dst[1] = reinterpret_cast<uint4*>(v)[1];
}

__global__ void dequant_mxfp8(const __nv_fp8_e4m3* __restrict__ codes, const uint8_t* __restrict__ scales,
                              const int32_t* __restrict__ experts, __nv_bfloat16* __restrict__ out,
                              int64_t blocks_per_expert, int64_t total) {
    int64_t b = blockIdx.x * (int64_t)blockDim.x + threadIdx.x;
    if (b >= total) return;
    int64_t slot = b / blocks_per_expert, within = b % blocks_per_expert;
    int64_t src = experts[slot] * blocks_per_expert + within;
    float s = exp2f(float(scales[src]) - 127.f);
    const uint4* c = reinterpret_cast<const uint4*>(codes + src * 32);
    uint4* dst = reinterpret_cast<uint4*>(out + (slot * blocks_per_expert + within) * 32);
#pragma unroll
    for (int q = 0; q < 2; ++q) {
        uint4 raw = c[q];
        const __nv_fp8_e4m3* f = reinterpret_cast<const __nv_fp8_e4m3*>(&raw);
        __align__(16) __nv_bfloat16 v[16];
#pragma unroll
        for (int i = 0; i < 16; ++i) v[i] = __float2bfloat16(float(f[i]) * s);
        dst[2 * q] = reinterpret_cast<uint4*>(v)[0];
        dst[2 * q + 1] = reinterpret_cast<uint4*>(v)[1];
    }
}

}  // namespace

// codes [128, R, C/2] uint8 (NVFP4) or [128, R, C] float8_e4m3fn (MXFP8); experts int32 [n];
// out bf16 [n, R, C].
void dequant_experts(const void* codes, const void* scales, const float* factor, const std::int32_t* experts, int n,
                     int rows, int cols, Format fmt, __nv_bfloat16* out, cudaStream_t stream) {
    const bool nv = fmt == Format::NVFP4;
    if (n < 0 || rows <= 0 || cols % (nv ? 16 : 32) != 0 || (nv && !factor))
        throw std::invalid_argument("moe: dequant_experts: bad arguments");
    if (n == 0) return;
    int64_t per_expert_blocks = int64_t(rows) * (cols / (nv ? 16 : 32));
    int64_t total = n * per_expert_blocks;
    int threads = 256;
    int64_t grid = (total + threads - 1) / threads;
    if (nv) {
        dequant_nvfp4<<<grid, threads, 0, stream>>>(
            static_cast<const uint8_t*>(codes), static_cast<const __nv_fp8_e4m3*>(scales), factor, experts, out,
            per_expert_blocks, total);
    } else {
        dequant_mxfp8<<<grid, threads, 0, stream>>>(
            static_cast<const __nv_fp8_e4m3*>(codes), static_cast<const uint8_t*>(scales), experts, out,
            per_expert_blocks, total);
    }
    CUDA_CHECK(cudaGetLastError());
}

namespace {

// Find per-expert amax over MXFP8 blocks (32 elements per E8M0 scale). One CTA per expert.
__global__ void mxfp8_amax_kernel(const __nv_fp8_e4m3* __restrict__ codes, const uint8_t* __restrict__ scales,
                                  int64_t blocks32_per_expert, float* __restrict__ factors) {
    __shared__ float s_max[8];
    const int e = blockIdx.x;
    const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
    const int64_t base = int64_t(e) * blocks32_per_expert;
    float amax = 0.f;
    for (int64_t b = tid; b < blocks32_per_expert; b += blockDim.x) {
        const float s = exp2f(float(scales[base + b]) - 127.f);
        const uint4* c = reinterpret_cast<const uint4*>(codes + (base + b) * 32);
        float bmax = 0.f;
#pragma unroll
        for (int q = 0; q < 2; ++q) {
            const uint4 raw = c[q];
            const __nv_fp8_e4m3* f = reinterpret_cast<const __nv_fp8_e4m3*>(&raw);
#pragma unroll
            for (int i = 0; i < 16; ++i) bmax = fmaxf(bmax, fabsf(float(f[i])));
        }
        amax = fmaxf(amax, bmax * s);
    }
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, o));
    if (lane == 0) s_max[warp] = amax;
    __syncthreads();
    if (warp == 0) {
        float v = lane < 8 ? s_max[lane] : 0.f;
#pragma unroll
        for (int o = 4; o > 0; o >>= 1) v = fmaxf(v, __shfl_xor_sync(0xffffffffu, v, o));
        // NVFP4 outer factor = amax / (6.0 * 448.0) so that block_amax / factor <= 448 * 6
        if (lane == 0) factors[e] = v > 0.f ? (v / (6.f * 448.f)) : 1.f;
    }
}

__device__ __forceinline__ uint8_t quant_e2m1(float x) {
    const uint8_t sign = x < 0.f ? 8u : 0u;
    const float a = fabsf(x);
    uint8_t mag = 0u;
    if (a >= 5.0f) mag = 7u;       // 6.0
    else if (a >= 3.5f) mag = 6u;  // 4.0
    else if (a >= 2.5f) mag = 5u;  // 3.0
    else if (a >= 1.75f) mag = 4u; // 2.0
    else if (a >= 1.25f) mag = 3u; // 1.5
    else if (a >= 0.75f) mag = 2u; // 1.0
    else if (a >= 0.25f) mag = 1u; // 0.5
    return sign | mag;
}

// One thread converts one 32-element MXFP8 block into two 16-element NVFP4 blocks.
__global__ void mxfp8_to_nvfp4_kernel(const __nv_fp8_e4m3* __restrict__ codes, const uint8_t* __restrict__ scales,
                                      const float* __restrict__ factors, int64_t blocks32_per_expert, int64_t total32,
                                      uint8_t* __restrict__ out_codes, __nv_fp8_e4m3* __restrict__ out_scales) {
    const int64_t b = blockIdx.x * int64_t(blockDim.x) + threadIdx.x;
    if (b >= total32) return;
    const int e = int(b / blocks32_per_expert);
    const float s = exp2f(float(scales[b]) - 127.f);
    const float inv_f = 1.f / fmaxf(factors[e], 1e-30f);
    const uint4* c = reinterpret_cast<const uint4*>(codes + b * 32);
#pragma unroll
    for (int half = 0; half < 2; ++half) {
        const uint4 raw = c[half];
        const __nv_fp8_e4m3* f = reinterpret_cast<const __nv_fp8_e4m3*>(&raw);
        float vals[16];
        float bmax = 0.f;
#pragma unroll
        for (int i = 0; i < 16; ++i) {
            vals[i] = float(f[i]) * s;
            bmax = fmaxf(bmax, fabsf(vals[i]));
        }
        const __nv_fp8_e4m3 bs_fp8 = __nv_fp8_e4m3(fminf((bmax * inv_f) * (1.f / 6.f), 448.f));
        const float bs = float(bs_fp8);
        const int64_t b16 = b * 2 + half;
        out_scales[b16] = bs_fp8;
        const float denom = bs * factors[e];
        const float inv_d = denom > 0.f ? (1.f / denom) : 0.f;
        uint8_t packed[8];
#pragma unroll
        for (int i = 0; i < 8; ++i) {
            const uint8_t lo = quant_e2m1(vals[2 * i] * inv_d);
            const uint8_t hi = quant_e2m1(vals[2 * i + 1] * inv_d);
            packed[i] = lo | uint8_t(hi << 4);
        }
        *reinterpret_cast<uint2*>(out_codes + b16 * 8) = *reinterpret_cast<const uint2*>(packed);
    }
}

}  // namespace

void mxfp8_to_nvfp4(const void* mxfp8_codes, const void* mxfp8_scales, int rows, int cols,
                    void* nvfp4_codes, void* nvfp4_scales, float* nvfp4_factors, cudaStream_t stream) {
    if (rows <= 0 || cols <= 0 || cols % 32 != 0 || !mxfp8_codes || !mxfp8_scales || !nvfp4_codes ||
        !nvfp4_scales || !nvfp4_factors)
        throw std::invalid_argument("moe: mxfp8_to_nvfp4: bad arguments");
    const int64_t blocks32_per_expert = int64_t(rows) * (cols / 32);
    const int64_t total32 = 128 * blocks32_per_expert;
    mxfp8_amax_kernel<<<128, 256, 0, stream>>>(
        static_cast<const __nv_fp8_e4m3*>(mxfp8_codes), static_cast<const uint8_t*>(mxfp8_scales),
        blocks32_per_expert, nvfp4_factors);
    CUDA_CHECK(cudaGetLastError());
    const int threads = 256;
    const int64_t grid = (total32 + threads - 1) / threads;
    mxfp8_to_nvfp4_kernel<<<grid, threads, 0, stream>>>(
        static_cast<const __nv_fp8_e4m3*>(mxfp8_codes), static_cast<const uint8_t*>(mxfp8_scales),
        nvfp4_factors, blocks32_per_expert, total32,
        static_cast<uint8_t*>(nvfp4_codes), static_cast<__nv_fp8_e4m3*>(nvfp4_scales));
    CUDA_CHECK(cudaGetLastError());
}

}  // namespace ingine::kernels::moe
