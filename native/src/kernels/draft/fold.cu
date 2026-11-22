// Load-time fold of the Markov head's hidden_out into the unembedding (dspark.Drafter.__init__):
// fold_w = bf16(U @ hw), fold_b = bf16(U @ hb), computed in FP64 from the dequantized unembedding.
#include "kernels/draft/draft.h"
#include "kernels/draft/draft_device.cuh"

#include "core/device.h"

#include <cuda_fp16.h>
#include <cuda_fp8.h>

namespace ingine::kernels::draft {
namespace {

constexpr int FOLD_ROWS = 32;  // unembedding rows per CTA
constexpr int FOLD_K = 16;     // k tile
constexpr int FOLD_THREADS = RANK;
static_assert(H % FOLD_K == 0);

// weights.dequant: BF16 rows as is, FP8 rows -> bf16(float(q) * scale), NVFP4 rows -> bf16(e2m1 * bscale * scale)
__device__ __forceinline__ double u_at(const void* u, const float* scale, const void* bscale, int row, int k) {
  if (bscale != nullptr) {
    const uint8_t byte = static_cast<const uint8_t*>(u)[(size_t(row) * H + k) >> 1];
    const uint8_t nib = (k & 1) ? (byte >> 4) : (byte & 0xFu);
    const float kE2M1[16] = {0.f, .5f, 1.f, 1.5f, 2.f, 3.f, 4.f, 6.f, -0.f, -.5f, -1.f, -1.5f, -2.f, -3.f, -4.f, -6.f};
    const __nv_fp8_storage_t bs_q = static_cast<const __nv_fp8_storage_t*>(bscale)[size_t(row) * (H / 16) + (k >> 4)];
    const __half_raw hr = __nv_cvt_fp8_to_halfraw(bs_q, __NV_E4M3);
    const float bs = __half2float(__half(hr));
    return double(bf_round(__fmul_rn(kE2M1[nib] * bs, scale[row])));
  }
  if (scale == nullptr) return double(bf(static_cast<const __nv_bfloat16*>(u)[size_t(row) * H + k]));
  const __nv_fp8_storage_t q = static_cast<const __nv_fp8_storage_t*>(u)[size_t(row) * H + k];
  const __half_raw hr = __nv_cvt_fp8_to_halfraw(q, __NV_E4M3);
  const float f = __half2float(__half(hr));
  return double(bf_round(__fmul_rn(f, scale[row])));
}

// torch double -> bfloat16: through float (round to nearest), then to BF16 (round to nearest even)
__device__ __forceinline__ __nv_bfloat16 to_bf16(double x) { return __float2bfloat16_rn(__double2float_rn(x)); }

__global__ void __launch_bounds__(FOLD_THREADS) fold_w_kernel(const void* __restrict__ u,
                                                              const float* __restrict__ scale,
                                                              const void* __restrict__ bscale, int rows,
                                                              const __nv_bfloat16* __restrict__ hw,
                                                              __nv_bfloat16* __restrict__ fold_w) {
  __shared__ double us[FOLD_ROWS][FOLD_K];
  __shared__ double hs[FOLD_K][RANK];
  const int r0 = blockIdx.x * FOLD_ROWS, t = threadIdx.x;
  double acc[FOLD_ROWS];
#pragma unroll
  for (int r = 0; r < FOLD_ROWS; ++r) acc[r] = 0.0;
  for (int k0 = 0; k0 < H; k0 += FOLD_K) {
    for (int idx = t; idx < FOLD_ROWS * FOLD_K; idx += FOLD_THREADS) {
      const int r = idx / FOLD_K, k = idx % FOLD_K;
      us[r][k] = r0 + r < rows ? u_at(u, scale, bscale, r0 + r, k0 + k) : 0.0;
    }
#pragma unroll
    for (int k = 0; k < FOLD_K; ++k) hs[k][t] = double(bf(hw[size_t(k0 + k) * RANK + t]));
    __syncthreads();
#pragma unroll
    for (int k = 0; k < FOLD_K; ++k) {
      const double h = hs[k][t];
#pragma unroll
      for (int r = 0; r < FOLD_ROWS; ++r) acc[r] = fma(us[r][k], h, acc[r]);
    }
    __syncthreads();
  }
#pragma unroll
  for (int r = 0; r < FOLD_ROWS; ++r)
    if (r0 + r < rows) fold_w[size_t(r0 + r) * RANK + t] = to_bf16(acc[r]);
}

// warp per row
__global__ void __launch_bounds__(256) fold_b_kernel(const void* __restrict__ u, const float* __restrict__ scale,
                                                     const void* __restrict__ bscale, int rows,
                                                     const __nv_bfloat16* __restrict__ hb,
                                                     __nv_bfloat16* __restrict__ fold_b) {
  const int row = (blockIdx.x * blockDim.x + threadIdx.x) >> 5, lane = threadIdx.x & 31;
  if (row >= rows) return;
  double acc = 0.0;
  for (int k = lane; k < H; k += 32) acc = fma(u_at(u, scale, bscale, row, k), double(bf(hb[k])), acc);
#pragma unroll
  for (int o = 16; o > 0; o >>= 1) acc += __shfl_xor_sync(0xffffffffu, acc, o);
  if (lane == 0) fold_b[row] = to_bf16(acc);
}

}  // namespace

void fold_head(const void* u, const float* u_scale, int rows, const __nv_bfloat16* hw, const __nv_bfloat16* hb,
               __nv_bfloat16* fold_w, __nv_bfloat16* fold_b, cudaStream_t stream, const void* u_bscale) {
  if (rows <= 0) return;
  fold_w_kernel<<<(rows + FOLD_ROWS - 1) / FOLD_ROWS, FOLD_THREADS, 0, stream>>>(u, u_scale, u_bscale, rows, hw,
                                                                                 fold_w);
  CUDA_CHECK(cudaGetLastError());
  fold_b_kernel<<<(rows * 32 + 255) / 256, 256, 0, stream>>>(u, u_scale, u_bscale, rows, hb, fold_b);
  CUDA_CHECK(cudaGetLastError());
}

}  // namespace ingine::kernels::draft
