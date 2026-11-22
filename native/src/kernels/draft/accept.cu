// Speculative acceptance (engine.Engine._accept): greedy prefix match and speculative rejection sampling.
#include "kernels/draft/draft.h"
#include "kernels/draft/draft_device.cuh"

#include <stdexcept>

namespace ingine::kernels::draft {
namespace {

__global__ void greedy_accept_kernel(const std::int32_t* __restrict__ target, const std::int32_t* __restrict__ drafts,
                                     int n, AcceptResult* __restrict__ out) {
  int acc = 0;
  while (acc < n && drafts[acc] == target[acc]) ++acc;
  out->accepted = acc;
  out->next = target[acc];
}

constexpr int R_THREADS = 256;
constexpr int R_PER = 4;  // elements per thread in the residual kernels
constexpr int R_SPAN = R_THREADS * R_PER;

__host__ __device__ __forceinline__ int residual_blocks(int V) { return (V + R_SPAN - 1) / R_SPAN; }

// q_r(x) = softmax(draft_logits[r] / T)(x) from the row's (max, sum) statistics
__device__ __forceinline__ float q_of(const float* draft_logits, int V, int r, int x, float temperature,
                                      const float* qstats) {
  const float s = __fdiv_rn(draft_logits[size_t(r) * V + x], temperature);
  return __fdiv_rn(expf(__fsub_rn(s, qstats[2 * r])), qstats[2 * r + 1]);
}

// accepted count (every CTA computes it), then the residual row and this CTA's part of its sum (ws[block])
__global__ void __launch_bounds__(R_THREADS) residual_kernel(const float* p, const float* draft_logits, int V,
                                                             const std::int32_t* drafts, int n, float temperature,
                                                             const float* qstats, const float* u, float* ws,
                                                             float* resid, AcceptResult* out) {
  __shared__ int sh_acc;
  __shared__ float scratch[33];
  pdl_trigger();
  pdl_wait();
  if (threadIdx.x == 0) {
    int acc = 0;
    for (; acc < n; ++acc) {
      const int d = drafts[acc];
      const float qd = q_of(draft_logits, V, acc, d, temperature, qstats);
      const float pd = p[size_t(acc) * V + d];
      if (!(__fmul_rn(u[acc], qd) < pd)) break;
    }
    sh_acc = acc;
    if (blockIdx.x == 0) out->accepted = acc;
  }
  __syncthreads();
  const int a = sh_acc;
  const float* pa = p + size_t(a) * V;
  float s = 0.f;
  const int x0 = blockIdx.x * R_SPAN;
#pragma unroll
  for (int j = 0; j < R_PER; ++j) {
    const int x = x0 + j * R_THREADS + threadIdx.x;
    if (x >= V) break;
    const float w = a < n ? fmaxf(__fsub_rn(pa[x], q_of(draft_logits, V, a, x, temperature, qstats)), 0.f) : pa[x];
    resid[x] = w;
    s += w;
  }
  s = block_sum(s, scratch);
  if (threadIdx.x == 0) ws[blockIdx.x] = s;
}

// all-zero residual: draw from p_a instead (the partial sums are added in a fixed order)
__global__ void __launch_bounds__(R_THREADS) residual_fallback_kernel(const float* p, int V, const float* ws,
                                                                      float* resid, const AcceptResult* out) {
  __shared__ float scratch[33];
  pdl_trigger();
  pdl_wait();
  float s = 0.f;
  for (int b = threadIdx.x; b < residual_blocks(V); b += R_THREADS) s += ws[b];
  if (block_sum(s, scratch) > 0.f) return;
  const float* pa = p + size_t(out->accepted) * V;
  const int x0 = blockIdx.x * R_SPAN;
#pragma unroll
  for (int j = 0; j < R_PER; ++j) {
    const int x = x0 + j * R_THREADS + threadIdx.x;
    if (x < V) resid[x] = pa[x];
  }
}

}  // namespace

void greedy_accept(const std::int32_t* target, const std::int32_t* drafts, int n, AcceptResult* out,
                   cudaStream_t stream) {
  if (n < 0 || n > GAMMA) throw std::invalid_argument("greedy_accept: n must be in 0..7");
  greedy_accept_kernel<<<1, 1, 0, stream>>>(target, drafts, n, out);
  CUDA_CHECK(cudaGetLastError());
}

std::size_t sampled_accept_workspace_floats(int V) { return std::size_t(residual_blocks(V)); }

void sampled_accept(const float* p, const float* draft_logits, int V, const std::int32_t* drafts, int n,
                    float temperature, const float* qstats, const float* u, float* ws, float* resid,
                    AcceptResult* out, cudaStream_t stream) {
  if (n < 1 || n > GAMMA) throw std::invalid_argument("sampled_accept: n must be in 1..7");
  if (!(temperature > 0.f)) throw std::invalid_argument("sampled_accept: temperature must be positive");
  const dim3 blocks(residual_blocks(V));
  launch_pdl(residual_kernel, blocks, dim3(R_THREADS), stream, p, draft_logits, V, drafts, n, temperature, qstats, u,
             ws, resid, out);
  launch_pdl(residual_fallback_kernel, blocks, dim3(R_THREADS), stream, p, V, static_cast<const float*>(ws), resid,
             static_cast<const AcceptResult*>(out));
}

}  // namespace ingine::kernels::draft
