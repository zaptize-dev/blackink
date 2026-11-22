// router: FP32 logits, sigmoid, biased top-8, normalised x8 weights, for M = 1..8 tokens.
//
// Design. The router matrix is small (128 x 2304: 1.2 MB FP32 / 0.6 MB BF16) but top-8 needs all 128
// logits of a token, so the op is one cluster of CL = 16 CTAs (8 if the device cannot co-schedule 16):
// * every CTA copies the M BF16 activation rows into shared memory (cp.async, issued first), then CTA r
//   loads its rows [r * 128/CL, +128/CL) whole into registers (one warp per row, 16-byte streaming loads:
//   FP32 18 per lane, BF16 9), so the weight stream never waits on activation latency. Spreading the
//   1.2 MB over 16 SMs keeps per-SM traffic at ~74 KB.
// * the M logits of a row are reduce-scattered over the warp (lane m holds token m), gathered per token
//   and sent to CTA m, which owns token m's selection, as 16-byte st.async stores (4 rows each) completing
//   on CTA m's mbarrier (dsmem.cuh; a relaxed cluster arrive at kernel start + wait before the remote
//   stores guarantees every CTA is resident and its barrier initialised). CTAs that own no token exit
//   after sending.
// * selection in CTA m: thread e computes v_e = sigmoid(logit_e) + bias_e and its rank among the 128
//   values (#greater + #equal with a lower index: ties go to the lower expert) as the number of greater
//   unique keys (order-preserving bits of v_e, then 127 - e), counted with broadcast shared reads by all
//   threads of the CTA (each a part of the 128 keys), so the 8 winners land in rank order (descending
//   score) without a serial argmax chain; lanes 0..7 then write idx and wts = s / (sum s + 1e-20) * 8.
// * pdl=True: weights and bias are loaded before griddepcontrol.wait (pdl_wait_fenced: ptxas would sink
//   them below a plain wait), the activations after it.
// Ported from engine/csrc/dense/router.cu (torch launchers -> raw pointers; mbarrier exchange).
#include <cooperative_groups.h>

#include "kernels/common/l2.cuh"
#include "kernels/dense/cluster.cuh"
#include "kernels/dense/dense.cuh"
#include "kernels/dense/dense.h"
#include "kernels/dense/dsmem.cuh"

#include <type_traits>

namespace cg = cooperative_groups;

namespace ingine::kernels::dense {
namespace {

constexpr int EXPERTS = 128, TOPK = 8;
// L2 class of the weight loads (l2.cuh): none (see gates.cu: evict-last residency made the step slower)
constexpr l2::Evict WEIGHT_L2 = l2::Evict::none;

template <int CL, bool WF32, int M>
__global__ void __launch_bounds__(32 * EXPERTS / CL) router_kernel(const __nv_bfloat16* __restrict__ x,
                                                                   const void* __restrict__ w,
                                                                   const float* __restrict__ bias,
                                                                   int* __restrict__ idx, float* __restrict__ wts,
                                                                   uint32_t zero) {
  constexpr int RPC = EXPERTS / CL, NT = 32 * RPC;
  constexpr int EPC = WF32 ? 4 : 8;  // weights per 16-byte chunk
  constexpr int CH = H / (EPC * 32);
  __shared__ __align__(16) __nv_bfloat16 xs[M][H];
  __shared__ __align__(16) float logits[EXPERTS];  // token `rank`'s logits (CTAs rank < M)
  __shared__ __align__(8) uint64_t mbar;           // completes when the 128 logits arrived
  __shared__ __align__(16) float part[M][RPC];     // this CTA's logits
  __shared__ unsigned long long keys[EXPERTS];    // selection keys of sigmoid + bias
  __shared__ int cnt[NT];                          // partial ranks
  __shared__ float sel_s[TOPK];
  __shared__ int sel_e[TOPK];

  cg::cluster_group cluster = cg::this_cluster();
  const int rank = cluster.block_rank();
  const uint32_t logits_u = smem_u32(logits), mb = smem_u32(&mbar);
  if (threadIdx.x == 0) {
    mb_init(mb);
    fence_mbarrier_init();
    if (rank < M) mb_expect(mb, 4u * EXPERTS);
  }
  cluster_arrive_relaxed();
  const int t = threadIdx.x, warp = t >> 5, lane = t & 31;
  const int row = rank * RPC + warp;

  // weights and bias (constants) before the PDL wait, activations after it
  const uint8_t* wr = reinterpret_cast<const uint8_t*>(w) + size_t(row) * H * (WF32 ? 4 : 2);
  uint4 wv[CH];
  const std::uint64_t wpol = l2::policy<WEIGHT_L2>();
#pragma unroll
  for (int i = 0; i < CH; ++i) wv[i] = l2::ld_stream<WEIGHT_L2>(wr + (size_t(i) * 32 + lane) * 16, wpol);
  const float b = t < EXPERTS ? __ldg(bias + t) : 0.f;  // for the selection phase

  pdl_wait_fenced(zero);  // keeps the weight loads above ahead of the wait
  pdl_trigger();
  for (int k = t; k < M * H / 8; k += NT) cp_async16(&xs[0][0] + k * 8, x + size_t(k) * 8);
  asm volatile("cp.async.commit_group;" ::: "memory");
  asm volatile("cp.async.wait_all;" ::: "memory");
  __syncthreads();

  float acc[M];
#pragma unroll
  for (int m = 0; m < M; ++m) acc[m] = 0.f;
#pragma unroll
  for (int i = 0; i < CH; ++i) {
    const int col = (i * 32 + lane) * EPC;
    float wf[EPC], xf[EPC];
    if constexpr (WF32) {
      wf[0] = __uint_as_float(wv[i].x); wf[1] = __uint_as_float(wv[i].y);
      wf[2] = __uint_as_float(wv[i].z); wf[3] = __uint_as_float(wv[i].w);
    } else {
      unpack_bf16x8(wv[i], wf);
    }
#pragma unroll
    for (int m = 0; m < M; ++m) {
      if constexpr (WF32) {
        const uint2 u = *reinterpret_cast<const uint2*>(&xs[m][col]);
        xf[0] = bf_lo(u.x); xf[1] = bf_hi(u.x); xf[2] = bf_lo(u.y); xf[3] = bf_hi(u.y);
      } else {
        unpack_bf16x8(*reinterpret_cast<const uint4*>(&xs[m][col]), xf);
      }
      float a = acc[m];
#pragma unroll
      for (int j = 0; j < EPC; ++j) a = fmaf(wf[j], xf[j], a);
      acc[m] = a;
    }
  }
  const float logit = reduce_scatter<32, M>(acc);  // lane m: token m

  // gather the CTA's RPC logits of each token, then one 16-byte store per (token, 4 rows)
  if (lane < M) part[lane][warp] = logit;
  __syncthreads();
  cluster_wait();  // every CTA has started and initialised its barrier
  if (t < M * (RPC / 4)) {
    const int tok = t % M, grp = t / M;
    st_async4(remote(logits_u + 4u * (rank * RPC + grp * 4), tok),
              *reinterpret_cast<const float4*>(&part[tok][grp * 4]), remote(mb, tok));
  }
  if (rank >= M) return;
  mb_wait(mb);

  // selection: unique 39-bit keys (order-preserving bits of v, then 127 - e: ties go to the lower expert),
  // rank = number of greater keys, counted by all NT threads (NT / 128 parts of the key range)
  const int m = rank;
  float s = 0.f;
  if (t < EXPERTS) {
    s = 1.f / (1.f + expf(-logits[t]));
    const uint32_t u = __float_as_uint(s + b + 0.f);  // + 0: -0 -> +0, which compares equal
    const uint32_t o = (u & 0x80000000u) ? ~u : (u | 0x80000000u);
    keys[t] = (static_cast<unsigned long long>(o) << 7) | static_cast<unsigned>(EXPERTS - 1 - t);
  }
  __syncthreads();
  {
    constexpr int PARTS = NT / EXPERTS, W = EXPERTS / PARTS;
    const int e = t % EXPERTS, part = t / EXPERTS;
    const unsigned long long mine = keys[e];
    int c[4] = {0, 0, 0, 0};
#pragma unroll
    for (int j = 0; j < W; j += 4)
#pragma unroll
      for (int k = 0; k < 4; ++k) c[k] += keys[part * W + j + k] > mine;
    cnt[t] = c[0] + c[1] + c[2] + c[3];
  }
  __syncthreads();
  if (t < EXPERTS) {
    int r = 0;
#pragma unroll
    for (int p = 0; p < NT / EXPERTS; ++p) r += cnt[p * EXPERTS + t];
    if (r < TOPK) {
      sel_e[r] = t;
      sel_s[r] = s;
    }
  }
  __syncthreads();
  if (t < TOPK) {
    const float my_s = sel_s[t];
    float sum = my_s;
#pragma unroll
    for (int o = 4; o > 0; o >>= 1) sum += __shfl_xor_sync(0xffu, sum, o);
    idx[m * TOPK + t] = sel_e[t];
    wts[m * TOPK + t] = my_s / (sum + 1e-20f) * 8.f;
  }
}

template <int CL, bool WF32>
void launch_router(int M, const __nv_bfloat16* x, const void* w, const float* bias, int* idx, float* wts, bool pdl,
                   cudaStream_t stream) {
  auto go = [&](auto mc) {
    constexpr int MM = decltype(mc)::value;
    launch_ex(router_kernel<CL, WF32, MM>, dim3(CL), 32 * EXPERTS / CL, 0, CL, pdl, stream, x, w, bias, idx, wts,
              0u);
  };
  switch (M) {
    case 1: go(std::integral_constant<int, 1>{}); break;
    case 2: go(std::integral_constant<int, 2>{}); break;
    case 3: go(std::integral_constant<int, 3>{}); break;
    case 4: go(std::integral_constant<int, 4>{}); break;
    case 5: go(std::integral_constant<int, 5>{}); break;
    case 6: go(std::integral_constant<int, 6>{}); break;
    case 7: go(std::integral_constant<int, 7>{}); break;
    case 8: go(std::integral_constant<int, 8>{}); break;
    default: require(false, "router: M must be in 1..8");
  }
}

// 16-CTA clusters need the non-portable opt-in and a GPC with 16 free SMs; fall back to 8 otherwise.
int router_cluster_size() {
  static const int cl = [] {
    auto k = router_kernel<16, true, 8>;
    if (cudaFuncSetAttribute(k, cudaFuncAttributeNonPortableClusterSizeAllowed, 1) != cudaSuccess) {
      (void)cudaGetLastError();
      return 8;
    }
    cudaLaunchConfig_t cfg = {};
    cfg.gridDim = dim3(16);
    cfg.blockDim = dim3(32 * EXPERTS / 16);
    cudaLaunchAttribute attr[1];
    attr[0].id = cudaLaunchAttributeClusterDimension;
    attr[0].val.clusterDim.x = 16; attr[0].val.clusterDim.y = 1; attr[0].val.clusterDim.z = 1;
    cfg.attrs = attr; cfg.numAttrs = 1;
    int n = 0;
    if (cudaOccupancyMaxActiveClusters(&n, k, &cfg) != cudaSuccess) { (void)cudaGetLastError(); n = 0; }
    return n > 0 ? 16 : 8;
  }();
  return cl;
}

}  // namespace

void router(const __nv_bfloat16* x, int M, const void* w, bool w_fp32, const float* bias, int* idx, float* wts,
            bool pdl, cudaStream_t stream) {
  require(x && w && bias && idx && wts, "router: null argument");
  const bool c16 = router_cluster_size() == 16;
  if (w_fp32) {
    if (c16) launch_router<16, true>(M, x, w, bias, idx, wts, pdl, stream);
    else launch_router<8, true>(M, x, w, bias, idx, wts, pdl, stream);
  } else {
    if (c16) launch_router<16, false>(M, x, w, bias, idx, wts, pdl, stream);
    else launch_router<8, false>(M, x, w, bias, idx, wts, pdl, stream);
  }
}

}  // namespace ingine::kernels::dense
