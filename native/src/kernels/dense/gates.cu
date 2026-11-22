// message_gates: loop-pass message gates (rank-64 down projection, tanh forget/gain, gated mix, RMS norm).
//
// Design. Per gate the work is a chain of all-to-all dependencies (row norms -> 64 down outputs -> 2304
// gated channels -> their RMS norm), so each gate is one cluster of CL = 16 CTAs (non-portable size; with
// 8 the FP32 forget/gain slices would not fit in shared memory) and the dependencies are crossed with
// distributed shared memory instead of extra launches. Per-SM DRAM concurrency is the limit for such a
// small grid (~110 KB per SM at G = 3), so every weight byte is requested before the PDL wait. CTA r of
// gate g:
// * issues cp.async copies of its forget/gain row slices (rows [r * 144, +144), padded rows in shared
//   memory) and 16-byte streaming loads of its 4 rows of `w_down` (2 warps per row: half 0 against n(h),
//   half 1 against n(msg)) into registers;
// * row norms and down projection, up to 4 tokens at a time (one for FP32 forget/gain slices: no room for
//   more staging buffers): every thread loads its 9 inputs of h and of msg per token once and keeps them in
//   registers; one block sum gives the group's sums of squares (each CTA computes all of them, so all CTAs
//   normalise identically); bf16(x * rinv) is staged in shared memory, one buffer per token, and after one
//   barrier every warp dots its row half against each. (The inputs are not read twice: with the slices in
//   shared memory the L1 keeps little of them, and a second read is a second fetch from the L2: 0.8 us per
//   call at M = 2, 3 us at M = 8.)
// * the 4 BF16-rounded down outputs of a token go as one 16-byte st.async store to every CTA's copy of g
//   [M, 64], completing on the receiver's mbarrier (dsmem.cuh: no cluster barrier round trip); this thread's
//   h, msg, norm_w values load meanwhile;
// * forget/gain: one thread per channel reads its two weight rows from shared memory (16-byte chunks,
//   the 16-byte row padding puts 8 consecutive threads in distinct banks) against broadcast g, applies
//   gated = (1 - tanh) h + tanh msg in registers, a block sum gives the CTA's per-token sum of squares,
//   sent to every CTA the same way; each CTA then normalises and writes its own channels.
// Every CTA sums the received partials in the same order, so all CTAs use identical g and norms.
// pdl=True: all weight copies/loads are issued before griddepcontrol.wait (pdl_wait_fenced: ptxas would sink
// them below a plain wait), h/msg are read after it.
// Ported from engine/csrc/dense/gates.cu (torch launchers -> raw pointers). Results are bit-identical to that
// kernel: same thread layout, expressions and order of every FP32 sum; changed are the barriers (norms of
// several tokens in one reduction, tokens staged in groups, single-barrier block sums, mbarrier exchanges),
// the single read of the inputs and the prefetch.
#include <cooperative_groups.h>

#include "kernels/common/l2.cuh"
#include "kernels/dense/cluster.cuh"
#include "kernels/dense/dense.cuh"
#include "kernels/dense/dense.h"
#include "kernels/dense/dsmem.cuh"
#include "kernels/dense/reduce.cuh"

#include <type_traits>

namespace cg = cooperative_groups;

namespace ingine::kernels::dense {
namespace {

constexpr int RANK = 64;
// L2 class of the weight loads (l2.cuh): none. Evict::last does keep the gate weights (56 MB over the loop
// layers) and the router matrices in the L2 through a token's weight streams, but the decode step was ~2%
// slower with it: their fetch already overlaps the previous kernel, and the L2 (~96 MB) then has ~25 MB
// left for everything else.
constexpr l2::Evict WEIGHT_L2 = l2::Evict::none;

struct GateArgs {
  const float* h;    // [G, M, H], or [M, H] read by every gate (h_shared)
  bool h_shared;
  const float* msg;  // [G, M, H]
  const __nv_bfloat16* w_down;  // [G, 64, 2H]
  const void* w_forget;         // [G, H, 64] fp32 or bf16
  const void* w_gain;
  const float* norm_w;  // [G, H]
  __nv_bfloat16* out_bf16;  // [G, M, H]
  float* out_f32;           // [G, M, H] or null
  uint32_t zero;            // 0 (opaque to the compiler: pdl_wait_fenced)
};

template <int CL, bool FG32>
struct GateCfg {
  static constexpr int RPC = RANK / CL;             // down rows per CTA (4 for CL=16, 8 for CL=8)
  static constexpr int RGROUPS = RPC / 4;           // groups of 4 down rows per CTA (1 for CL=16, 2 for CL=8)
  static constexpr int NW = 8;                      // warps: 4 (row_in_group, half) pairs
  static constexpr int NT = 32 * NW;                // 256 threads (divides H = 2304: XPT = 9)
  static constexpr int CPC = H / CL;                // channels per CTA (144 for CL=16, 288 for CL=8)
  static constexpr int CPASS = (CPC + NT - 1) / NT; // channel passes per thread (1 for CL=16, 2 for CL=8)
  static constexpr int FGB = FG32 ? 4 : 2;          // bytes per forget/gain weight
  static constexpr int ROW = RANK * FGB;            // bytes of one forget/gain row
  static constexpr int RS = ROW + 16;               // padded row stride in shared memory (no bank conflicts)
  static constexpr int SLICE = CPC * RS;            // bytes of one padded matrix slice
  static constexpr bool STAGE_FG = !(CL == 8 && FG32);
  static_assert(ROW % 16 == 0 && (RPC == 4 || RPC == 8), "uneven split");
  static constexpr int SMEM = STAGE_FG ? 2 * SLICE : 0;
  static constexpr int XMAX = (CL == 16 && !FG32) ? 4 : 1;  // tokens staged per barrier (one staging buffer each)
};

template <int CL, bool FG32, int M>
__global__ void __launch_bounds__(GateCfg<CL, FG32>::NT) gates_kernel(GateArgs a) {
  using C = GateCfg<CL, FG32>;
  constexpr int DCH = H / (8 * 32);  // 9 chunks of 8 down weights per lane
  constexpr int XPT = H / C::NT;     // inputs per thread per half when staging x
  constexpr int MQ = (M + 3) / 4;    // 16-byte stores holding M values
  constexpr int XG = M < C::XMAX ? M : C::XMAX;  // tokens per staging group
  static_assert(H % C::NT == 0, "x staging split");
  extern __shared__ __align__(16) uint8_t fg[];  // forget slice, then gain slice (padded rows)
  __shared__ __align__(16) float gall[M][RANK];  // full down output (BF16 values), received
  // bf16([n(h), n(msg)]) of the tokens of a group (M = 1 with FP16 slices: two, the shared-memory size the
  // plain step was tuned with)
  __shared__ __align__(16) __nv_bfloat16 xs[M == 1 && C::XMAX > 1 ? 2 : XG][2 * H];
  __shared__ __align__(16) float part[M][2][C::RPC];  // per-half partial dots
  __shared__ __align__(16) float ssall[CL][4 * MQ];   // per-CTA sums of squares of the gated output, received
  __shared__ float red[C::NW * 32], red2[C::NW * 32];
  __shared__ __align__(8) uint64_t mbar[2];  // g, sums of squares

  cg::cluster_group cluster = cg::this_cluster();
  const int rank = cluster.block_rank();
  const int g = blockIdx.x / CL;
  const int t = threadIdx.x, warp = t >> 5, lane = t & 31;
  const int rl = warp >> 1, half = warp & 1;  // down projection: warp = (row_local, half)
  const float* hg = a.h + (a.h_shared ? size_t(0) : size_t(g) * M * H);
  const float* mg = a.msg + size_t(g) * M * H;
  const uint32_t mb_g = smem_u32(&mbar[0]), mb_s = smem_u32(&mbar[1]);
  if (t == 0) {
    mb_init(mb_g);
    mb_init(mb_s);
    fence_mbarrier_init();
    mb_expect(mb_g, 16u * CL * C::RGROUPS * M);
    mb_expect(mb_s, 16u * CL * MQ);
  }
  cluster_arrive_relaxed();  // waited for before the first remote store

  // forget/gain slices (padded rows) -> shared memory, asynchronous, waited for before the forget/gain phase
  const std::uint64_t wpol = l2::policy<WEIGHT_L2>();
  const uint8_t* const fsrc = reinterpret_cast<const uint8_t*>(a.w_forget) + (size_t(g) * H + rank * C::CPC) * C::ROW;
  const uint8_t* const gsrc = reinterpret_cast<const uint8_t*>(a.w_gain) + (size_t(g) * H + rank * C::CPC) * C::ROW;
  if constexpr (C::STAGE_FG) {
    constexpr int RCH = C::ROW / 16;
    for (int k = t; k < C::CPC * RCH; k += C::NT) {
      const int dst = (k / RCH) * C::RS + (k % RCH) * 16;
      l2::cp_async16<WEIGHT_L2>(fg + dst, fsrc + size_t(k) * 16, wpol);
      l2::cp_async16<WEIGHT_L2>(fg + C::SLICE + dst, gsrc + size_t(k) * 16, wpol);
    }
    asm volatile("cp.async.commit_group;" ::: "memory");
  }

  // down projection weights -> registers
  uint4 wv[C::RGROUPS][DCH];
  {
#pragma unroll
    for (int rg = 0; rg < C::RGROUPS; ++rg) {
      const __nv_bfloat16* wr = a.w_down + (size_t(g) * RANK + rank * C::RPC + rg * 4 + rl) * 2 * H + half * H;
#pragma unroll
      for (int i = 0; i < DCH; ++i) wv[rg][i] = l2::ld_stream<WEIGHT_L2>(wr + (i * 32 + lane) * 8, wpol);
    }
  }

  // all weights are in flight; h and msg come from earlier kernels
  pdl_wait_fenced(a.zero);
  pdl_trigger();

  // row norms and down projection, XG tokens at a time (see above)
#pragma unroll
  for (int m0 = 0; m0 < M; m0 += XG) {
    if (m0 > 0) __syncthreads();  // the previous group's dots are done with the staging buffers
    // inputs: XPT of h and of msg per thread and token, coalesced; thread partials of the sums of squares
    float xh[XG][XPT], xm[XG][XPT], ss[2 * XG];
#pragma unroll
    for (int j = 0; j < XG; ++j) {
      const int m = m0 + j;
      float sh = 0.f, sm = 0.f;
      if (m < M) {
#pragma unroll
        for (int k = 0; k < XPT; ++k) {
          xh[j][k] = hg[size_t(m) * H + k * C::NT + t];
          xm[j][k] = mg[size_t(m) * H + k * C::NT + t];
          sh = fmaf(xh[j][k], xh[j][k], sh);
          sm = fmaf(xm[j][k], xm[j][k], sm);
        }
      }
      ss[2 * j] = sh;
      ss[2 * j + 1] = sm;
    }
    block_sum_bcast<2 * XG, C::NW>(ss, red2);
#pragma unroll
    for (int j = 0; j < XG; ++j) {
      if (m0 + j < M) {
        const float rh = rsqrtf(ss[2 * j] / H + EPS), rm = rsqrtf(ss[2 * j + 1] / H + EPS);
        __nv_bfloat16* xb = xs[j];
#pragma unroll
        for (int k = 0; k < XPT; ++k) {
          xb[k * C::NT + t] = __float2bfloat16_rn(xh[j][k] * rh);
          xb[H + k * C::NT + t] = __float2bfloat16_rn(xm[j][k] * rm);
        }
      }
    }
    __syncthreads();
#pragma unroll
    for (int j = 0; j < XG; ++j) {
      const int m = m0 + j;
      if (m < M) {
        const __nv_bfloat16* xb = xs[j];
#pragma unroll
        for (int rg = 0; rg < C::RGROUPS; ++rg) {
          float acc = 0.f;
#pragma unroll
          for (int i = 0; i < DCH; ++i) {
            float wf[8], xf[8];
            unpack_bf16x8(wv[rg][i], wf);
            unpack_bf16x8(*reinterpret_cast<const uint4*>(&xb[half * H + (i * 32 + lane) * 8]), xf);
#pragma unroll
            for (int e = 0; e < 8; ++e) acc = fmaf(wf[e], xf[e], acc);
          }
          acc = group_sum<32>(acc);
          if (lane == 0) part[m][half][rg * 4 + rl] = acc;
        }
      }
    }
  }
  __syncthreads();

  cluster_wait();  // all CTAs resident, barriers initialised: remote stores may start
  if (t < M * CL * C::RGROUPS) {  // thread (group rg, token m, destination d): 4 rows as one 16-byte store
    const int rg = t / (M * CL), rem = t % (M * CL);
    const int m = rem % M, d = rem / M;
    const float4 p0 = *reinterpret_cast<const float4*>(&part[m][0][4 * rg]);
    const float4 p1 = *reinterpret_cast<const float4*>(&part[m][1][4 * rg]);
    const float4 v = make_float4(round_bf16(p0.x + p1.x), round_bf16(p0.y + p1.y), round_bf16(p0.z + p1.z),
                                 round_bf16(p0.w + p1.w));
    st_async4(remote(smem_u32(&gall[m][rank * C::RPC + 4 * rg]), d), v, remote(mb_g, d));
  }
  // this thread's channel inputs fly during the exchange
  float hv[C::CPASS][M], mv[C::CPASS][M], nw[C::CPASS];
#pragma unroll
  for (int p = 0; p < C::CPASS; ++p) {
    nw[p] = 0.f;
#pragma unroll
    for (int m = 0; m < M; ++m) hv[p][m] = mv[p][m] = 0.f;
    const int tl = t + p * C::NT;
    if (tl < C::CPC) {
      const int c = rank * C::CPC + tl;
#pragma unroll
      for (int m = 0; m < M; ++m) {
        hv[p][m] = hg[size_t(m) * H + c];
        mv[p][m] = mg[size_t(m) * H + c];
      }
      nw[p] = a.norm_w[size_t(g) * H + c];
    }
  }
  if constexpr (C::STAGE_FG) {
    asm volatile("cp.async.wait_all;" ::: "memory");
    __syncthreads();  // forget/gain slices from all threads' copies are visible
  }
  mb_wait(mb_g);

  // forget / gain: one thread per channel; its two weight rows are read as 16-byte chunks from the padded
  // slices (8 consecutive threads hit 8 distinct bank groups), g is a broadcast read
  float sq[M];
#pragma unroll
  for (int m = 0; m < M; ++m) sq[m] = 0.f;
#pragma unroll
  for (int p = 0; p < C::CPASS; ++p) {
    const int tl = t + p * C::NT;
    if (tl < C::CPC) {
      const int c = rank * C::CPC + tl;
      constexpr int EPC = 16 / C::FGB;  // weights per chunk
      const uint8_t* fr = C::STAGE_FG ? (fg + tl * C::RS) : (fsrc + tl * C::ROW);
      const uint8_t* gr = C::STAGE_FG ? (fg + C::SLICE + tl * C::RS) : (gsrc + tl * C::ROW);
      float df[M], dg[M];
#pragma unroll
      for (int m = 0; m < M; ++m) df[m] = dg[m] = 0.f;
#pragma unroll 4
      for (int k = 0; k < RANK / EPC; ++k) {
        const uint4 uf = *reinterpret_cast<const uint4*>(fr + k * 16);
        const uint4 ug = *reinterpret_cast<const uint4*>(gr + k * 16);
        float wf[EPC], wg[EPC];
        if constexpr (FG32) {
          wf[0] = __uint_as_float(uf.x); wf[1] = __uint_as_float(uf.y); wf[2] = __uint_as_float(uf.z); wf[3] = __uint_as_float(uf.w);
          wg[0] = __uint_as_float(ug.x); wg[1] = __uint_as_float(ug.y); wg[2] = __uint_as_float(ug.z); wg[3] = __uint_as_float(ug.w);
        } else {
          unpack_bf16x8(uf, wf);
          unpack_bf16x8(ug, wg);
        }
#pragma unroll
        for (int m = 0; m < M; ++m) {
#pragma unroll
          for (int e = 0; e < EPC; e += 4) {
            const float4 gv = *reinterpret_cast<const float4*>(&gall[m][k * EPC + e]);
            df[m] = fmaf(wf[e], gv.x, fmaf(wf[e + 1], gv.y, fmaf(wf[e + 2], gv.z, fmaf(wf[e + 3], gv.w, df[m]))));
            dg[m] = fmaf(wg[e], gv.x, fmaf(wg[e + 1], gv.y, fmaf(wg[e + 2], gv.z, fmaf(wg[e + 3], gv.w, dg[m]))));
          }
        }
      }
#pragma unroll
      for (int m = 0; m < M; ++m) {
        hv[p][m] = (1.f - tanhf(df[m])) * hv[p][m] + tanhf(dg[m]) * mv[p][m];  // gated
        sq[m] += hv[p][m] * hv[p][m];
        if (a.out_f32) a.out_f32[(size_t(g) * M + m) * H + c] = hv[p][m];
      }
    }
  }

  // per-token sums of squares of this CTA's channels, sent to every CTA
  block_sum_bcast<M, C::NW>(sq, red);
  if (t < CL) {  // to CTA t: the M sums as 16-byte stores
    float r[4 * MQ] = {};
#pragma unroll
    for (int m = 0; m < M; ++m) r[m] = sq[m];
#pragma unroll
    for (int q = 0; q < MQ; ++q)
      st_async4(remote(smem_u32(&ssall[rank][4 * q]), t), make_float4(r[4 * q], r[4 * q + 1], r[4 * q + 2], r[4 * q + 3]),
                remote(mb_s, t));
  }
  mb_wait(mb_s);

#pragma unroll
  for (int p = 0; p < C::CPASS; ++p) {
    const int tl = t + p * C::NT;
    if (tl < C::CPC) {
      const int c = rank * C::CPC + tl;
#pragma unroll
      for (int m = 0; m < M; ++m) {
        float ss = 0.f;
#pragma unroll
        for (int d = 0; d < CL; ++d) ss += ssall[d][m];
        a.out_bf16[(size_t(g) * M + m) * H + c] = __float2bfloat16_rn(hv[p][m] * rsqrtf(ss / H + EPS) * nw[p]);
      }
    }
  }
}

template <int CL, bool FG32>
void launch_gates(int G, int M, const GateArgs& a, bool pdl, cudaStream_t stream) {
  using C = GateCfg<CL, FG32>;
  auto go = [&](auto mc) {
    constexpr int MM = decltype(mc)::value;
    launch_ex(gates_kernel<CL, FG32, MM>, dim3(G * CL), C::NT, C::SMEM, CL, pdl, stream, a);
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
    default: require(false, "message_gates: M must be in 1..8");
  }
}

// 16-CTA clusters need the non-portable opt-in and a GPC with 16 free SMs; fall back to 8 otherwise.
int gates_cluster_size() {
  static const int cl = [] {
    using C = GateCfg<16, false>;
    auto k = gates_kernel<16, false, 8>;
    if (cudaFuncSetAttribute(k, cudaFuncAttributeNonPortableClusterSizeAllowed, 1) != cudaSuccess ||
        cudaFuncSetAttribute(k, cudaFuncAttributeMaxDynamicSharedMemorySize, C::SMEM) != cudaSuccess) {
      (void)cudaGetLastError();
      return 8;
    }
    cudaLaunchConfig_t cfg = {};
    cfg.gridDim = dim3(16);
    cfg.blockDim = dim3(C::NT);
    cfg.dynamicSmemBytes = C::SMEM;
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

void message_gates(const float* h, bool h_shared, const float* msg, int G, int M, const __nv_bfloat16* w_down,
                   const void* w_forget, const void* w_gain, bool fg_fp32, const float* norm_w,
                   __nv_bfloat16* out_bf16, float* out_f32, bool pdl, cudaStream_t stream) {
  require(G >= 1 && G <= 3 && M >= 1 && M <= 8, "message_gates: G in 1..3, M in 1..8");
  require(h && msg && w_down && w_forget && w_gain && norm_w && out_bf16, "message_gates: null argument");
  GateArgs a;
  a.h = h;
  a.h_shared = h_shared;
  a.msg = msg;
  a.w_down = w_down;
  a.w_forget = w_forget;
  a.w_gain = w_gain;
  a.norm_w = norm_w;
  a.out_bf16 = out_bf16;
  a.out_f32 = out_f32;
  a.zero = 0;
  const bool c16 = gates_cluster_size() == 16;
  if (fg_fp32) {
    if (c16) launch_gates<16, true>(G, M, a, pdl, stream);
    else launch_gates<8, true>(G, M, a, pdl, stream);
  } else {
    if (c16) launch_gates<16, false>(G, M, a, pdl, stream);
    else launch_gates<8, false>(G, M, a, pdl, stream);
  }
}

}  // namespace ingine::kernels::dense
