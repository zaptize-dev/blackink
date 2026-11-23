// FP4 block-scaled Tensor Core (sm_120a) grouped MoE GEMMs for prefill (adapted from Basalt's moe_fp4.cu).
// Uses two FP4 activation terms (W4A4, ~8-bit activation accuracy) with weights read directly from BlackInk's
// NVFP4 experts (w13/w2 codes + E4M3 scales + per-expert float factors).
#include "kernels/prefill/common.cuh"
#include "kernels/prefill/prefill.h"

#include <cuda_fp8.h>
#include <stdexcept>

namespace ingine::kernels::prefill {
namespace {

constexpr int kTileTokens = 64;
constexpr int kBlockTiles = 2;
constexpr int kRowsPerBlock = 128;
constexpr int kKTile = 128;
constexpr int kThreads = 256;
constexpr int kStages = 3;
constexpr int kTwo = 2;

constexpr int kACodeBytes = kTileTokens * kKTile / 2;    // 4096
constexpr int kAScaleBytes = kTileTokens * kKTile / 16;  // 512
constexpr int kAStage = kACodeBytes + kAScaleBytes;      // 4608
constexpr int kBRowStride = 80;                          // 64B codes + 8B scales + 8B pad (16B aligned)
constexpr int kBStage = kRowsPerBlock * kBRowStride;     // 10240
constexpr int kSmemBytes = kStages * (kTwo * kBlockTiles * kAStage + kBStage);  // 86016

__device__ __forceinline__ float softcap(float x, float c) { return c * tanhf(x / c); }

__device__ __forceinline__ void mma_fp4(float* c, const unsigned* a, unsigned b0, unsigned b1, unsigned sfa,
                                        unsigned sfb) {
  constexpr unsigned short kId = 0;
  asm volatile(
      "mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3 "
      "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3}, {%10}, {%11,%12}, {%13}, {%14,%15};\n"
      : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
      : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1), "r"(sfa), "h"(kId), "h"(kId), "r"(sfb),
        "h"(kId), "h"(kId));
}

__device__ __forceinline__ void cp_async8(void* dst, const void* src) {
  asm volatile("cp.async.ca.shared.global [%0], [%1], 8;\n" ::"r"(smem_addr(dst)), "l"(src));
}

__device__ __forceinline__ int swz_fp4(int rr, int lb) { return (((lb >> 4) ^ ((rr >> 1) & 3)) << 4) + (lb & 15); }

__device__ __forceinline__ float e2m1_mag(unsigned c) {
  const unsigned m = c & 7u;
  if (m < 2u) return 0.5f * float(m);
  return ((m & 1u) ? 1.5f : 1.0f) * float(1u << ((m >> 1) - 1u));
}

__device__ __forceinline__ void e2m1_block(const float* x, float inv, uint32_t* w) {
  w[0] = w[1] = 0;
#pragma unroll
  for (int j = 0; j < 8; ++j) {
    unsigned short c;
    asm("{ .reg .b8 t; cvt.rn.satfinite.e2m1x2.f32 t, %2, %1; cvt.u16.u8 %0, t; }"
        : "=h"(c)
        : "f"(x[j] * inv), "f"(x[j + 8] * inv));
    w[j >> 2] |= uint32_t(c & 0xff) << (8 * (j & 3));
  }
}

__device__ __forceinline__ float e2m1_residual(float* x, const uint32_t* w, float sd_g) {
  float m = 0.0f;
#pragma unroll
  for (int j = 0; j < 16; ++j) {
    const unsigned c = (w[(j & 7) >> 2] >> (8 * (j & 3) + (j >= 8 ? 4 : 0))) & 15u;
    const float d = e2m1_mag(c) * sd_g;
    x[j] -= (c & 8u) ? -d : d;
    m = fmaxf(m, fabsf(x[j]));
  }
  return m;
}

__global__ void plan_from_counts_kernel(const int* __restrict__ counts, int* __restrict__ bounds,
                                        int* __restrict__ tile_off, int* __restrict__ blk_off) {
  if (threadIdx.x != 0) return;
  int b = 0, t = 0, k = 0;
  for (int e = 0; e < kExperts; ++e) {
    bounds[e] = b;
    tile_off[e] = t;
    blk_off[e] = k;
    const int c = counts[e];
    b += c;
    t += (c + kTileTokens - 1) / kTileTokens;
    k += (c + kBlockTiles * kTileTokens - 1) / (kBlockTiles * kTileTokens);
  }
  bounds[kExperts] = b;
  tile_off[kExperts] = t;
  blk_off[kExperts] = k;
}

// Quantizes BF16 rows into two-term FP4 planes (TOKENS=true: flat token planes, TOKENS=false: swizzled expert tiles).
template <int NBL, bool TOKENS>
__global__ void quantize_bf16_two_kernel(const __nv_bfloat16* __restrict__ x, const int* __restrict__ bounds,
                                         const int* __restrict__ tile_off, uint8_t* __restrict__ a4,
                                         uint8_t* __restrict__ a4s, float* __restrict__ gs, int cols, int64_t rows) {
  const int lane = threadIdx.x & 31;
  const int64_t r = int64_t(blockIdx.x) * (blockDim.x >> 5) + (threadIdx.x >> 5);
  if (r >= rows) return;

  int64_t pos = r, tile = 0;
  int rr = 0;
  if constexpr (!TOKENS) {
    int lo = 0, hi = kExperts;
    while (hi - lo > 1) {
      const int mid = (lo + hi) >> 1;
      if (bounds[mid] <= r) lo = mid;
      else hi = mid;
    }
    pos = int64_t(tile_off[lo]) * kTileTokens + (r - bounds[lo]);
    tile = pos / kTileTokens;
    rr = int(pos % kTileTokens);
  }

  const int nkt = (cols / kKTile) * 2;
  const int nb = cols / 16;
  const uint16_t* xr = reinterpret_cast<const uint16_t*>(x + r * cols);

  float xv[NBL][16];
  float bm[NBL];
  float amax = 0.0f;
#pragma unroll
  for (int i = 0; i < NBL; ++i) {
    const int b = lane + 32 * i;
    float m = 0.0f;
    if (b < nb) {
      const uint4* p = reinterpret_cast<const uint4*>(xr + 16 * b);
      const uint4 u0 = p[0], u1 = p[1];
      const uint32_t words[8] = {u0.x, u0.y, u0.z, u0.w, u1.x, u1.y, u1.z, u1.w};
#pragma unroll
      for (int q = 0; q < 8; ++q) {
        const float2 fv = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&words[q]));
        xv[i][2 * q] = fv.x;
        xv[i][2 * q + 1] = fv.y;
        m = fmaxf(m, fmaxf(fabsf(fv.x), fabsf(fv.y)));
      }
    }
    bm[i] = m;
    amax = fmaxf(amax, m);
  }
#pragma unroll
  for (int o = 16; o > 0; o >>= 1) amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, o));
  const float g = amax * (1.0f / (6.0f * 448.0f));
  if (lane == 0) gs[pos] = g;
  const float inv_g = amax > 0.0f ? 1.0f / g : 0.0f;
  uint8_t* tbase = TOKENS ? nullptr : (a4 + tile * (int64_t(nkt) * kAStage));

#pragma unroll
  for (int i = 0; i < NBL; ++i) {
    const int b = lane + 32 * i;
    if (b >= nb) continue;
    const __nv_fp8_storage_t sb = __nv_cvt_float_to_fp8(bm[i] * inv_g * (1.0f / 6.0f), __NV_SATFINITE, __NV_E4M3);
    const float sd = __half2float(__half(__nv_cvt_fp8_to_halfraw(sb, __NV_E4M3)));
    const float inv = sd > 0.0f ? 1.0f / (sd * g) : 0.0f;
    uint32_t w[2] = {0, 0};
    e2m1_block(xv[i], inv, w);
    const float m2 = e2m1_residual(xv[i], w, sd * g);
    const __nv_fp8_storage_t sb2 = __nv_cvt_float_to_fp8(m2 * inv_g * (1.0f / 6.0f), __NV_SATFINITE, __NV_E4M3);
    const float sd2 = __half2float(__half(__nv_cvt_fp8_to_halfraw(sb2, __NV_E4M3)));
    uint32_t w2[2] = {0, 0};
    e2m1_block(xv[i], sd2 > 0.0f ? 1.0f / (sd2 * g) : 0.0f, w2);

    const int kt = b / 8;
    const int sub = b % 8;
    if constexpr (TOKENS) {
      const size_t code_off = size_t(r) * cols + size_t(2 * kt) * 64 + size_t(sub) * 8;
      const size_t sc_off = size_t(r) * (cols / 8) + size_t(2 * kt) * 8 + size_t(sub);
      *reinterpret_cast<uint2*>(a4 + code_off) = make_uint2(w[0], w[1]);
      *reinterpret_cast<uint2*>(a4 + code_off + 64) = make_uint2(w2[0], w2[1]);
      a4s[sc_off] = uint8_t(sb);
      a4s[sc_off + 8] = uint8_t(sb2);
    } else {
      const int lb = sub * 8;
      uint8_t* k1 = tbase + int64_t(2 * kt) * kAStage;
      uint8_t* k2 = k1 + kAStage;
      *reinterpret_cast<uint2*>(k1 + rr * 64 + swz_fp4(rr, lb)) = make_uint2(w[0], w[1]);
      k1[kACodeBytes + rr * 8 + sub] = uint8_t(sb);
      *reinterpret_cast<uint2*>(k2 + rr * 64 + swz_fp4(rr, lb)) = make_uint2(w2[0], w2[1]);
      k2[kACodeBytes + rr * 8 + sub] = uint8_t(sb2);
    }
  }
}

template <bool GATHER>
__device__ __forceinline__ void load_stage(int stage, int kt, uint8_t* smem, const uint8_t* a_src, size_t tile_stride,
                                           int ntiles, const uint8_t* const* a_code, const int* a_dst,
                                           const uint8_t* a_scale, int s_dst, const uint8_t* w_code,
                                           const uint8_t* w_scale, size_t row_code_bytes, size_t row_scale_bytes,
                                           int rb) {
  uint8_t* as0 = smem + stage * kTwo * kBlockTiles * kAStage;
  uint8_t* bs = smem + kStages * kTwo * kBlockTiles * kAStage + stage * kBStage;
#pragma unroll
  for (int u = 0; u < kTwo; ++u) {
    uint8_t* as = as0 + u * kBlockTiles * kAStage;
    const int ka = kTwo * kt + u;
    if constexpr (GATHER) {
#pragma unroll
      for (int i = 0; i < kBlockTiles * kTileTokens * 4 / kThreads; ++i)
        if (i < ntiles) cp_async16(as + a_dst[i], a_code[i] + size_t(ka) * 64);
      if (int(threadIdx.x) < ntiles * kTileTokens) cp_async8(as + s_dst, a_scale + size_t(ka) * 8);
    } else {
      for (int i = threadIdx.x; i < ntiles * (kAStage / 16); i += kThreads) {
        const int t = i / (kAStage / 16), c = i % (kAStage / 16);
        cp_async16(as + t * kAStage + 16 * c, a_src + t * tile_stride + size_t(ka) * kAStage + 16 * c);
      }
    }
  }
  const int row = int(threadIdx.x) >> 1, part = int(threadIdx.x) & 1;
  const int b_row = rb * kRowsPerBlock + row;
  uint8_t* dst = bs + row * kBRowStride;
  if (part == 0) {
    const uint8_t* src = w_code + size_t(b_row) * row_code_bytes + size_t(kt) * 64;
    cp_async16(dst + 0, src + 0);
    cp_async16(dst + 16, src + 16);
    cp_async16(dst + 32, src + 32);
    cp_async16(dst + 48, src + 48);
  } else {
    const uint8_t* ssrc = w_scale + size_t(b_row) * row_scale_bytes + size_t(kt) * 8;
    cp_async8(dst + 64, ssrc);
  }
}

struct Fp4GemmArgs {
  const uint8_t* w_codes;
  const uint8_t* w_scales;
  const float* w_factors;
  int w_rows;
  int w_cols;
  const uint8_t* a4;
  const uint8_t* a4s;
  const float* gs;
  const int* bounds;
  const int* tile_off;
  const int* blk_off;
  const int* pair_token;
  const int* pair_dst;
  uint16_t* out;
};

template <bool GATHER, bool SWIGLU>
__global__ __launch_bounds__(kThreads, 1) void moe_fp4_gemm_kernel(const Fp4GemmArgs a) {
  extern __shared__ __align__(1024) uint8_t smem[];
  __shared__ int tok_s[kBlockTiles * kTileTokens];
  const int rb = blockIdx.x, y = blockIdx.y;
  if (y >= a.blk_off[kExperts]) return;

  int lo = 0, hi = kExperts;
  while (hi - lo > 1) {
    const int mid = (lo + hi) >> 1;
    if (a.blk_off[mid] <= y) lo = mid;
    else hi = mid;
  }
  const int j = lo, tp = y - a.blk_off[lo];
  const int cnt = a.bounds[j + 1] - a.bounds[j];
  const int first = tp * kBlockTiles * kTileTokens;
  if (first >= cnt) return;
  const int ntiles = first + kTileTokens < cnt ? 2 : 1;

  const int nkt = a.w_cols / kKTile;
  const size_t row_code_bytes = size_t(a.w_cols) / 2;
  const size_t row_scale_bytes = size_t(a.w_cols) / 16;
  const size_t tile_stride = size_t(kTwo) * nkt * kAStage;
  const uint8_t* w_code = a.w_codes + size_t(j) * a.w_rows * row_code_bytes;
  const uint8_t* w_scale = a.w_scales + size_t(j) * a.w_rows * row_scale_bytes;
  const int64_t tile0 = GATHER ? 0 : int64_t(a.tile_off[j]) + kBlockTiles * tp;
  const uint8_t* a_src = GATHER ? nullptr : (a.a4 + tile0 * int64_t(tile_stride));
  const int64_t brow0 = a.bounds[j] + first;

  const uint8_t* a_code[kBlockTiles * kTileTokens * 4 / kThreads] = {};
  int a_dst[kBlockTiles * kTileTokens * 4 / kThreads] = {};
  const uint8_t* a_scale = nullptr;
  int s_dst = 0;
  if constexpr (GATHER) {
    if (threadIdx.x < kBlockTiles * kTileTokens)
      tok_s[threadIdx.x] = a.pair_token[brow0 + (first + int(threadIdx.x) < cnt ? int(threadIdx.x) : 0)];
    __syncthreads();
#pragma unroll
    for (int i = 0; i < kBlockTiles * kTileTokens * 4 / kThreads; ++i) {
      const int c = int(threadIdx.x) + i * kThreads, r = c >> 2, q = c & 3, rr = r % kTileTokens;
      a_code[i] = a.a4 + size_t(tok_s[r]) * size_t(a.w_cols) + 16 * q;
      a_dst[i] = (r / kTileTokens) * kAStage + rr * 64 + ((q ^ ((rr >> 1) & 3)) << 4);
    }
    if (threadIdx.x < kBlockTiles * kTileTokens) {
      const int r = threadIdx.x;
      a_scale = a.a4s + size_t(tok_s[r]) * size_t(a.w_cols / 8);
      s_dst = (r / kTileTokens) * kAStage + kACodeBytes + (r % kTileTokens) * 8;
    }
  }

  const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
  const int warp_m = warp >> 2, warp_n = warp & 3;
  const bool live = warp_m < ntiles;
  const int group = lane >> 2, tig = lane & 3;
  const int a_row = (lane & 7) + (((lane >> 3) & 1) << 3);
  const int a_col = (lane >> 4) * 16;
  const int sfa_row = ((lane & 1) << 3) | (lane >> 2);

  float acc[4][4][4] = {};
#pragma unroll
  for (int s = 0; s < kStages - 1; ++s) {
    if (s < nkt)
      load_stage<GATHER>(s, s, smem, a_src, tile_stride, ntiles, a_code, a_dst, a_scale, s_dst, w_code, w_scale,
                         row_code_bytes, row_scale_bytes, rb);
    cp_commit();
  }
  for (int kt = 0; kt < nkt; ++kt) {
    cp_wait<kStages - 2>();
    __syncthreads();
    if (kt + kStages - 1 < nkt)
      load_stage<GATHER>((kt + kStages - 1) % kStages, kt + kStages - 1, smem, a_src, tile_stride, ntiles, a_code,
                         a_dst, a_scale, s_dst, w_code, w_scale, row_code_bytes, row_scale_bytes, rb);
    cp_commit();
    if (!live) continue;
    const int stage = kt % kStages;
    const uint8_t* as0 = smem + stage * kTwo * kBlockTiles * kAStage + warp_m * kAStage;
    const uint8_t* bs = smem + kStages * kTwo * kBlockTiles * kAStage + stage * kBStage;
#pragma unroll
    for (int s = 0; s < 2; ++s) {
      unsigned b0[4], b1[4], sb[4];
#pragma unroll
      for (int r = 0; r < 4; ++r) {
        const int sr = warp_n * 32 + r * 8 + group;
        const uint8_t* rowp = bs + sr * kBRowStride;
        const uint8_t* pcode = rowp + s * 32;
        b0[r] = *reinterpret_cast<const uint32_t*>(pcode + 4 * tig);
        b1[r] = *reinterpret_cast<const uint32_t*>(pcode + 16 + 4 * tig);
        const unsigned v = *reinterpret_cast<const uint32_t*>(rowp + 64 + s * 4) & 0x7f7f7f7fu;
        sb[r] = v & ~__vcmpeq4(v, 0x7f7f7f7fu);
      }
#pragma unroll
      for (int u = 0; u < kTwo; ++u) {
        const uint8_t* as = as0 + u * kBlockTiles * kAStage;
        unsigned af[4][4], asc[4];
#pragma unroll
        for (int mt = 0; mt < 4; ++mt) {
          const int row = mt * 16 + a_row;
          ldmatrix_x4(af[mt][0], af[mt][1], af[mt][2], af[mt][3],
                      smem_addr(as + row * 64 + swz_fp4(row, s * 32 + a_col)));
          asc[mt] = *reinterpret_cast<const uint32_t*>(as + kACodeBytes + (mt * 16 + sfa_row) * 8 + 4 * s);
        }
#pragma unroll
        for (int mt = 0; mt < 4; ++mt)
#pragma unroll
          for (int r = 0; r < 4; ++r) mma_fp4(acc[mt][r], af[mt], b0[r], b1[r], asc[mt], sb[r]);
      }
    }
  }
  cp_wait<0>();
  if (!live) return;

  const int tl = kBlockTiles * tp + warp_m;
  const int64_t tile = tile0 + warp_m;
  const int64_t row0 = a.bounds[j] + int64_t(tl) * kTileTokens;
  const float factor = a.w_factors[j];

#pragma unroll
  for (int mt = 0; mt < 4; ++mt) {
#pragma unroll
    for (int h = 0; h < 2; ++h) {
      const int t = mt * 16 + group + 8 * h;
      if (tl * kTileTokens + t >= cnt) continue;
      const float g = (GATHER ? a.gs[tok_s[warp_m * kTileTokens + t]] : a.gs[tile * kTileTokens + t]) * factor;
      if constexpr (SWIGLU) {
        uint16_t* out = a.out + (row0 + t) * kInter + rb * 64 + warp_n * 16 + tig;
#pragma unroll
        for (int r = 0; r < 4; ++r) {
          const float gt = round_bf16(acc[mt][r][2 * h] * g);
          const float u = round_bf16(acc[mt][r][2 * h + 1] * g);
          const float sig = 1.f / (1.f + expf(-gt));
          const float v = __fmul_rn(__fmul_rn(softcap(gt, 4.f), sig), softcap(u, 25.f));
          out[4 * r] = f2bf_bits(v);
        }
      } else {
        const int dst = a.pair_dst[row0 + t];
        uint16_t* out = a.out + size_t(dst) * kHidden + rb * 128 + warp_n * 32 + 2 * tig;
#pragma unroll
        for (int r = 0; r < 4; ++r) {
          *reinterpret_cast<uint32_t*>(out + 8 * r) =
              pack_bf16x2(acc[mt][r][2 * h] * g, acc[mt][r][2 * h + 1] * g);
        }
      }
    }
  }
}

template <bool GATHER, bool SWIGLU>
void launch_gemm(const Fp4GemmArgs& a, int max_T, cudaStream_t stream) {
  static const cudaError_t attr =
      cudaFuncSetAttribute(moe_fp4_gemm_kernel<GATHER, SWIGLU>, cudaFuncAttributeMaxDynamicSharedMemorySize, kSmemBytes);
  if (attr != cudaSuccess) throw std::runtime_error("prefill moe_fp4: shared memory attribute failed");
  const int grid_x = a.w_rows / kRowsPerBlock;
  const int grid_y = (max_T * kTopK + kBlockTiles * kTileTokens - 1) / (kBlockTiles * kTileTokens) + kExperts;
  moe_fp4_gemm_kernel<GATHER, SWIGLU><<<dim3(grid_x, grid_y), kThreads, kSmemBytes, stream>>>(a);
  CUDA_CHECK(cudaGetLastError());
}

struct WsLayout {
  size_t bounds_off;
  size_t tile_off;
  size_t blk_off;
  size_t gs_off;
  size_t a4s_off;
  size_t a4_off;
  size_t total_bytes;
};

WsLayout layout_ws(int max_T) {
  auto align256 = [](size_t n) { return (n + 255) / 256 * 256; };
  const size_t pairs = size_t(max_T) * kTopK;
  const size_t tiles_max = (pairs + kTileTokens - 1) / kTileTokens + kExperts;
  WsLayout l{};
  size_t off = 0;
  l.bounds_off = off;
  off += align256((kExperts + 1) * sizeof(int));
  l.tile_off = off;
  off += align256((kExperts + 1) * sizeof(int));
  l.blk_off = off;
  off += align256((kExperts + 1) * sizeof(int));
  l.gs_off = off;
  off += align256(std::max<size_t>(max_T, tiles_max * kTileTokens) * sizeof(float));
  l.a4s_off = off;
  off += align256(size_t(max_T) * (kHidden / 8));
  l.a4_off = off;
  const size_t up_bytes = size_t(max_T) * kHidden;
  const size_t down_bytes = tiles_max * size_t(kTwo * (kInter / kKTile)) * kAStage;
  off += align256(std::max(up_bytes, down_bytes));
  l.total_bytes = off;
  return l;
}

}  // namespace

std::size_t moe_fp4_workspace_bytes(int max_T) { return layout_ws(max_T).total_bytes; }

void moe_fp4_gate_up(const MoeExperts& e, const __nv_bfloat16* x, int T, const MoeWork& w, void* fp4_ws,
                     __nv_bfloat16* act, cudaStream_t stream) {
  const WsLayout l = layout_ws(T);
  auto* base = static_cast<uint8_t*>(fp4_ws);
  auto* bounds = reinterpret_cast<int*>(base + l.bounds_off);
  auto* tile_off = reinterpret_cast<int*>(base + l.tile_off);
  auto* blk_off = reinterpret_cast<int*>(base + l.blk_off);
  auto* gs = reinterpret_cast<float*>(base + l.gs_off);
  auto* a4s = base + l.a4s_off;
  auto* a4 = base + l.a4_off;

  plan_from_counts_kernel<<<1, 32, 0, stream>>>(w.counts, bounds, tile_off, blk_off);
  CUDA_CHECK(cudaGetLastError());

  // kHidden = 2304 -> nb = 144 -> NBL = 5 (32 * 5 = 160 >= 144)
  quantize_bf16_two_kernel<5, true><<<(T + 7) / 8, 256, 0, stream>>>(x, nullptr, nullptr, a4, a4s, gs, kHidden, T);
  CUDA_CHECK(cudaGetLastError());

  Fp4GemmArgs args{};
  args.w_codes = static_cast<const uint8_t*>(e.w13);
  args.w_scales = static_cast<const uint8_t*>(e.w13_scale);
  args.w_factors = e.w13_factor;
  args.w_rows = 2 * kInter;
  args.w_cols = kHidden;
  args.a4 = a4;
  args.a4s = a4s;
  args.gs = gs;
  args.bounds = bounds;
  args.tile_off = tile_off;
  args.blk_off = blk_off;
  args.pair_token = w.pair_token;
  args.pair_dst = w.pair_dst;
  args.out = reinterpret_cast<uint16_t*>(act);
  launch_gemm<true, true>(args, T, stream);
}

void moe_fp4_down(const MoeExperts& e, const __nv_bfloat16* act, int T, const MoeWork& w, void* fp4_ws,
                  __nv_bfloat16* down, cudaStream_t stream) {
  const WsLayout l = layout_ws(T);
  auto* base = static_cast<uint8_t*>(fp4_ws);
  auto* bounds = reinterpret_cast<int*>(base + l.bounds_off);
  auto* tile_off = reinterpret_cast<int*>(base + l.tile_off);
  auto* blk_off = reinterpret_cast<int*>(base + l.blk_off);
  auto* gs = reinterpret_cast<float*>(base + l.gs_off);
  auto* a4 = base + l.a4_off;

  const int pairs = T * kTopK;
  // kInter = 1152 -> nb = 72 -> NBL = 3 (32 * 3 = 96 >= 72)
  quantize_bf16_two_kernel<3, false>
      <<<(pairs + 7) / 8, 256, 0, stream>>>(act, bounds, tile_off, a4, nullptr, gs, kInter, pairs);
  CUDA_CHECK(cudaGetLastError());

  Fp4GemmArgs args{};
  args.w_codes = static_cast<const uint8_t*>(e.w2);
  args.w_scales = static_cast<const uint8_t*>(e.w2_scale);
  args.w_factors = e.w2_factor;
  args.w_rows = kHidden;
  args.w_cols = kInter;
  args.a4 = a4;
  args.a4s = nullptr;
  args.gs = gs;
  args.bounds = bounds;
  args.tile_off = tile_off;
  args.blk_off = blk_off;
  args.pair_token = w.pair_token;
  args.pair_dst = w.pair_dst;
  args.out = reinterpret_cast<uint16_t*>(down);
  launch_gemm<false, false>(args, T, stream);
}

}  // namespace ingine::kernels::prefill
