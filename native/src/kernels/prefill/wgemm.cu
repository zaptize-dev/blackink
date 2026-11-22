// Grouped expert GEMMs of the prefill (model._moe_grouped) over NVFP4 / MXFP8 experts decoded on the fly.
//
// Design. A CTA computes up to 128 activation rows x 128 weight rows, K in steps of 64, mma.m16n8k16 BF16 with
// FP32 accumulation. The 8 warps split the weight rows (16 each) and every warp covers all of the tile's
// activation rows, so each weight element is decoded exactly once per CTA, straight from the staged checkpoint
// codes into the MMA B fragments in registers (no BF16 weight tile in shared memory):
//   NVFP4: bf16((e2m1 * e4m3 block scale) * factor)      (model.decode_experts)
//   MXFP8: bf16(e4m3 * 2^(e8m0 - 127))                    (model.decode_experts)
// Activation rows (gathered by pair for the experts) and the weight codes stream in through 16-byte cp.async:
// activations in 3 stages of 64 elements, the codes in 2 slots of 128-byte (64-byte for the NVFP4 down
// projection) row pieces, so DRAM reads whole lines and small chunks, which are bound by the weight bytes, keep
// enough loads in flight. One barrier per step. Code rows are padded in shared memory so the 8 rows of a fragment
// load hit distinct banks. m16 row blocks past the tile's row count skip their MMAs (zero-filled copies).
// The per-element products and the k order of the FP32 sums are those of the BF16 MMA over decoded rows.
// Epilogues: gate/up (w13 rows interleaved, 2i gate and 2i + 1 up, so each thread holds gate and up of the same
// i): act = bf16(softcap(bf16 g, 4) * sigmoid(bf16 g) * softcap(bf16 u, 25)); down: bf16 per pair placed at
// t * 8 + slot (moe_combine sums the slots in order: deterministic).
#include "kernels/prefill/common.cuh"
#include "kernels/prefill/prefill.h"

#include <cmath>
#include <stdexcept>

namespace ingine::kernels::prefill {
namespace {

enum class Mode { Up, Down };
constexpr int FMT_NVFP4 = 0, FMT_MXFP8 = 1;

constexpr int TN = 128, BK = 64;
constexpr int AS = 3;  // activation stages (BK elements each)
constexpr int NT = 256;
constexpr int KCH = BK / 8;  // 16-byte chunks per activation tile row

__constant__ float kE2M1[16] = {0.f, .5f, 1.f, 1.5f, 2.f, 3.f, 4.f, 6.f, -0.f, -.5f, -1.f, -1.5f, -2.f, -3.f, -4.f, -6.f};

template <int FMT, int K, int TM>
struct Cfg {
  static constexpr int CODE_ROW = FMT == FMT_NVFP4 ? K / 2 : K;  // checkpoint bytes per weight row
  static constexpr int SCALE_ROW = FMT == FMT_NVFP4 ? K / 16 : K / 32;
  // weight chunk: 64 code bytes per row (NVFP4 128 elements, FP8 64)
  static constexpr int WK = FMT == FMT_NVFP4 ? 128 : 64;
  static constexpr int R = WK / BK;  // activation steps per weight chunk
  static constexpr int KC = K / WK;  // weight chunks
  // weight slots: the chunk of step s + 1 must have landed (and be published by the barrier) at step s, when its
  // B fragments are decoded under step s's MMAs: R (WS - 1) >= AS. Small-row tiles prefetch further.
  static constexpr int WS = TM <= 32 ? (FMT == FMT_NVFP4 ? 6 : 7) : (FMT == FMT_NVFP4 ? 3 : 4);
  static_assert(R * (WS - 1) >= AS);
  static constexpr int WCODE = FMT == FMT_NVFP4 ? WK / 2 : WK;  // code bytes per row per chunk
  static constexpr int WSTRIDE = WCODE + 16;                    // padded: fragment loads on distinct banks
  static constexpr int W_BYTES = TN * WSTRIDE;
  static constexpr int A_BYTES = TM * BK * 2;
  static constexpr int SC_BYTES = TN * SCALE_ROW;               // block scales of the whole tile
  static constexpr int OFF_W = AS * A_BYTES, OFF_SC = OFF_W + WS * W_BYTES;
  static constexpr int OFF_LUT = (OFF_SC + SC_BYTES + 15) / 16 * 16;
  static constexpr int SMEM = OFF_LUT + 64;
  static_assert(K % WK == 0 && SMEM <= 101376);
};

struct GemmArgs {
  const uint8_t* codes = nullptr;   // NVFP4 / MXFP8 expert codes [128, N, ..]
  const uint8_t* scales = nullptr;  // block scales
  const float* factors = nullptr;   // NVFP4 per-expert factor
  const uint16_t* a = nullptr;      // activations (bf16)
  MoeWork w;                        // tiles and pair lists
  uint16_t* out = nullptr;
};

__device__ __forceinline__ void cp_async4(void* smem_dst, const void* gmem_src) {
  asm volatile("cp.async.ca.shared.global [%0], [%1], 4;\n" ::"r"(smem_addr(smem_dst)), "l"(gmem_src));
}

__device__ __forceinline__ float softcap(float x, float cap) { return tanhf(x * (1.f / cap)) * cap; }

// One activation step of MMAs for the first MT 16-row blocks: per k16 group, the A fragments by ldmatrix, then the
// MMAs, then `after(kk)` (the next step's B decoding, which runs under the issued MMAs).
template <int MT, int TM, class After>
__device__ __forceinline__ void mma_step(float (&acc)[TM / 16][2][4], const uint16_t* A,
                                         const unsigned (&b)[BK / 16][2][2], int lane, After&& after) {
#pragma unroll
  for (int kk = 0; kk < BK / 16; ++kk) {
    unsigned a[MT][4];
#pragma unroll
    for (int mt = 0; mt < MT; ++mt)
      ldmatrix_x4(a[mt][0], a[mt][1], a[mt][2], a[mt][3],
                  smem_addr(A + swz<KCH>(mt * 16 + (lane & 15), kk * 2 + (lane >> 4))));
#pragma unroll
    for (int mt = 0; mt < MT; ++mt) {
      mma_bf16(acc[mt][0], a[mt], b[kk][0][0], b[kk][0][1]);
      mma_bf16(acc[mt][1], a[mt], b[kk][1][0], b[kk][1][1]);
    }
    after(kk);
  }
}

// Up: A = x rows of the pairs' tokens (K = H), W = w13 (N = 2I), out act [pairs, I].
// Down: A = act rows (K = I), W = w2 (N = H), out down [T * 8, H] at pair_dst.
template <int FMT, Mode MODE, int K, int TM>
__global__ void __launch_bounds__(NT, 1) wgemm_kernel(const GemmArgs g) {
  constexpr int KS = K / BK;
  constexpr int NE = MODE == Mode::Up ? 2 * kInter : kHidden;  // expert weight rows
  using C = Cfg<FMT, K, TM>;
  extern __shared__ __align__(128) unsigned char smem[];
  unsigned char* a_smem = smem;
  unsigned char* w_smem = smem + C::OFF_W;
  uint8_t* sc_smem = smem + C::OFF_SC;
  float* lut = reinterpret_cast<float*>(smem + C::OFF_LUT);
  const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;

  // ---- tile: n pair rows (from pb) of expert e x weight rows n0..n0+127
  constexpr int NTILES_N = NE / TN;
  const int tile_idx = blockIdx.x / NTILES_N;
  if (tile_idx >= *g.w.tiles_n) return;
  const int4 tile = g.w.tiles[tile_idx];
  const int e = tile.x, pb = tile.y, n = tile.z;
  const int n0 = (blockIdx.x % NTILES_N) * TN;
  const uint8_t* code_tile = g.codes + (int64_t(e) * NE + n0) * C::CODE_ROW;
  const uint8_t* scale_tile = g.scales + (int64_t(e) * NE + n0) * C::SCALE_ROW;
  if (tid < 16) lut[tid] = kE2M1[tid];

  // activation rows of this thread's cp.async chunks: rows tid / 8 + 32 i, chunk tid % 8
  const uint16_t* arow[TM / 32];
#pragma unroll
  for (int i = 0; i < TM / 32; ++i) {
    const int r = (tid >> 3) + 32 * i;
    const int rr = r < n ? r : 0;
    if constexpr (MODE == Mode::Up) arow[i] = g.a + size_t(g.w.pair_token[pb + rr]) * K;
    else arow[i] = g.a + size_t(pb + rr) * K;
  }

  auto load_a = [&](int ks) {
    uint16_t* A = reinterpret_cast<uint16_t*>(a_smem + (ks % AS) * C::A_BYTES);
#pragma unroll
    for (int i = 0; i < TM / 32; ++i) {
      const int r = (tid >> 3) + 32 * i, c = tid & 7;
      cp_async16(A + swz<KCH>(r, c), arow[i] + ks * BK + c * 8, r < n ? 16 : 0);
    }
  };
  auto load_w = [&](int chunk) {
    unsigned char* ws = w_smem + (chunk % C::WS) * C::W_BYTES;
    constexpr int CPR = C::WCODE / 16;  // 16-byte pieces per row
#pragma unroll
    for (int i = 0; i < TN * CPR / NT; ++i) {
      const int id = tid + i * NT, row = id / CPR, ch = id % CPR;
      cp_async16(ws + row * C::WSTRIDE + ch * 16, code_tile + int64_t(row) * C::CODE_ROW + chunk * C::WCODE + ch * 16);
    }
  };
  // group j: activation step j (plus whatever weight chunk was issued since the previous commit)
  auto issue = [&](int j) {
    if (j < KS) load_a(j);
    cp_commit();
  };

  // the tile's block scales, with the first activation group
  for (int id = tid; id < C::SC_BYTES / 4; id += NT)
    cp_async4(sc_smem + id * 4, scale_tile + (id * 4 / C::SCALE_ROW) * int64_t(C::SCALE_ROW) + (id * 4) % C::SCALE_ROW);
#pragma unroll
  for (int j = 0; j < C::WS - 1; ++j)
    if (j < C::KC) load_w(j);
#pragma unroll
  for (int j = 0; j < AS - 1; ++j) issue(j);

  const int c = lane & 3;
  const float factor = FMT == FMT_NVFP4 ? __ldg(g.factors + e) : 1.f;
  const int mact = min((n + 15) / 16, TM / 16);  // active m16 row blocks (same for every warp)
  float acc[TM / 16][2][4];
#pragma unroll
  for (int a = 0; a < TM / 16; ++a)
#pragma unroll
    for (int b = 0; b < 2; ++b) acc[a][b][0] = acc[a][b][1] = acc[a][b][2] = acc[a][b][3] = 0.f;

  // B fragments of step ks (k16 group kk) decoded from the staged codes of its weight chunk
  auto scales_of = [&](int ks, float (&sc)[2][4]) {
#pragma unroll
    for (int nt = 0; nt < 2; ++nt) {
      const int row = warp * 16 + nt * 8 + (lane >> 2);
      if constexpr (FMT == FMT_NVFP4) {
        const uint32_t s4 = *reinterpret_cast<const uint32_t*>(sc_smem + row * C::SCALE_ROW + ks * 4);
        const float2 lo = e4m3x2_to_float2(uint16_t(s4 & 0xffff)), hi = e4m3x2_to_float2(uint16_t(s4 >> 16));
        sc[nt][0] = lo.x, sc[nt][1] = lo.y, sc[nt][2] = hi.x, sc[nt][3] = hi.y;
      } else {
        const uint8_t* ms = sc_smem + row * C::SCALE_ROW + ks * 2;
        sc[nt][0] = sc[nt][1] = exp2f(float(ms[0]) - 127.f);
        sc[nt][2] = sc[nt][3] = exp2f(float(ms[1]) - 127.f);
      }
    }
  };
  auto decode = [&](int ks, int kk, const float (&sc)[2][4], unsigned (&b)[BK / 16][2][2]) {
    const unsigned char* ws = w_smem + ((ks / C::R) % C::WS) * C::W_BYTES;
    const int kc = (ks % C::R) * BK + kk * 16;  // first element of the k16 group inside the chunk
#pragma unroll
    for (int nt = 0; nt < 2; ++nt) {
      const int row = warp * 16 + nt * 8 + (lane >> 2);
      const float s = sc[nt][kk];
      if constexpr (FMT == FMT_NVFP4) {
        // elements 2c, 2c+1 (byte c) and 8 + 2c, +1 (byte 4 + c) of the k16 group
        const uint2 wd = *reinterpret_cast<const uint2*>(ws + row * C::WSTRIDE + kc / 2);
        const uint32_t b0 = (wd.x >> (8 * c)) & 0xffu, b1 = (wd.y >> (8 * c)) & 0xffu;
        b[kk][nt][0] = pack_bf16x2((lut[b0 & 0xf] * s) * factor, (lut[b0 >> 4] * s) * factor);
        b[kk][nt][1] = pack_bf16x2((lut[b1 & 0xf] * s) * factor, (lut[b1 >> 4] * s) * factor);
      } else {
        const uint4 q = *reinterpret_cast<const uint4*>(ws + row * C::WSTRIDE + kc);
        const uint32_t w0 = (c >> 1) ? q.y : q.x, w1 = (c >> 1) ? q.w : q.z;
        const float2 f0 = e4m3x2_to_float2(uint16_t(w0 >> (16 * (c & 1))));
        const float2 f1 = e4m3x2_to_float2(uint16_t(w1 >> (16 * (c & 1))));
        b[kk][nt][0] = pack_bf16x2(f0.x * s, f0.y * s);
        b[kk][nt][1] = pack_bf16x2(f1.x * s, f1.y * s);
      }
      asm volatile("" : "+r"(b[kk][nt][0]), "+r"(b[kk][nt][1]));
    }
  };

  // Software pipeline: step ks's MMAs run on fragments decoded during step ks - 1; the decoding of step ks + 1
  // is interleaved with them (its chunk has landed: R (WS - 1) >= AS, and the barrier published it).
  unsigned bcur[BK / 16][2][2], bnext[BK / 16][2][2];
#pragma unroll 1
  for (int ks = 0; ks < KS; ++ks) {
    cp_wait<AS - 2>();
    __syncthreads();
    const int chunk = ks / C::R;
    // first step of a chunk: the chunk WS - 1 ahead into the slot the previous chunk used (read before the barrier)
    if (ks % C::R == 0 && chunk + C::WS - 1 < C::KC) load_w(chunk + C::WS - 1);
    issue(ks + AS - 1);
    if (ks == 0) {
      float sc0[2][4];
      scales_of(0, sc0);
#pragma unroll
      for (int kk = 0; kk < BK / 16; ++kk) decode(0, kk, sc0, bcur);
    }
    const uint16_t* A = reinterpret_cast<const uint16_t*>(a_smem + (ks % AS) * C::A_BYTES);
    const bool more = ks + 1 < KS;
    float scn[2][4];
    if (more) scales_of(ks + 1, scn);
    auto after = [&](int kk) {
      if (more) decode(ks + 1, kk, scn, bnext);
    };
    // the row-block count is a template constant inside mma_step: a runtime `mt < mact` test would be if-converted
    // into predicated MMAs, which still occupy the tensor pipe (8x the work for a 16-row tile)
    switch (mact) {
      case 1: if constexpr (1 <= TM / 16) mma_step<1, TM>(acc, A, bcur, lane, after); break;
      case 2: if constexpr (2 <= TM / 16) mma_step<2, TM>(acc, A, bcur, lane, after); break;
      case 3: if constexpr (3 <= TM / 16) mma_step<3, TM>(acc, A, bcur, lane, after); break;
      case 4: if constexpr (4 <= TM / 16) mma_step<4, TM>(acc, A, bcur, lane, after); break;
      case 5: if constexpr (5 <= TM / 16) mma_step<5, TM>(acc, A, bcur, lane, after); break;
      case 6: if constexpr (6 <= TM / 16) mma_step<6, TM>(acc, A, bcur, lane, after); break;
      case 7: if constexpr (7 <= TM / 16) mma_step<7, TM>(acc, A, bcur, lane, after); break;
      default: mma_step<TM / 16, TM>(acc, A, bcur, lane, after); break;
    }
#pragma unroll
    for (int kk = 0; kk < BK / 16; ++kk)
#pragma unroll
      for (int nt = 0; nt < 2; ++nt) bcur[kk][nt][0] = bnext[kk][nt][0], bcur[kk][nt][1] = bnext[kk][nt][1];
  }
  cp_wait<0>();

#pragma unroll
  for (int mt = 0; mt < TM / 16; ++mt) {
    if (mt >= mact) continue;
#pragma unroll
    for (int half = 0; half < 2; ++half) {
      const int row = mt * 16 + (lane >> 2) + half * 8;
      if (row >= n) continue;
#pragma unroll
      for (int nt = 0; nt < 2; ++nt) {
        const int cl = warp * 16 + nt * 8 + c * 2;  // column inside the tile
        const float c0 = acc[mt][nt][2 * half], c1 = acc[mt][nt][2 * half + 1];
        if constexpr (MODE == Mode::Up) {
          const float gt = round_bf16(c0), u = round_bf16(c1);
          const float sig = 1.f / (1.f + expf(-gt));
          const float v = __fmul_rn(__fmul_rn(softcap(gt, 4.f), sig), softcap(u, 25.f));
          g.out[size_t(pb + row) * kInter + (n0 + cl) / 2] = f2bf_bits(v);
        } else {
          const int dst = g.w.pair_dst[pb + row];
          *reinterpret_cast<uint32_t*>(g.out + size_t(dst) * kHidden + n0 + cl) = pack_bf16x2(c0, c1);
        }
      }
    }
  }
}

template <int FMT, Mode MODE, int K, int TM>
void launch(const GemmArgs& g, int grid, cudaStream_t stream) {
  constexpr int SMEM = Cfg<FMT, K, TM>::SMEM;
  static const cudaError_t attr =
      cudaFuncSetAttribute(wgemm_kernel<FMT, MODE, K, TM>, cudaFuncAttributeMaxDynamicSharedMemorySize, SMEM);
  if (attr != cudaSuccess) throw std::runtime_error("prefill wgemm: shared memory attribute failed");
  wgemm_kernel<FMT, MODE, K, TM><<<grid, NT, SMEM, stream>>>(g);
}

template <int FMT, Mode MODE, int K>
void launch_rows(const GemmArgs& g, int tm, int grid_per_row_tile, int row_tiles, cudaStream_t stream) {
  if (tm == 32) launch<FMT, MODE, K, 32>(g, grid_per_row_tile * row_tiles, stream);
  else launch<FMT, MODE, K, 128>(g, grid_per_row_tile * row_tiles, stream);
}

GemmArgs expert_args(const void* codes, const void* scales, const float* factors, const __nv_bfloat16* a,
                     const MoeWork& w, __nv_bfloat16* out) {
  GemmArgs g;
  g.codes = static_cast<const uint8_t*>(codes);
  g.scales = static_cast<const uint8_t*>(scales);
  g.factors = factors;
  g.a = reinterpret_cast<const uint16_t*>(a);
  g.w = w;
  g.out = reinterpret_cast<uint16_t*>(out);
  return g;
}

}  // namespace

void moe_gate_up(const MoeExperts& e, const __nv_bfloat16* x, int T, const MoeWork& w, __nv_bfloat16* act,
                 cudaStream_t stream) {
  const GemmArgs g = expert_args(e.w13, e.w13_scale, e.w13_factor, x, w, act);
  if (e.fmt == 0) launch_rows<FMT_NVFP4, Mode::Up, kHidden>(g, moe_tile_rows(T), 2 * kInter / TN, moe_max_tiles(T), stream);
  else launch_rows<FMT_MXFP8, Mode::Up, kHidden>(g, moe_tile_rows(T), 2 * kInter / TN, moe_max_tiles(T), stream);
}

void moe_down(const MoeExperts& e, const __nv_bfloat16* act, int T, const MoeWork& w, __nv_bfloat16* down,
              cudaStream_t stream) {
  const GemmArgs g = expert_args(e.w2, e.w2_scale, e.w2_factor, act, w, down);
  if (e.fmt == 0) launch_rows<FMT_NVFP4, Mode::Down, kInter>(g, moe_tile_rows(T), kHidden / TN, moe_max_tiles(T), stream);
  else launch_rows<FMT_MXFP8, Mode::Down, kInter>(g, moe_tile_rows(T), kHidden / TN, moe_max_tiles(T), stream);
}

}  // namespace ingine::kernels::prefill
