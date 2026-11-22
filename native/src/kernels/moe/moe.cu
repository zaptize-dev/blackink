// Routed-expert kernels for Inkling-2-Mini decode: moe_up (W13 GEMV + softcapped SwiGLU) and moe_down
// (W2 GEMV + routing-weighted sum + post-MLP short conv). Contract: native/KERNELS.md, area `moe`.
//
// Design
// ------
// * Weights are streamed once per call in their checkpoint encoding and decoded in registers to exactly
//   the BF16 values `model.decode_experts` produces (NVFP4: bf16(fp32(code * e4m3_scale) * factor), with
//   code * scale exact in f16; MXFP8: bf16(fp8 * 2^(s-127)), exact). Products run on tensor cores
//   (mma.m16n8k16 bf16 -> fp32): A = 16 weight rows x 16 k, B = 16 k x 8 tokens, so M = 1..8 tokens cost
//   the same weight traffic and instruction count as M = 1 (tokens >= M are zero columns).
// * Work unit: a "chunk" is 32 consecutive columns of one row (NVFP4: 16 B codes + 2 B scales; MXFP8:
//   32 B codes + 1 B scale). A warp owns one 128-column group of a 16-row tile: lane (g = lane/4,
//   t = lane%4) loads chunk 4*group + t of rows g and g+8, so a quad reads 64 B (NVFP4) / 128 B (MXFP8)
//   contiguous per row with 16-byte vector loads that bypass L1. The k order inside an mma is permuted
//   (lane t owns columns 32c+4i..32c+4i+3 of mma i) and B, read with 16-byte loads from L1/L2-resident
//   activations, uses the same permutation, so fragments need no shuffles and no shared memory.
// * Latency hiding: one chunk pair per warp per step and many warps per SM. (Giving a warp several chunks
//   does not help: ptxas sinks the extra loads next to their uses, measured 64% vs 89% of DRAM bandwidth.)
// * Expert deduplication for M > 1: the slots (token, k) that select the same expert share one pass over
//   its weights; tokens that do not select it get zero B columns and their outputs are discarded.
// * moe_up: grid (144 tiles of 16 W13 rows = 8 gate/up pairs) x (M*8 slots); a block whose slot is not the
//   first occurrence of its expert exits at once. 18 warps cover the 18 column groups of K = 2304; their
//   accumulators are summed in a fixed order through shared memory, then gate/up are rounded to BF16 and
//   the activation written for every slot that selected the expert.
// * moe_down: grid of 144 tiles of 16 W2 rows (output channels). A block handles all experts of all tokens
//   for its channels, so the weighted expert sum, its BF16 rounding and the post-MLP conv (with the state
//   update) are fused without atomics, workspaces or extra launches. 18 warps = 2 experts x 9 column
//   groups of K = 1152 per round; the next round's weights and activations are loaded into registers
//   before the current round is computed (ptxas cannot sink them past the round's barriers), so a block
//   keeps 18 KB (NVFP4) / 36 KB (MXFP8) of weights in flight. Per round the 9 partial dots of each expert
//   are summed in a fixed order and rounded to BF16; the weighted sum runs in slot order k = 0..7.
//   (Measured alternatives that were slower at M = 1: one warp per expert with 9 chunk pairs per lane
//   (ptxas sinks the loads), 4-block clusters sharing the tile through DSMEM (2x slower), and L2 prefetch
//   of later rounds.) Only 144 blocks exist, so the ~2 us of prologue/epilogue latency is visible; with
//   pdl (below) the prologue and first weight loads overlap the tail of moe_up.
// * Optional programmatic dependent launch (pdl=True): moe_up waits for its predecessor before reading
//   anything, then lets dependents launch; moe_down reads idx, enumerates experts and issues its first
//   round of weight loads before griddepcontrol.wait, and reads act/wts/conv state only after it. This is
//   only valid when the kernel before moe_down cannot still be writing idx (true for moe_up, which
//   triggers after its own wait). Without pdl both are ordinary stream-ordered launches.
// * Deterministic (fixed reduction orders) and capturable: no allocation, no host sync, current stream;
//   M is a launch-shape parameter of the captured graph and all routing is read from device memory.
// Ported from engine/csrc/moe/moe.cu (kernel bodies unchanged; torch launchers -> raw pointers).
#include "kernels/moe/moe.h"

#include "kernels/common/l2.cuh"

#include "core/device.h"

#include <cuda_bf16.h>
#include <cuda_fp16.h>

#include <cstdint>
#include <stdexcept>
#include <string>

namespace ingine::kernels::moe {
namespace {

constexpr int H = 2304;
constexpr int I = 1152;
constexpr int TOPK = 8;
constexpr int MAXM = 8;
constexpr int TILE = 16;                    // weight rows per block
constexpr int UP_WARPS = H / 128;           // 18: one column group of W13 per warp
constexpr int DOWN_GROUPS = I / 128;        // 9 column groups of W2
constexpr int DOWN_PAR = 2;                 // experts per round
constexpr int DOWN_WARPS = DOWN_GROUPS * DOWN_PAR;

// ------------------------------------------------------------------------------------------- loads

// Expert weights are read once per step: their L2 lines are filled evict-first (kernels/common/l2.cuh), so
// they stop evicting the activations, KV rings and the small kernels' evict-last constants.
constexpr l2::Evict WEIGHT_L2 = l2::Evict::first;

__device__ __forceinline__ uint4 ld_stream(const void* p, std::uint64_t pol) {
    uint4 v;
    if constexpr (l2::hinted<WEIGHT_L2>) {
        asm volatile("ld.global.nc.L1::no_allocate.L2::cache_hint.L2::256B.v4.u32 {%0,%1,%2,%3}, [%4], %5;"
                     : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w)
                     : "l"(p), "l"(pol));
    } else {
        asm volatile("ld.global.nc.L1::no_allocate.L2::256B.v4.u32 {%0,%1,%2,%3}, [%4];"
                     : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w)
                     : "l"(p));
    }
    return v;
}
__device__ __forceinline__ uint16_t ld_scale2(const void* p, std::uint64_t pol) {
    uint16_t v;
    if constexpr (l2::hinted<WEIGHT_L2>) asm volatile("ld.global.nc.L2::cache_hint.u16 %0, [%1], %2;" : "=h"(v) : "l"(p), "l"(pol));
    else asm volatile("ld.global.nc.u16 %0, [%1];" : "=h"(v) : "l"(p));
    return v;
}
__device__ __forceinline__ uint8_t ld_scale1(const void* p, std::uint64_t pol) {
    uint16_t v;
    if constexpr (l2::hinted<WEIGHT_L2>) asm volatile("ld.global.nc.L2::cache_hint.u8 %0, [%1], %2;" : "=h"(v) : "l"(p), "l"(pol));
    else asm volatile("ld.global.nc.u8 %0, [%1];" : "=h"(v) : "l"(p));
    return uint8_t(v);
}

// Inputs written by earlier kernels (x, act, idx, wts, conv_state) use coherent loads in volatile asm with a
// memory clobber: with programmatic dependent launch the producer may still be running when this kernel
// starts, so these loads must neither be non-coherent (.nc requires data read-only for the kernel's
// lifetime) nor be movable above griddepcontrol.wait (an __ldg is an invariant load the compiler may hoist).
__device__ __forceinline__ uint4 ld_dep(const void* p) {
    uint4 v;
    asm volatile("ld.global.v4.u32 {%0,%1,%2,%3}, [%4];"
                 : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w)
                 : "l"(p)
                 : "memory");
    return v;
}
__device__ __forceinline__ int ld_dep(const int* p) {
    int v;
    asm volatile("ld.global.s32 %0, [%1];" : "=r"(v) : "l"(p) : "memory");
    return v;
}
__device__ __forceinline__ float ld_dep(const float* p) {
    float v;
    asm volatile("ld.global.f32 %0, [%1];" : "=f"(v) : "l"(p) : "memory");
    return v;
}

// One 32-column chunk of one row in checkpoint encoding.
template <int FMT>
struct Chunk;
template <>
struct Chunk<0> {  // NVFP4: 32 E2M1 codes (even element in the low nibble) + 2 E4M3 block scales
    uint4 q;
    uint16_t s;
};
template <>
struct Chunk<1> {  // MXFP8: 32 E4M3 codes + 1 E8M0 block scale
    uint4 q0, q1;
    uint8_t s;
};

template <int FMT>
struct Layout {
    // bytes per row of codes / scales for a matrix with K columns
    static constexpr int code_bytes(int K) { return FMT == 0 ? K / 2 : K; }
    static constexpr int scale_bytes(int K) { return FMT == 0 ? K / 16 : K / 32; }
};

template <int FMT>
__device__ __forceinline__ Chunk<FMT> load_chunk(const uint8_t* codes, const uint8_t* scales, int c, std::uint64_t pol) {
    Chunk<FMT> ch;
    if constexpr (FMT == 0) {
        ch.q = ld_stream(codes + 16 * c, pol);
        ch.s = ld_scale2(scales + 2 * c, pol);
    } else {
        ch.q0 = ld_stream(codes + 32 * c, pol);
        ch.q1 = ld_stream(codes + 32 * c + 16, pol);
        ch.s = ld_scale1(scales + c, pol);
    }
    return ch;
}

// ------------------------------------------------------------------------------------------ decode

__device__ __forceinline__ uint32_t f16x2_from_e4m3x2(uint16_t v) {
    uint32_t r;
    asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(r) : "h"(v));
    return r;
}

__device__ __forceinline__ void f16x2_from_e2m1x4(uint32_t w, uint32_t (&h)[4]) {
    asm("{\n"
        ".reg .b8 b0, b1, b2, b3;\n"
        "mov.b32 {b0, b1, b2, b3}, %4;\n"
        "cvt.rn.f16x2.e2m1x2 %0, b0;\n"
        "cvt.rn.f16x2.e2m1x2 %1, b1;\n"
        "cvt.rn.f16x2.e2m1x2 %2, b2;\n"
        "cvt.rn.f16x2.e2m1x2 %3, b3;\n"
        "}"
        : "=r"(h[0]), "=r"(h[1]), "=r"(h[2]), "=r"(h[3])
        : "r"(w));
}

__device__ __forceinline__ uint32_t pack_bf16(float lo, float hi) {
    __nv_bfloat162 v = __floats2bfloat162_rn(lo, hi);
    return *reinterpret_cast<uint32_t*>(&v);
}

// Decodes a chunk into 16 bf16x2 words (element pairs 0..15) holding exactly decode_experts' values.
__device__ __forceinline__ void decode(const Chunk<0>& ch, float factor, uint32_t (&o)[16]) {
    const uint32_t s2 = f16x2_from_e4m3x2(ch.s);
    const __half2 sc = *reinterpret_cast<const __half2*>(&s2);
    const __half2 slo = __low2half2(sc), shi = __high2half2(sc);
    const uint32_t words[4] = {ch.q.x, ch.q.y, ch.q.z, ch.q.w};
#pragma unroll
    for (int w = 0; w < 4; ++w) {
        uint32_t h[4];
        f16x2_from_e2m1x4(words[w], h);
#pragma unroll
        for (int b = 0; b < 4; ++b) {
            // code * scale is exact in f16 (<= 5 significant bits, |v| in [2^-10, 2688]); the factor
            // multiply rounds once in fp32, then to bf16, as in decode_experts.
            const __half2 cs = __hmul2(*reinterpret_cast<const __half2*>(&h[b]), w < 2 ? slo : shi);
            const float2 v = __half22float2(cs);
            o[4 * w + b] = pack_bf16(v.x * factor, v.y * factor);
        }
    }
}

__device__ __forceinline__ void decode(const Chunk<1>& ch, float /*factor*/, uint32_t (&o)[16]) {
    // E8M0 -> 2^(s-127); s = 0 is 2^-127 (fp32 subnormal), s = 255 would be NaN (never produced).
    const float sc = ch.s ? __uint_as_float(uint32_t(ch.s) << 23) : __uint_as_float(0x00400000u);
    const uint32_t words[8] = {ch.q0.x, ch.q0.y, ch.q0.z, ch.q0.w, ch.q1.x, ch.q1.y, ch.q1.z, ch.q1.w};
#pragma unroll
    for (int w = 0; w < 8; ++w) {
#pragma unroll
        for (int hh = 0; hh < 2; ++hh) {
            const uint32_t h2 = f16x2_from_e4m3x2(uint16_t(words[w] >> (16 * hh)));
            const float2 v = __half22float2(*reinterpret_cast<const __half2*>(&h2));
            o[2 * w + hh] = pack_bf16(v.x * sc, v.y * sc);  // exact: fp8 value times a power of two
        }
    }
}

// ------------------------------------------------------------------------------------------- mma

__device__ __forceinline__ void mma_bf16(float (&d)[4], uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3,
                                         uint32_t b0, uint32_t b1) {
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, "
        "{%0,%1,%2,%3};"
        : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
        : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
}

// One warp step: chunk c of rows g and g+8 of a 16-row tile, plus the lane's B column (token g) for the
// same 32 columns (zero if the token does not use this expert).
template <int FMT>
struct Frag {
    Chunk<FMT> r0, r1;
    uint32_t x[16];
    float factor;
};

// Weights: `codes`/`scales` point at row g of the tile (row g+8 is 8 rows further).
template <int FMT, int K>
__device__ __forceinline__ void load_w(Frag<FMT>& f, const uint8_t* codes, const uint8_t* scales, int c,
                                       float factor, std::uint64_t pol) {
    constexpr int CB = Layout<FMT>::code_bytes(K), SB = Layout<FMT>::scale_bytes(K);
    f.r0 = load_chunk<FMT>(codes, scales, c, pol);
    f.r1 = load_chunk<FMT>(codes + 8 * CB, scales + 8 * SB, c, pol);
    f.factor = factor;
}

// Activations: `xrow` is the lane's token activation row (K bf16) or nullptr (zero column).
template <int FMT>
__device__ __forceinline__ void load_x(Frag<FMT>& f, const __nv_bfloat16* xrow, int c) {
    if (xrow) {
        const uint8_t* xp = reinterpret_cast<const uint8_t*>(xrow + 32 * c);
#pragma unroll
        for (int v = 0; v < 4; ++v) {
            const uint4 q = ld_dep(xp + 16 * v);
            f.x[4 * v] = q.x, f.x[4 * v + 1] = q.y, f.x[4 * v + 2] = q.z, f.x[4 * v + 3] = q.w;
        }
    } else {
#pragma unroll
        for (int v = 0; v < 16; ++v) f.x[v] = 0;
    }
}

// Programmatic dependent launch (no-ops when the kernel is launched without the PDL attribute).
__device__ __forceinline__ void pdl_wait() { asm volatile("griddepcontrol.wait;" ::: "memory"); }
__device__ __forceinline__ void pdl_trigger() { asm volatile("griddepcontrol.launch_dependents;" ::: "memory"); }

template <int FMT>
__device__ __forceinline__ void mma_frag(const Frag<FMT>& f, float (&d)[4]) {
    uint32_t a[16], b[16];
    decode(f.r0, f.factor, a);
    decode(f.r1, f.factor, b);
#pragma unroll
    for (int i = 0; i < 8; ++i) mma_bf16(d, a[2 * i], b[2 * i], a[2 * i + 1], b[2 * i + 1], f.x[2 * i], f.x[2 * i + 1]);
}

// Slots 0..63 (token * 8 + k) whose expert equals e, from a warp-wide view of idx (lo = idx[lane],
// hi = idx[lane + 32], -1 beyond M*8).
__device__ __forceinline__ uint64_t slot_mask(int e, int lo, int hi) {
    const uint32_t m0 = __ballot_sync(0xffffffffu, lo == e);
    const uint32_t m1 = __ballot_sync(0xffffffffu, hi == e);
    return uint64_t(m0) | (uint64_t(m1) << 32);
}

__device__ __forceinline__ float bf16_round(float v) { return __bfloat162float(__float2bfloat16_rn(v)); }

// --------------------------------------------------------------------------------------- moe_up

template <int FMT>
__global__ void __launch_bounds__(UP_WARPS * 32)
moe_up_kernel(const __nv_bfloat16* __restrict__ x, const int* __restrict__ idx, const uint8_t* __restrict__ w,
              const uint8_t* __restrict__ ws, const float* __restrict__ wfactor, int M,
              __nv_bfloat16* __restrict__ out) {
    constexpr int K = H, R = 2 * I;
    constexpr int CB = Layout<FMT>::code_bytes(K), SB = Layout<FMT>::scale_bytes(K);
    __shared__ float red[UP_WARPS][32][4];
    __shared__ float tile_out[TILE][MAXM + 1];

    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    const int g = lane >> 2, t = lane & 3;
    const int tile = blockIdx.x, slot = blockIdx.y;
    const int nslots = M * TOPK;
    pdl_wait();     // x and idx come from earlier kernels
    pdl_trigger();  // moe_down may launch once every block got here (it waits for us before reading act)
    const int lo = lane < nslots ? ld_dep(idx + lane) : -1;
    const int hi = lane + 32 < nslots ? ld_dep(idx + lane + 32) : -1;
    const int e = ld_dep(idx + slot);
    const uint64_t mask = slot_mask(e, lo, hi);
    if (__ffsll(static_cast<long long>(mask)) - 1 != slot) return;  // an earlier slot owns this expert

    const size_t row = static_cast<size_t>(e) * R + tile * TILE + g;
    const bool on = g < M && ((mask >> (8 * g)) & 0xffu);
    Frag<FMT> f;
    load_w<FMT, K>(f, w + row * CB, ws + row * SB, 4 * warp + t, FMT == 0 ? wfactor[e] : 1.f,
                   l2::policy<WEIGHT_L2>());
    load_x(f, on ? x + static_cast<size_t>(g) * K : nullptr, 4 * warp + t);
    float d[4] = {0.f, 0.f, 0.f, 0.f};
    mma_frag(f, d);

#pragma unroll
    for (int v = 0; v < 4; ++v) red[warp][lane][v] = d[v];
    __syncthreads();
    if (threadIdx.x < 32) {
        float s[4] = {0.f, 0.f, 0.f, 0.f};
#pragma unroll
        for (int ww = 0; ww < UP_WARPS; ++ww)
#pragma unroll
            for (int v = 0; v < 4; ++v) s[v] += red[ww][lane][v];
        tile_out[g][2 * t] = s[0];
        tile_out[g][2 * t + 1] = s[1];
        tile_out[g + 8][2 * t] = s[2];
        tile_out[g + 8][2 * t + 1] = s[3];
    }
    __syncthreads();
    if (threadIdx.x < TILE / 2 * MAXM) {
        const int p = threadIdx.x & 7, n = threadIdx.x >> 3;  // gate/up pair p, token n
        uint32_t kmask = n < M ? uint32_t(mask >> (8 * n)) & 0xffu : 0u;
        if (kmask) {
            const float gate = bf16_round(tile_out[2 * p][n]);
            const float up = bf16_round(tile_out[2 * p + 1][n]);
            const float a = tanhf(gate / 4.f) * 4.f * (1.f / (1.f + expf(-gate))) * (tanhf(up / 25.f) * 25.f);
            const __nv_bfloat16 ab = __float2bfloat16_rn(a);
            while (kmask) {
                const int k = __ffs(kmask) - 1;
                kmask &= kmask - 1;
                out[(static_cast<size_t>(n) * TOPK + k) * I + tile * (TILE / 2) + p] = ab;
            }
        }
    }
}

// ------------------------------------------------------------------------------------- moe_down

template <int FMT>
__global__ void __launch_bounds__(DOWN_WARPS * 32)
moe_down_kernel(const __nv_bfloat16* __restrict__ act, const int* __restrict__ idx, const float* __restrict__ wts,
                const uint8_t* __restrict__ w, const uint8_t* __restrict__ ws, const float* __restrict__ wfactor,
                int M, const float* __restrict__ conv_w, float* __restrict__ conv_state, float* __restrict__ out,
                float* __restrict__ conv_hist) {
    constexpr int K = I, R = H;
    constexpr int CB = Layout<FMT>::code_bytes(K), SB = Layout<FMT>::scale_bytes(K);
    __shared__ float part[DOWN_WARPS][32][4];        // per-warp partial dots of the current round
    __shared__ float ybuf[TOPK * MAXM][TILE][MAXM];  // BF16-rounded expert outputs [unique expert][row][token]
    __shared__ int uexp[TOPK * MAXM];                // unique experts in first-slot order
    __shared__ unsigned long long umask[TOPK * MAXM];  // slots selecting each unique expert
    __shared__ uint8_t slot_u[TOPK * MAXM];          // slot -> unique expert index
    __shared__ int nuniq;
    __shared__ float sbuf[MAXM][TILE];

    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    const int g = lane >> 2, t = lane & 3;
    const int tile = blockIdx.x;
    const int nslots = M * TOPK;

    if (warp == 0) {
        // idx is read before pdl_wait on purpose: it was written by the router, which completed before
        // moe_up passed its own wait and triggered this launch.
        const int lo = lane < nslots ? ld_dep(idx + lane) : -1;
        const int hi = lane + 32 < nslots ? ld_dep(idx + lane + 32) : -1;
        uint64_t remaining = nslots == 64 ? ~0ull : ((1ull << nslots) - 1);
        int u = 0;
        for (; remaining; ++u) {
            const int s = __ffsll(static_cast<long long>(remaining)) - 1;
            const int e = __shfl_sync(0xffffffffu, s < 32 ? lo : hi, s & 31);
            const uint64_t mask = slot_mask(e, lo, hi);
            remaining &= ~mask;
            if (lane == 0) uexp[u] = e, umask[u] = mask;
            if ((mask >> lane) & 1) slot_u[lane] = uint8_t(u);
            if ((mask >> (lane + 32)) & 1) slot_u[lane + 32] = uint8_t(u);
        }
        if (lane == 0) nuniq = u;
    }
    __syncthreads();

    const int U = nuniq;
    const int p = warp / DOWN_GROUPS, c = 4 * (warp % DOWN_GROUPS) + t;
    auto xrow = [&](int u) -> const __nv_bfloat16* {
        const uint32_t kmask = g < M ? uint32_t(umask[u] >> (8 * g)) & 0xffu : 0u;
        return kmask ? act + (static_cast<size_t>(g) * TOPK + (__ffs(kmask) - 1)) * K : nullptr;
    };
    const std::uint64_t wpol = l2::policy<WEIGHT_L2>();
    auto load = [&](int u, Frag<FMT>& f) {
        const int e = uexp[u];
        const size_t row = static_cast<size_t>(e) * R + tile * TILE + g;
        load_w<FMT, K>(f, w + row * CB, ws + row * SB, c, FMT == 0 ? wfactor[e] : 1.f, wpol);
    };

    // The first round's weights do not depend on moe_up: fetch them before waiting for it (PDL).
    Frag<FMT> nxt;
    if (p < U) load(p, nxt);
    pdl_wait();
    if (p < U) load_x(nxt, xrow(p), c);
#pragma unroll 1
    for (int base = 0; base < U; base += DOWN_PAR) {
        const Frag<FMT> cur = nxt;
        const int u = base + p;
        if (u + DOWN_PAR < U) load(u + DOWN_PAR, nxt), load_x(nxt, xrow(u + DOWN_PAR), c);  // next round
        float d[4] = {0.f, 0.f, 0.f, 0.f};
        if (u < U) mma_frag(cur, d);
#pragma unroll
        for (int v = 0; v < 4; ++v) part[warp][lane][v] = d[v];
        __syncthreads();
        if (threadIdx.x < DOWN_PAR * 32 && base + (threadIdx.x >> 5) < U) {
            const int pp = threadIdx.x >> 5;
            float s[4] = {0.f, 0.f, 0.f, 0.f};
#pragma unroll
            for (int j = 0; j < DOWN_GROUPS; ++j)
#pragma unroll
                for (int v = 0; v < 4; ++v) s[v] += part[pp * DOWN_GROUPS + j][lane][v];
            float(&y)[TILE][MAXM] = ybuf[base + pp];
            y[g][2 * t] = bf16_round(s[0]);
            y[g][2 * t + 1] = bf16_round(s[1]);
            y[g + 8][2 * t] = bf16_round(s[2]);
            y[g + 8][2 * t + 1] = bf16_round(s[3]);
        }
        __syncthreads();
    }
    pdl_trigger();  // only the short epilogue remains; dependents still wait for our completion

    if (threadIdx.x < TILE * M) {
        const int r = threadIdx.x & (TILE - 1), n = threadIdx.x / TILE;
        float s = 0.f;
#pragma unroll
        for (int k = 0; k < TOPK; ++k) {
            const int sl = n * TOPK + k;
            s = __fadd_rn(s, __fmul_rn(ld_dep(wts + sl), ybuf[slot_u[sl]][r][n]));
        }
        sbuf[n][r] = bf16_round(s);
    }
    __syncthreads();

    if (threadIdx.x < TILE) {
        const int ch = tile * TILE + threadIdx.x;
        const float4 cw = *reinterpret_cast<const float4*>(conv_w + 4 * ch);
        float ext[MAXM + 3];
        ext[0] = ld_dep(conv_state + ch);
        ext[1] = ld_dep(conv_state + H + ch);
        ext[2] = ld_dep(conv_state + 2 * H + ch);
#pragma unroll
        for (int m = 0; m < MAXM; ++m) ext[m + 3] = m < M ? sbuf[m][threadIdx.x] : 0.f;
        constexpr float inv_sqrt5 = 1.f / 2.2360679774997896f;  // torch divides by a scalar via its reciprocal
#pragma unroll
        for (int m = 0; m < MAXM; ++m) {
            if (m < M) {
                // same op order and roundings as model.static_conv in FP32 (bit-exact)
                float acc = __fmul_rn(cw.x, ext[m]);
                acc = __fadd_rn(acc, __fmul_rn(cw.y, ext[m + 1]));
                acc = __fadd_rn(acc, __fmul_rn(cw.z, ext[m + 2]));
                acc = __fadd_rn(acc, __fmul_rn(cw.w, ext[m + 3]));
                out[static_cast<size_t>(m) * H + ch] = __fmul_rn(__fadd_rn(acc, ext[m + 3]), inv_sqrt5);
                if (conv_hist) conv_hist[static_cast<size_t>(m) * H + ch] = ext[m + 3];  // for rollback
            }
        }
#pragma unroll
        for (int j = 0; j < 3; ++j)
#pragma unroll
            for (int m = 1; m <= MAXM; ++m)  // constant indices keep ext[] in registers
                if (m == M) conv_state[j * H + ch] = ext[m + j];
    }
}

void require(bool ok, const char* what) {
    if (!ok) throw std::invalid_argument(std::string("moe: ") + what);
}

// Launch on `stream`; with pdl the kernel may start before its predecessor finishes
// (programmatic dependent launch, captured as a programmatic edge in CUDA graphs).
template <typename... Params, typename... Args>
void launch(void (*kernel)(Params...), dim3 grid, int threads, bool pdl, cudaStream_t stream, Args... args) {
    cudaLaunchConfig_t cfg = {};
    cfg.gridDim = grid;
    cfg.blockDim = dim3(threads);
    cfg.stream = stream;
    cudaLaunchAttribute attr[1];
    attr[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
    attr[0].val.programmaticStreamSerializationAllowed = 1;
    cfg.attrs = attr;
    cfg.numAttrs = pdl ? 1 : 0;
    CUDA_CHECK(cudaLaunchKernelEx(&cfg, kernel, static_cast<Params>(args)...));
}

}  // namespace

void moe_up(const __nv_bfloat16* x, int M, const int* idx, const void* w13, const void* w13_scale,
            const float* w13_factor, Format fmt, __nv_bfloat16* out, bool pdl, cudaStream_t stream) {
    require(M >= 1 && M <= MAXM, "M must be in 1..8");
    require(x && idx && w13 && w13_scale && out, "moe_up: null argument");
    require(fmt == Format::MXFP8 || w13_factor, "NVFP4 experts need a factor tensor");
    const bool nv = fmt == Format::NVFP4;
    const dim3 grid(2 * I / TILE, M * TOPK);
    launch(nv ? moe_up_kernel<0> : moe_up_kernel<1>, grid, UP_WARPS * 32, pdl, stream, x, idx,
           static_cast<const uint8_t*>(w13), static_cast<const uint8_t*>(w13_scale), nv ? w13_factor : nullptr, M,
           out);
}

void moe_down(const __nv_bfloat16* act, int M, const int* idx, const float* wts, const void* w2,
              const void* w2_scale, const float* w2_factor, Format fmt, const float* conv_w, float* conv_state,
              float* out, float* conv_hist, bool pdl, cudaStream_t stream) {
    require(M >= 1 && M <= MAXM, "M must be in 1..8");
    require(act && idx && wts && w2 && w2_scale && conv_w && conv_state && out, "moe_down: null argument");
    require(fmt == Format::MXFP8 || w2_factor, "NVFP4 experts need a factor tensor");
    const bool nv = fmt == Format::NVFP4;
    launch(nv ? moe_down_kernel<0> : moe_down_kernel<1>, dim3(H / TILE), DOWN_WARPS * 32, pdl, stream, act, idx,
           wts, static_cast<const uint8_t*>(w2), static_cast<const uint8_t*>(w2_scale), nv ? w2_factor : nullptr,
           M, conv_w, conv_state, out, conv_hist);
}

}  // namespace ingine::kernels::moe
