// DSpark draft block attention (dspark.Drafter._attention) on tensor cores: partial softmax-weighted sums
// over key chunks, then a combine pass. PDL launches (see block.cu).
//
// A CTA of the partial kernel handles one kv head's 24 queries (3 heads x 8 block rows) over one chunk of
// keys, a tile of ATT_TILE = 16 keys at a time. Warp h < 3 owns query head 3 g + h (8 queries = the n of
// mma.m16n8k16); all four warps load and stage the tile:
//   load    every thread loads 16 bytes of the tile's keys and of its values (BF16 rows: two loads each), so a
//           load instruction reads whole rows; they go to shared memory as BF16 [16 keys][128] (E4M3 rows are
//           converted, one instruction per pair). Two tiles are in flight: a tile's registers are reloaded as
//           soon as it is staged. One barrier per tile (the staged tiles alternate between two buffers).
//   scores  S^T = K Q^T: A = 16 keys x 16 dims (ldmatrix), B = 16 dims x 8 queries (registers), 8 mma.
//   softmax running max/sum per query over the lanes holding its 16 keys; probabilities rounded to BF16.
//   values  O^T += V^T P^T: A = 16 dims x 16 keys (ldmatrix.trans of the staged values), B = 16 keys x 8
//           queries (the probabilities, transposed through shared memory), 8 mma.
// About 350 warp instructions per tile (8 KB of BF16 rows), where the CUDA-core kernel this replaces took
// 0.5 per byte and was compute bound (64K keys: 305 us for 268 MB; local layers: 10 us for 4 MB).
#include "kernels/draft/draft.h"
#include "kernels/draft/draft_device.cuh"

#include <algorithm>
#include <stdexcept>

namespace ingine::kernels::draft {
namespace {

constexpr int QH = HEADS / KV_HEADS;  // 3 query heads per kv head
constexpr int NQ = QH * BLOCK;        // 24 queries per kv head: query qi = head * 8 + row
constexpr int ATT_TILE = 16;          // keys per tile (the m of the score mma, the k of the value mma)
constexpr int ATT_THREADS = 128;      // warps 0..2: a query head each; all four stage the tiles
constexpr int ATT_MAX_CHUNKS = 128;   // chunks per kv head (at most; one combine thread each)
#ifndef ATT_MIN_TILES
#define ATT_MIN_TILES 2               // tiles per chunk (at least)
#endif
static_assert(QH == 3 && BLOCK == 8 && HEAD_DIM == 128 && ATT_MAX_CHUNKS <= HEAD_DIM);

int att_tiles_per_chunk(int nkeys) {
  return std::max(ATT_MIN_TILES, (nkeys + ATT_TILE * ATT_MAX_CHUNKS - 1) / (ATT_TILE * ATT_MAX_CHUNKS));
}
int att_chunks(int nkeys) {
  const int ck = ATT_TILE * att_tiles_per_chunk(nkeys);
  return (nkeys + ck - 1) / ck;
}

__device__ __forceinline__ void mma_bf16(float (&c)[4], const uint32_t (&a)[4], const uint32_t (&b)[2]) {
  asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, "
               "{%0,%1,%2,%3};"
               : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
               : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
}
// ldmatrix: lane l < 8 n gives the address of row l % 8 (8 BF16) of matrix l / 8; every lane (g, t) receives
// elements (g, 2 t), (g, 2 t + 1) of each matrix, or with `trans` elements (2 t, g), (2 t + 1, g).
__device__ __forceinline__ void ldmatrix_x4(uint32_t (&r)[4], const void* row) {
  const unsigned addr = static_cast<unsigned>(__cvta_generic_to_shared(row));
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];"
               : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3])
               : "r"(addr)
               : "memory");
}
__device__ __forceinline__ void ldmatrix_x4_trans(uint32_t (&r)[4], const void* row) {
  const unsigned addr = static_cast<unsigned>(__cvta_generic_to_shared(row));
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0,%1,%2,%3}, [%4];"
               : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3])
               : "r"(addr)
               : "memory");
}
__device__ __forceinline__ void ldmatrix_x2(uint32_t (&r)[2], const void* row) {
  const unsigned addr = static_cast<unsigned>(__cvta_generic_to_shared(row));
  asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0,%1}, [%2];" : "=r"(r[0]), "=r"(r[1]) : "r"(addr) : "memory");
}

// 16-byte coherent load issued where it is written (queries and cache rows are written by earlier kernels).
// The caches are read once per draft: no L1 allocation.
__device__ __forceinline__ uint4 ld_cache(const void* p) {
  uint4 r;
  asm volatile("ld.global.L1::no_allocate.v4.u32 {%0,%1,%2,%3}, [%4];"
               : "=r"(r.x), "=r"(r.y), "=r"(r.z), "=r"(r.w)
               : "l"(p)
               : "memory");
  return r;
}

// A tile's raw keys and values in registers (16 bytes per thread and load) and, in threads 0..15 of local
// layers, the ring position of key t0 + thread.
template <bool FP8>
struct TileRegs {
  static constexpr int NL = FP8 ? 1 : 2;  // loads per thread: a tile's keys are 16 x 128 (x 2) bytes
  uint4 k[NL], v[NL];
  int cp;
};

// grid (chunks, KV_HEADS), block 128. LOCAL: keys = ring rows [0, WINDOW) valid when P - WINDOW <= ctx_pos < P
// plus the block rows; else keys [0, nkeys) valid below P + 8. FP8: E4M3 cache rows (else BF16).
// ws_o fp32 [chunks, 8, 24 queries, 128 dims], ws_ml fp32 [chunks, 8, 24, 2] (running max, sum).
template <bool LOCAL, bool FP8>
__global__ void __launch_bounds__(ATT_THREADS) attn_partial_kernel(const __nv_bfloat16* q, const uint8_t* k_cache,
                                                                  const uint8_t* v_cache, int nkeys, int chunk_keys,
                                                                  const std::int32_t* ctx_pos, const DraftParams* p,
                                                                  float* ws_o, float* ws_ml) {
  constexpr int ROW_BYTES = FP8 ? HEAD_DIM : 2 * HEAD_DIM;
  constexpr int NL = TileRegs<FP8>::NL;
  __shared__ __align__(16) __nv_bfloat16 qs[QH][BLOCK][HEAD_DIM];    // the queries, by head
  __shared__ __align__(16) __nv_bfloat16 kt[2][ATT_TILE][HEAD_DIM];  // keys of a tile, by tile parity
  __shared__ __align__(16) __nv_bfloat16 vt[2][ATT_TILE][HEAD_DIM];  // values
  __shared__ __align__(16) __nv_bfloat16 pt[QH][BLOCK][ATT_TILE];    // a head's probabilities [query][key]
  __shared__ unsigned vmask[2];                                      // valid keys of a tile
  pdl_trigger();
  pdl_wait();
  const int chunk = blockIdx.x, g = blockIdx.y, tid = threadIdx.x, w = tid >> 5, lane = tid & 31;
  const int gl = lane >> 2, t = lane & 3, lm = lane >> 3, lr = lane & 7;  // ldmatrix: matrix lm, row lr
  const int k0 = chunk * chunk_keys, k1 = min(k0 + chunk_keys, nkeys);
  const int P = p->P;
  const int kend = LOCAL ? k1 : min(k1, P + BLOCK);  // global: no valid key at or beyond P + 8
  constexpr float scale = 0.08838834764831845f;      // 1 / sqrt(128)

  auto load = [&](TileRegs<FP8>& x, int t0) {
#pragma unroll
    for (int j = 0; j < NL; ++j) {
      const int idx = tid + ATT_THREADS * j;
      const int key = t0 + idx / (ROW_BYTES / 16), piece = idx % (ROW_BYTES / 16);
      const size_t at = (size_t(key) * KV_HEADS + g) * ROW_BYTES + 16 * piece;
      const bool live = key < kend;
      x.k[j] = live ? ld_cache(k_cache + at) : make_uint4(0, 0, 0, 0);
      x.v[j] = live ? ld_cache(v_cache + at) : make_uint4(0, 0, 0, 0);
    }
    x.cp = 0;
    if constexpr (LOCAL) {
      if (tid < ATT_TILE && t0 + tid < WINDOW) x.cp = ctx_pos[t0 + tid];
    }
  };
  TileRegs<FP8> ra, rb;
  if (k0 < kend) load(ra, k0);
  if (k0 + ATT_TILE < kend) load(rb, k0 + ATT_TILE);

  // the queries of heads 3 g + (0, 1, 2), rows 0..7
  {
    uint4 qv[QH];
#pragma unroll
    for (int j = 0; j < QH; ++j) {
      const int idx = tid + ATT_THREADS * j, hh = idx >> 7, r = (idx >> 4) & 7, piece = idx & 15;
      qv[j] = *reinterpret_cast<const uint4*>(q + (size_t(r) * HEADS + g * QH + hh) * HEAD_DIM + 8 * piece);
    }
#pragma unroll
    for (int j = 0; j < QH; ++j) {
      const int idx = tid + ATT_THREADS * j;
      *reinterpret_cast<uint4*>(&qs[idx >> 7][(idx >> 4) & 7][8 * (idx & 15)]) = qv[j];
    }
  }
  __syncthreads();
  const bool head = w < QH;       // warp-uniform
  const int hw = head ? w : 0;
  uint32_t qb[HEAD_DIM / 16][2];  // this head's queries as B fragments (16 dims x 8 queries), per 16 dims
#pragma unroll
  for (int s = 0; s < HEAD_DIM / 16; ++s) ldmatrix_x2(qb[s], &qs[hw][lr][16 * s + 8 * (lm & 1)]);

  float m_run[2] = {-INFINITY, -INFINITY}, l_run[2] = {0.f, 0.f};  // queries 2 t, 2 t + 1
  float oacc[HEAD_DIM / 16][4];  // value sums: dims 16 d + gl (0, 1) and 16 d + 8 + gl (2, 3) of queries 2 t, 2 t + 1
#pragma unroll
  for (int d = 0; d < HEAD_DIM / 16; ++d)
#pragma unroll
    for (int i = 0; i < 4; ++i) oacc[d][i] = 0.f;

  auto tile = [&](TileRegs<FP8>& x, int t0, int buf) {
    // stage the tile as BF16 [16 keys][128]
#pragma unroll
    for (int j = 0; j < NL; ++j) {
      const int idx = tid + ATT_THREADS * j;
      const int key = idx / (ROW_BYTES / 16), piece = idx % (ROW_BYTES / 16);
      if constexpr (FP8) {
        const uint32_t kw[4] = {x.k[j].x, x.k[j].y, x.k[j].z, x.k[j].w}, vw[4] = {x.v[j].x, x.v[j].y, x.v[j].z, x.v[j].w};
        uint32_t kb[8], vb[8];
#pragma unroll
        for (int i = 0; i < 4; ++i) {
          kb[2 * i] = e4m3x2_to_bf16x2(kw[i]);
          kb[2 * i + 1] = e4m3x2_to_bf16x2(kw[i] >> 16);
          vb[2 * i] = e4m3x2_to_bf16x2(vw[i]);
          vb[2 * i + 1] = e4m3x2_to_bf16x2(vw[i] >> 16);
        }
        uint4* kd = reinterpret_cast<uint4*>(&kt[buf][key][16 * piece]);
        uint4* vd = reinterpret_cast<uint4*>(&vt[buf][key][16 * piece]);
        kd[0] = make_uint4(kb[0], kb[1], kb[2], kb[3]);
        kd[1] = make_uint4(kb[4], kb[5], kb[6], kb[7]);
        vd[0] = make_uint4(vb[0], vb[1], vb[2], vb[3]);
        vd[1] = make_uint4(vb[4], vb[5], vb[6], vb[7]);
      } else {
        *reinterpret_cast<uint4*>(&kt[buf][key][8 * piece]) = x.k[j];
        *reinterpret_cast<uint4*>(&vt[buf][key][8 * piece]) = x.v[j];
      }
    }
    if (w == 0) {
      const int j = t0 + lane;
      bool v = lane < ATT_TILE && j < kend;
      if constexpr (LOCAL) {
        if (j < WINDOW) v = v && x.cp >= 0 && x.cp < P && x.cp >= P - WINDOW;
      }
      const unsigned mask = __ballot_sync(0xffffffffu, v);
      if (lane == 0) vmask[buf] = mask;
    }
    __syncthreads();
    // these registers are free: the tile after the next one goes in flight (block-uniform condition)
    if (t0 + 2 * ATT_TILE < kend) load(x, t0 + 2 * ATT_TILE);
    if (!head) return;

    // scores: keys gl (0, 1) and gl + 8 (2, 3) against queries 2 t, 2 t + 1
    float sc[4] = {0.f, 0.f, 0.f, 0.f};
#pragma unroll
    for (int s = 0; s < HEAD_DIM / 16; ++s) {
      uint32_t a[4];
      ldmatrix_x4(a, &kt[buf][8 * (lm & 1) + lr][16 * s + 8 * (lm >> 1)]);
      mma_bf16(sc, a, qb[s]);
    }
    const unsigned mask = vmask[buf];
    const bool valid[2] = {((mask >> gl) & 1u) != 0, ((mask >> (gl + 8)) & 1u) != 0};
    float al[2];
    __nv_bfloat16 pb16[2][2];  // [query e][key gl + 8 kk]
#pragma unroll
    for (int e = 0; e < 2; ++e) {
      float xs[2], mt = -INFINITY;
#pragma unroll
      for (int kk = 0; kk < 2; ++kk) {
        xs[kk] = valid[kk] ? sc[2 * kk + e] * scale : -INFINITY;
        mt = fmaxf(mt, xs[kk]);
      }
#pragma unroll
      for (int sh = 4; sh < 32; sh <<= 1) mt = fmaxf(mt, __shfl_xor_sync(0xffffffffu, mt, sh));
      const float m_new = fmaxf(m_run[e], mt);
      al[e] = m_new == m_run[e] ? 1.f : (m_run[e] == -INFINITY ? 0.f : __expf(m_run[e] - m_new));
      float ps = 0.f;
#pragma unroll
      for (int kk = 0; kk < 2; ++kk) {
        pb16[e][kk] = __float2bfloat16_rn(valid[kk] ? __expf(xs[kk] - m_new) : 0.f);  // the value mma takes BF16
        ps += __bfloat162float(pb16[e][kk]);
      }
#pragma unroll
      for (int sh = 4; sh < 32; sh <<= 1) ps += __shfl_xor_sync(0xffffffffu, ps, sh);
      l_run[e] = l_run[e] * al[e] + ps;
      m_run[e] = m_new;
    }
    // the probabilities as a B fragment (16 keys x 8 queries): transposed through shared memory
#pragma unroll
    for (int e = 0; e < 2; ++e)
#pragma unroll
      for (int kk = 0; kk < 2; ++kk) pt[w][2 * t + e][gl + 8 * kk] = pb16[e][kk];
    __syncwarp();
    uint32_t pf[2];
    ldmatrix_x2(pf, &pt[w][lr][8 * (lm & 1)]);
    // O^T = O^T alpha + V^T P^T: the tile's 16 keys are the k of the mma
    const bool rescale = __any_sync(0xffffffffu, al[0] != 1.f || al[1] != 1.f);
#pragma unroll
    for (int d = 0; d < HEAD_DIM / 16; ++d) {
      if (rescale) {
        oacc[d][0] *= al[0];
        oacc[d][1] *= al[1];
        oacc[d][2] *= al[0];
        oacc[d][3] *= al[1];
      }
      uint32_t a[4];
      ldmatrix_x4_trans(a, &vt[buf][8 * (lm >> 1) + lr][16 * d + 8 * (lm & 1)]);
      mma_bf16(oacc[d], a, pf);
    }
  };
  for (int t0 = k0; t0 < kend; t0 += 2 * ATT_TILE) {
    tile(ra, t0, 0);
    if (t0 + ATT_TILE < kend) tile(rb, t0 + ATT_TILE, 1);
  }
  if (!head) return;

  // partial outputs of queries qi = 8 w + 2 t + (0, 1) at dims 16 d + gl and 16 d + 8 + gl
  const size_t base = size_t(chunk) * KV_HEADS + g;
  const int qi = BLOCK * w + 2 * t;
#pragma unroll
  for (int d = 0; d < HEAD_DIM / 16; ++d) {
    float* at = ws_o + (base * NQ + qi) * HEAD_DIM + 16 * d + gl;  // query qi; the next query is 128 on
    at[0] = oacc[d][0];
    at[HEAD_DIM] = oacc[d][1];
    at[8] = oacc[d][2];
    at[HEAD_DIM + 8] = oacc[d][3];
  }
  if (gl == 0) {
    float* at = ws_ml + (base * NQ + qi) * 2;
    *reinterpret_cast<float4*>(at) = make_float4(m_run[0], l_run[0], m_run[1], l_run[1]);
  }
}

// grid KV_HEADS * NQ, block 128: thread c weighs chunk c, then every thread sums its dim over the chunks
__global__ void __launch_bounds__(HEAD_DIM) attn_combine_kernel(const float* ws_o, const float* ws_ml, int chunks,
                                                               __nv_bfloat16* out) {
  __shared__ float scratch[33];
  __shared__ float f[HEAD_DIM];
  pdl_trigger();
  pdl_wait();
  const int g = blockIdx.x / NQ, qi = blockIdx.x % NQ, d = threadIdx.x;
  float mc = -INFINITY, lc = 0.f;
  if (d < chunks) {
    const float2 ml = *reinterpret_cast<const float2*>(ws_ml + ((size_t(d) * KV_HEADS + g) * NQ + qi) * 2);
    mc = ml.x;
    lc = ml.y;
  }
  const float m = block_max(mc, scratch);
  const float fc = mc == -INFINITY ? 0.f : __expf(mc - m);
  f[d] = fc;
  const float l = block_sum(lc * fc, scratch);  // its barriers publish f
  float o = 0.f;
#pragma unroll 8
  for (int c = 0; c < chunks; ++c) o += ws_o[((size_t(c) * KV_HEADS + g) * NQ + qi) * HEAD_DIM + d] * f[c];
  const int hh = qi / BLOCK, r = qi % BLOCK, h = g * QH + hh;
  out[size_t(r) * Q_DIM + h * HEAD_DIM + d] = __float2bfloat16_rn(l > 0.f ? o / l : 0.f);
}

}  // namespace

std::size_t attention_workspace_floats(int nkeys) {
  return std::size_t(att_chunks(nkeys)) * KV_HEADS * NQ * (HEAD_DIM + 2);
}

void block_attention(const __nv_bfloat16* q, const void* k_cache, const void* v_cache, int nkeys,
                     const std::int32_t* ctx_pos, bool fp8, const DraftParams* p, float* ws, __nv_bfloat16* out,
                     cudaStream_t stream) {
  if (nkeys <= 0) throw std::invalid_argument("block_attention: nkeys must be positive");
  if (ctx_pos != nullptr && (nkeys != WINDOW + BLOCK || fp8))
    throw std::invalid_argument("block_attention: local layers attend to WINDOW + 8 BF16 keys");
  const int chunks = att_chunks(nkeys), chunk_keys = ATT_TILE * att_tiles_per_chunk(nkeys);
  float* ws_o = ws;
  float* ws_ml = ws + std::size_t(chunks) * KV_HEADS * NQ * HEAD_DIM;
  const dim3 grid(chunks, KV_HEADS), block(ATT_THREADS);
  const auto* k = static_cast<const uint8_t*>(k_cache);
  const auto* v = static_cast<const uint8_t*>(v_cache);
  if (ctx_pos != nullptr)
    launch_pdl(attn_partial_kernel<true, false>, grid, block, stream, q, k, v, nkeys, chunk_keys, ctx_pos, p, ws_o, ws_ml);
  else if (fp8)
    launch_pdl(attn_partial_kernel<false, true>, grid, block, stream, q, k, v, nkeys, chunk_keys, ctx_pos, p, ws_o, ws_ml);
  else
    launch_pdl(attn_partial_kernel<false, false>, grid, block, stream, q, k, v, nkeys, chunk_keys, ctx_pos, p, ws_o,
               ws_ml);
  launch_pdl(attn_combine_kernel, dim3(KV_HEADS * NQ), dim3(HEAD_DIM), stream, static_cast<const float*>(ws_o),
             static_cast<const float*>(ws_ml), chunks, out);
}

}  // namespace ingine::kernels::draft
