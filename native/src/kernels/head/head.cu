// Adapted from fngine (github.com/satellitedown/fngine, Apache-2.0) for ingine.
// Token selection kernels (see head.h). fngine's sampler keeps a top-20 candidate list; engine.Engine
// samples from the full top-k/top-p distribution, so this file keeps fngine's ordering key, argmax tie rule
// and splitmix counter RNG and replaces the candidate pipeline with exact radix selections over the row:
//   * top-k: the k-th largest z by a 4-pass MSB radix select (8-bit digits) over order-preserving keys;
//   * top-p: the boundary token by the same descent over the probability mass in 2^-40 fixed point
//     (integer sums, so the result does not depend on the summation order);
//   * the draw: Gumbel-max (argmax of z + G, G = -log E, E exponential) over the kept tokens, an exact sample of
//     the renormalised distribution without a prefix scan.
// Up to 8 rows every selection runs as multi-block passes (256 blocks per row, the last block of a row combines;
// top-k / top-p one launch per digit); more rows use one CTA of 1024 threads per row (every pass streams the row
// from L2). Both give the same draws.
#include "kernels/head/head.h"

#include "core/device.h"

#include <cuda_runtime.h>

#include <climits>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <math_constants.h>
#include <stdexcept>
#include <string>

namespace ingine::kernels::head {
namespace {

constexpr int kBlock = 1024;
constexpr int kWarps = kBlock / 32;
constexpr double kMassOne = 1099511627776.0;  // 2^40: fixed-point probability scale

// Order-preserving key of a finite float: larger float, larger key.
__device__ __forceinline__ std::uint32_t fkey(float x) {
  const std::uint32_t u = __float_as_uint(x);
  return (u & 0x80000000u) ? ~u : (u | 0x80000000u);
}

__host__ __device__ __forceinline__ unsigned long long splitmix64(unsigned long long x) {
  x += 0x9E3779B97F4A7C15ull;
  x = (x ^ (x >> 30)) * 0xBF58476D1CE4E5B9ull;
  x = (x ^ (x >> 27)) * 0x94D049BB133111EBull;
  return x ^ (x >> 31);
}

// 64 random bits for (seed, counter, row, index): a pure function of its inputs.
__device__ __forceinline__ unsigned long long random64(unsigned long long seed, unsigned long long counter,
                                                       std::uint32_t row, std::uint32_t index) {
  const unsigned long long k = splitmix64(seed ^ (counter * 0xD1B54A32D192ED03ull));
  return splitmix64(k ^ ((static_cast<unsigned long long>(row) << 32) | index));
}

// 24 random bits (uniform()).
__device__ __forceinline__ std::uint32_t random_bits(unsigned long long seed, unsigned long long counter,
                                                     std::uint32_t row, std::uint32_t index) {
  return static_cast<std::uint32_t>(random64(seed, counter, row, index) >> 40);
}

// Gumbel noise G = -log(E), E = -log(1 - v) a standard exponential, v uniform in (0, 1) from 64 random bits in
// floating point: the 24 bits from the leading one of k on (exact in float) and the leading-zero count as
// exponent, so v keeps 24 significant bits down to 2^-64 (v < 1 always; v > 0: k < 2^24 adds 1/2) and E (computed
// as -log1p(-v) for v < 1/2, -log(1 - v) above, both exact operations) resolves the extreme upper tail of G that
// decides draws of rare tokens: a token wins a Gumbel-max draw with its exact probability down to ratios of about
// e^-44 to the most likely one. (A 24-bit uniform resolves only e^-17: at a 200058-token vocabulary that
// under-samples the far tail, and its largest value rounded u to 1 in float, an infinite noise that made 1 draw
// in 84 a uniformly random token.)
__device__ __forceinline__ float gumbel(unsigned long long seed, unsigned long long counter, std::uint32_t row,
                                        std::uint32_t index) {
  const unsigned long long k = random64(seed, counter, row, index);
  const int z = min(__clzll(static_cast<long long>(k)), 40);
  const float m = static_cast<float>(static_cast<std::uint32_t>(k >> (40 - z))) + (z == 40 ? 0.5f : 0.0f);
  const float v = m * __int_as_float((127 - 24 - z) << 23);  // m * 2^-(24 + z)
  const float e = v < 0.5f ? -log1pf(-v) : -logf(1.0f - v);
  return -logf(e);
}

__device__ __forceinline__ bool better(float value, int index, float best_value, int best_index) {
  return value > best_value || (value == best_value && index < best_index);
}

struct Reduce {
  float f[kWarps];
  int i[kWarps];
  float bf;
  int bi;
};

// Block sum in a fixed order (deterministic). Contains two barriers.
__device__ float block_sum(float v, Reduce& r) {
  for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
  if ((threadIdx.x & 31) == 0) r.f[threadIdx.x >> 5] = v;
  __syncthreads();
  if (threadIdx.x == 0) {
    float s = 0.f;
    for (int w = 0; w < kWarps; ++w) s += r.f[w];
    r.bf = s;
  }
  __syncthreads();
  return r.bf;
}

__device__ float block_max(float v, Reduce& r) {
  for (int o = 16; o > 0; o >>= 1) v = fmaxf(v, __shfl_xor_sync(0xffffffffu, v, o));
  if ((threadIdx.x & 31) == 0) r.f[threadIdx.x >> 5] = v;
  __syncthreads();
  if (threadIdx.x == 0) {
    float s = r.f[0];
    for (int w = 1; w < kWarps; ++w) s = fmaxf(s, r.f[w]);
    r.bf = s;
  }
  __syncthreads();
  return r.bf;
}

// (value, index) argmax, ties to the lower index; returns the index (INT_MAX if every value was -inf/NaN).
__device__ int block_argmax(float v, int i, Reduce& r) {
  for (int o = 16; o > 0; o >>= 1) {
    const float ov = __shfl_xor_sync(0xffffffffu, v, o);
    const int oi = __shfl_xor_sync(0xffffffffu, i, o);
    if (better(ov, oi, v, i)) v = ov, i = oi;
  }
  if ((threadIdx.x & 31) == 0) r.f[threadIdx.x >> 5] = v, r.i[threadIdx.x >> 5] = i;
  __syncthreads();
  if (threadIdx.x == 0) {
    float bv = r.f[0];
    int bi = r.i[0];
    for (int w = 1; w < kWarps; ++w)
      if (better(r.f[w], r.i[w], bv, bi)) bv = r.f[w], bi = r.i[w];
    r.bi = bi;
  }
  __syncthreads();
  return r.bi;
}

__global__ void __launch_bounds__(kBlock) argmax_kernel(const float* __restrict__ logits, int V, long long stride,
                                                       std::int32_t* __restrict__ out) {
  __shared__ Reduce red;
  const float* row = logits + blockIdx.x * stride;
  float bv = -CUDART_INF_F;
  int bi = INT_MAX;
  for (int v = threadIdx.x; v < V; v += kBlock) {
    const float x = row[v];
    if (better(x, v, bv, bi)) bv = x, bi = v;
  }
  const int best = block_argmax(bv, bi, red);
  if (threadIdx.x == 0) out[blockIdx.x] = best == INT_MAX ? 0 : best;
}

struct Select {
  std::uint32_t hist[256];
  unsigned long long mass[256];
  std::uint32_t prefix;
  int nocut;
};

// Key of the k-th largest (1-based) z = row * inv_t over the row.
__device__ std::uint32_t kth_key(const float* row, int V, float inv_t, int k, Select& s) {
  std::uint32_t prefix = 0, mask = 0;
  int remaining = k;
  for (int shift = 24; shift >= 0; shift -= 8) {
    for (int b = threadIdx.x; b < 256; b += kBlock) s.hist[b] = 0;
    __syncthreads();
    for (int v = threadIdx.x; v < V; v += kBlock) {
      const std::uint32_t key = fkey(row[v] * inv_t);
      if ((key & mask) == prefix) atomicAdd(&s.hist[(key >> shift) & 255u], 1u);
    }
    __syncthreads();
    if (threadIdx.x == 0) {
      int b = 255;
      for (; b > 0; --b) {
        if (static_cast<int>(s.hist[b]) >= remaining) break;
        remaining -= static_cast<int>(s.hist[b]);
      }
      s.prefix = prefix | (static_cast<std::uint32_t>(b) << shift);
    }
    __syncthreads();
    prefix = s.prefix;
    mask |= 0xffu << shift;
    // every thread has read s.prefix before thread 0 rewrites it (after the next two barriers)
  }
  return prefix;
}

// Fixed-point probability of a token (0 outside the top-k set).
__device__ __forceinline__ unsigned long long fixed_mass(float p) {
  return static_cast<unsigned long long>(static_cast<double>(p) * kMassOne);
}

// Key of the last token top-p keeps: the first one (descending) whose inclusive mass exceeds `limit`.
// Sets s.nocut when no token does (everything is kept).
__device__ std::uint32_t top_p_key(const float* row, int V, float inv_t, std::uint32_t kth, float zmax, float inv_sum,
                                   unsigned long long limit, Select& s) {
  std::uint32_t prefix = 0, mask = 0;
  unsigned long long above = 0;
  if (threadIdx.x == 0) s.nocut = 0;
  for (int shift = 24; shift >= 0; shift -= 8) {
    for (int b = threadIdx.x; b < 256; b += kBlock) s.mass[b] = 0;
    __syncthreads();
    for (int v = threadIdx.x; v < V; v += kBlock) {
      const float z = row[v] * inv_t;
      const std::uint32_t key = fkey(z);
      if (key >= kth && (key & mask) == prefix) {
        const unsigned long long q = fixed_mass(expf(z - zmax) * inv_sum);
        if (q) atomicAdd(&s.mass[(key >> shift) & 255u], q);
      }
    }
    __syncthreads();
    if (threadIdx.x == 0) {
      int b = 255;
      bool hit = false;
      for (; b >= 0; --b) {
        if (above + s.mass[b] > limit) {
          hit = true;
          break;
        }
        above += s.mass[b];
      }
      if (!hit) {
        s.nocut = 1;
        b = 0;
      }
      s.prefix = prefix | (static_cast<std::uint32_t>(b) << shift);
    }
    __syncthreads();
    if (s.nocut) return kth;  // uniform across the block
    prefix = s.prefix;
    mask |= 0xffu << shift;
  }
  return prefix;
}

// MODE 0: one Gumbel-max draw per row into out; MODE 1: the distribution into probs [rows, V].
template <int MODE>
__global__ void __launch_bounds__(kBlock) dist_kernel(const float* __restrict__ logits, int V, long long stride,
                                                     float inv_t, int top_k, float top_p, unsigned long long seed,
                                                     unsigned long long counter, std::int32_t* __restrict__ out,
                                                     float* __restrict__ probs) {
  __shared__ Reduce red;
  __shared__ Select sel;
  const int r = blockIdx.x;
  const float* row = logits + r * stride;

  const std::uint32_t kth = (top_k > 0 && top_k < V) ? kth_key(row, V, inv_t, top_k, sel) : 0u;

  float mx = -CUDART_INF_F;
  for (int v = threadIdx.x; v < V; v += kBlock) mx = fmaxf(mx, row[v] * inv_t);
  const float zmax = block_max(mx, red);

  float s = 0.f;
  for (int v = threadIdx.x; v < V; v += kBlock) {
    const float z = row[v] * inv_t;
    if (fkey(z) >= kth) s += expf(z - zmax);
  }
  const float sum = block_sum(s, red);
  const float inv_sum = 1.f / sum;

  std::uint32_t thr = kth;
  bool cut = false;
  if (top_p < 1.f) {
    const unsigned long long limit = static_cast<unsigned long long>(static_cast<double>(top_p) * kMassOne);
    thr = top_p_key(row, V, inv_t, kth, zmax, inv_sum, limit, sel);
    cut = !sel.nocut;
  }

  if constexpr (MODE == 0) {
    float bv = -CUDART_INF_F;
    int bi = INT_MAX;
    for (int v = threadIdx.x; v < V; v += kBlock) {
      const float z = row[v] * inv_t;
      if (fkey(z) >= thr) {
        const float score = z + gumbel(seed, counter, static_cast<std::uint32_t>(r), static_cast<std::uint32_t>(v));
        if (better(score, v, bv, bi)) bv = score, bi = v;
      }
    }
    const int best = block_argmax(bv, bi, red);
    if (threadIdx.x == 0) out[r] = best == INT_MAX ? 0 : best;
  } else {
    float* pr = probs + static_cast<long long>(r) * V;
    if (cut) {
      float k = 0.f;
      for (int v = threadIdx.x; v < V; v += kBlock) {
        const float z = row[v] * inv_t;
        if (fkey(z) >= thr) k += expf(z - zmax) / sum;
      }
      const float kept = block_sum(k, red);
      for (int v = threadIdx.x; v < V; v += kBlock) {
        const float z = row[v] * inv_t;
        pr[v] = fkey(z) >= thr ? (expf(z - zmax) / sum) / kept : 0.f;
      }
    } else {
      for (int v = threadIdx.x; v < V; v += kBlock) {
        const float z = row[v] * inv_t;
        pr[v] = fkey(z) >= kth ? expf(z - zmax) / sum : 0.f;
      }
    }
  }
}

__global__ void __launch_bounds__(kBlock) multinomial_kernel(const float* __restrict__ probs, int V, long long stride,
                                                            unsigned long long seed, unsigned long long counter,
                                                            std::int32_t* __restrict__ out) {
  __shared__ Reduce red;
  const int r = blockIdx.x;
  const float* row = probs + r * stride;
  float bv = -CUDART_INF_F;
  int bi = INT_MAX;
  for (int v = threadIdx.x; v < V; v += kBlock) {
    const float p = row[v];
    if (p > 0.f) {
      const float score = logf(p) + gumbel(seed, counter, static_cast<std::uint32_t>(r), static_cast<std::uint32_t>(v));
      if (better(score, v, bv, bi)) bv = score, bi = v;
    }
  }
  const int best = block_argmax(bv, bi, red);
  if (threadIdx.x == 0) out[r] = best == INT_MAX ? 0 : best;
}

__global__ void uniform_kernel(int n, unsigned long long seed, unsigned long long counter, float* __restrict__ out) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) out[i] = static_cast<float>(random_bits(seed, counter, 0xffffffffu, static_cast<std::uint32_t>(i))) *
                      (1.0f / 16777216.0f);
}

// ---- multi-block passes (rows <= kFastRows): kParts blocks per row each reduce a contiguous slice; the last
// block of a row to finish (atomic counter, reset by it) combines the slice results in slice order. The
// result equals the one-CTA kernels' exactly for the selections (max score, lowest index on ties) and up to
// the summation order for the softmax sum.
constexpr int kFast = 256;
constexpr int kFastWarps = kFast / 32;
constexpr int kParts = 256;  // blocks per row (= kFast: the last block combines one part per thread)
constexpr int kFastRows = 8;

struct SelectPartials {
  float v[kFastRows][kParts];
  int i[kFastRows][kParts];
  unsigned done[kFastRows];
};
struct SoftmaxPartials {
  float m[kFastRows][kParts];  // slice max of z
  float s[kFastRows][kParts];  // slice sum of exp(z - m)
  float zmax[kFastRows], sum[kFastRows];
  unsigned done[kFastRows];
};
// one set per entry point (argmax, sample, sample_from_probs; sampling_probs)
__device__ SelectPartials g_select[3];
__device__ SoftmaxPartials g_softmax;

enum class Score { Logit, GumbelLogit, GumbelProb };

struct FastReduce {
  float f[kFastWarps];
  int i[kFastWarps];
  bool last;
};

// (value, index) argmax over the block, ties to the lower index; result in thread 0.
__device__ __forceinline__ void fast_argmax(float& v, int& i, FastReduce& r) {
  for (int o = 16; o > 0; o >>= 1) {
    const float ov = __shfl_xor_sync(0xffffffffu, v, o);
    const int oi = __shfl_xor_sync(0xffffffffu, i, o);
    if (better(ov, oi, v, i)) v = ov, i = oi;
  }
  if ((threadIdx.x & 31) == 0) r.f[threadIdx.x >> 5] = v, r.i[threadIdx.x >> 5] = i;
  __syncthreads();
  if (threadIdx.x == 0)
    for (int w = 1; w < kFastWarps; ++w)
      if (better(r.f[w], r.i[w], v, i)) v = r.f[w], i = r.i[w];
}

// True in every thread of the block that finished its row last (after publishing its own part).
__device__ __forceinline__ bool finish_part(unsigned* done, FastReduce& r) {
  if (threadIdx.x == 0) {
    __threadfence();
    r.last = atomicAdd(done, 1u) == kParts - 1;
  }
  __syncthreads();
  if (!r.last) return false;
  __threadfence();
  return true;
}

// Block-wide max / sum over kFast threads (fixed order: deterministic); every thread gets the result.
__device__ __forceinline__ float fast_max(float v, FastReduce& r) {
  for (int o = 16; o > 0; o >>= 1) v = fmaxf(v, __shfl_xor_sync(0xffffffffu, v, o));
  __syncthreads();
  if ((threadIdx.x & 31) == 0) r.f[threadIdx.x >> 5] = v;
  __syncthreads();
  v = r.f[0];
  for (int w = 1; w < kFastWarps; ++w) v = fmaxf(v, r.f[w]);
  return v;
}
__device__ __forceinline__ float fast_sum(float v, FastReduce& r) {
  for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
  __syncthreads();
  if ((threadIdx.x & 31) == 0) r.f[threadIdx.x >> 5] = v;
  __syncthreads();
  v = 0.f;
  for (int w = 0; w < kFastWarps; ++w) v += r.f[w];
  return v;
}

// thr (optional, per row): only tokens whose key fkey(z) >= thr[r] take part (top-k / top-p truncation)
template <Score S>
__global__ void __launch_bounds__(kFast) select_kernel(const float* __restrict__ in, int V, long long stride, float inv_t,
                                                       unsigned long long seed, unsigned long long counter,
                                                       SelectPartials* __restrict__ part, std::int32_t* __restrict__ out,
                                                       const std::uint32_t* __restrict__ thr) {
  __shared__ FastReduce red;
  const int r = blockIdx.y, b = blockIdx.x;
  const float* row = in + r * stride;
  const int chunk = (V + kParts - 1) / kParts;
  const int v1 = min(V, (b + 1) * chunk);
  float bv = -CUDART_INF_F;
  int bi = INT_MAX;
  for (int v = b * chunk + threadIdx.x; v < v1; v += kFast) {
    float s;
    if constexpr (S == Score::Logit) {
      s = row[v];
    } else if constexpr (S == Score::GumbelLogit) {
      const float z = __fmul_rn(row[v], inv_t);
      if (thr && fkey(z) < thr[r]) continue;
      s = __fadd_rn(z, gumbel(seed, counter, static_cast<std::uint32_t>(r), static_cast<std::uint32_t>(v)));
    } else {
      const float p = row[v];
      if (!(p > 0.f)) continue;
      s = __fadd_rn(logf(p), gumbel(seed, counter, static_cast<std::uint32_t>(r), static_cast<std::uint32_t>(v)));
    }
    if (better(s, v, bv, bi)) bv = s, bi = v;
  }
  fast_argmax(bv, bi, red);
  if (threadIdx.x == 0) part->v[r][b] = bv, part->i[r][b] = bi;
  if (!finish_part(&part->done[r], red)) return;
  bv = -CUDART_INF_F;
  bi = INT_MAX;
  if (threadIdx.x < kParts) {
    bv = const_cast<volatile float&>(part->v[r][threadIdx.x]);
    bi = const_cast<volatile int&>(part->i[r][threadIdx.x]);
  }
  fast_argmax(bv, bi, red);
  if (threadIdx.x == 0) {
    out[r] = bi == INT_MAX ? 0 : bi;
    part->done[r] = 0;
  }
}

// softmax statistics of z = row * inv_t per row: zmax and sum exp(z - zmax) into g_softmax
__global__ void __launch_bounds__(kFast) softmax_stats_kernel(const float* __restrict__ in, int V, long long stride,
                                                              float inv_t, SoftmaxPartials* __restrict__ part) {
  __shared__ FastReduce red;
  __shared__ float bmax;
  const int r = blockIdx.y, b = blockIdx.x;
  const float* row = in + r * stride;
  const int chunk = (V + kParts - 1) / kParts;
  const int v0 = b * chunk, v1 = min(V, (b + 1) * chunk);
  float m = -CUDART_INF_F;
  for (int v = v0 + threadIdx.x; v < v1; v += kFast) m = fmaxf(m, __fmul_rn(row[v], inv_t));
  for (int o = 16; o > 0; o >>= 1) m = fmaxf(m, __shfl_xor_sync(0xffffffffu, m, o));
  if ((threadIdx.x & 31) == 0) red.f[threadIdx.x >> 5] = m;
  __syncthreads();
  if (threadIdx.x == 0) {
    float x = red.f[0];
    for (int w = 1; w < kFastWarps; ++w) x = fmaxf(x, red.f[w]);
    bmax = x;
  }
  __syncthreads();
  m = bmax;
  float s = 0.f;
  if (m > -CUDART_INF_F)
    for (int v = v0 + threadIdx.x; v < v1; v += kFast) s += expf(__fmul_rn(row[v], inv_t) - m);
  for (int o = 16; o > 0; o >>= 1) s += __shfl_xor_sync(0xffffffffu, s, o);
  __syncthreads();
  if ((threadIdx.x & 31) == 0) red.f[threadIdx.x >> 5] = s;
  __syncthreads();
  if (threadIdx.x == 0) {
    float x = 0.f;
    for (int w = 0; w < kFastWarps; ++w) x += red.f[w];
    part->m[r][b] = m;
    part->s[r][b] = x;
  }
  if (!finish_part(&part->done[r], red)) return;
  // one part per thread (kParts == kFast), combined by block reductions
  const float mk = const_cast<volatile float&>(part->m[r][threadIdx.x]);
  const float sk = const_cast<volatile float&>(part->s[r][threadIdx.x]);
  const float zmax = fast_max(mk, red);
  const float sum = fast_sum(mk > -CUDART_INF_F ? sk * expf(mk - zmax) : 0.f, red);
  if (threadIdx.x == 0) {
    part->zmax[r] = zmax;
    part->sum[r] = sum;
    part->done[r] = 0;
  }
}

__global__ void __launch_bounds__(kFast) softmax_write_kernel(const float* __restrict__ in, int V, long long stride,
                                                              float inv_t, const SoftmaxPartials* __restrict__ part,
                                                              float* __restrict__ probs) {
  const int r = blockIdx.y, b = blockIdx.x;
  const float* row = in + r * stride;
  float* pr = probs + static_cast<long long>(r) * V;
  const float zmax = part->zmax[r], sum = part->sum[r];
  const int chunk = (V + kParts - 1) / kParts;
  const int v1 = min(V, (b + 1) * chunk);
  for (int v = b * chunk + threadIdx.x; v < v1; v += kFast) pr[v] = expf(__fmul_rn(row[v], inv_t) - zmax) / sum;
}

// ---- multi-block top-k / top-p: dist_kernel's selections pass by pass (one launch per 8-bit digit, the last
// block of a row decides it), so the k-th key and the top-p boundary are the one-CTA kernel's (the top-p mass is
// summed in fixed point, independent of the order; only the softmax sum's rounding may differ).
struct TruncState {
  std::uint32_t hist[kFastRows][256];
  unsigned long long mass[kFastRows][256];
  std::uint32_t prefix[kFastRows];
  int remaining[kFastRows];
  unsigned long long above[kFastRows];
  std::uint32_t kth[kFastRows], thr[kFastRows];
  int nocut[kFastRows];
  float zmax[kFastRows], sum[kFastRows], kept[kFastRows];
  float m[kFastRows][kParts], s[kFastRows][kParts];
  unsigned done[kFastRows];
};
__device__ TruncState g_trunc[2];  // sample, sampling_probs

// key bits above the digit at `shift` (the digits decided by earlier passes)
__device__ __forceinline__ std::uint32_t pass_mask(int shift) { return shift == 24 ? 0u : ~((1u << (shift + 8)) - 1u); }

__global__ void trunc_init_kernel(int rows, int top_k, TruncState* __restrict__ st) {
  const int r = threadIdx.x;
  if (r >= rows) return;
  st->prefix[r] = 0;
  st->remaining[r] = top_k;
  st->above[r] = 0;
  st->kth[r] = st->thr[r] = 0;
  st->nocut[r] = 0;
  st->kept[r] = 1.f;
}

// one digit of the k-th largest key (kth_key)
__global__ void __launch_bounds__(kFast) kth_pass_kernel(const float* __restrict__ in, int V, long long stride,
                                                         float inv_t, int shift, TruncState* __restrict__ st) {
  __shared__ FastReduce red;
  __shared__ unsigned h[256];
  const int r = blockIdx.y, b = blockIdx.x;
  const float* row = in + r * stride;
  for (int i = threadIdx.x; i < 256; i += kFast) h[i] = 0;
  __syncthreads();
  const std::uint32_t prefix = st->prefix[r], mask = pass_mask(shift);
  const int chunk = (V + kParts - 1) / kParts;
  const int v1 = min(V, (b + 1) * chunk);
  // most keys share the digit (similar magnitudes): accumulate runs per thread, one atomic per change of digit
  std::uint32_t run_bin = 256, run = 0;
  for (int v = b * chunk + threadIdx.x; v < v1; v += kFast) {
    const std::uint32_t key = fkey(__fmul_rn(row[v], inv_t));
    if ((key & mask) != prefix) continue;
    const std::uint32_t bin = (key >> shift) & 255u;
    if (bin != run_bin) {
      if (run) atomicAdd(&h[run_bin], run);
      run_bin = bin, run = 0;
    }
    ++run;
  }
  if (run) atomicAdd(&h[run_bin], run);
  __syncthreads();
  for (int i = threadIdx.x; i < 256; i += kFast)
    if (h[i]) atomicAdd(&st->hist[r][i], h[i]);
  if (!finish_part(&st->done[r], red)) return;
  // the row's histogram into shared memory in parallel (and cleared for the next pass), then the descent
  for (int i = threadIdx.x; i < 256; i += kFast) {
    h[i] = const_cast<volatile std::uint32_t&>(st->hist[r][i]);
    st->hist[r][i] = 0;
  }
  __syncthreads();
  if (threadIdx.x == 0) {
    int rem = st->remaining[r];
    int d = 255;
    for (; d > 0; --d) {
      const int c = int(h[d]);
      if (c >= rem) break;
      rem -= c;
    }
    const std::uint32_t p = prefix | (static_cast<std::uint32_t>(d) << shift);
    st->remaining[r] = rem;
    st->prefix[r] = p;
    if (shift == 0) st->kth[r] = st->thr[r] = p;
    st->done[r] = 0;
  }
}

// zmax over the row, sum of exp(z - zmax) over keys >= kth; resets the prefix for the top-p passes
__global__ void __launch_bounds__(kFast) trunc_stats_kernel(const float* __restrict__ in, int V, long long stride,
                                                            float inv_t, TruncState* __restrict__ st) {
  __shared__ FastReduce red;
  __shared__ float bmax;
  const int r = blockIdx.y, b = blockIdx.x;
  const float* row = in + r * stride;
  const std::uint32_t kth = st->kth[r];
  const int chunk = (V + kParts - 1) / kParts;
  const int v0 = b * chunk, v1 = min(V, (b + 1) * chunk);
  float m = -CUDART_INF_F;
  for (int v = v0 + threadIdx.x; v < v1; v += kFast) m = fmaxf(m, __fmul_rn(row[v], inv_t));
  for (int o = 16; o > 0; o >>= 1) m = fmaxf(m, __shfl_xor_sync(0xffffffffu, m, o));
  if ((threadIdx.x & 31) == 0) red.f[threadIdx.x >> 5] = m;
  __syncthreads();
  if (threadIdx.x == 0) {
    float x = red.f[0];
    for (int w = 1; w < kFastWarps; ++w) x = fmaxf(x, red.f[w]);
    bmax = x;
  }
  __syncthreads();
  m = bmax;
  float s = 0.f;
  if (m > -CUDART_INF_F)
    for (int v = v0 + threadIdx.x; v < v1; v += kFast) {
      const float z = __fmul_rn(row[v], inv_t);
      if (fkey(z) >= kth) s += expf(z - m);
    }
  for (int o = 16; o > 0; o >>= 1) s += __shfl_xor_sync(0xffffffffu, s, o);
  __syncthreads();
  if ((threadIdx.x & 31) == 0) red.f[threadIdx.x >> 5] = s;
  __syncthreads();
  if (threadIdx.x == 0) {
    float x = 0.f;
    for (int w = 0; w < kFastWarps; ++w) x += red.f[w];
    st->m[r][b] = m;
    st->s[r][b] = x;
  }
  if (!finish_part(&st->done[r], red)) return;
  const float mk = const_cast<volatile float&>(st->m[r][threadIdx.x]);
  const float sk = const_cast<volatile float&>(st->s[r][threadIdx.x]);
  const float zmax = fast_max(mk, red);
  const float sum = fast_sum(mk > -CUDART_INF_F ? sk * expf(mk - zmax) : 0.f, red);
  if (threadIdx.x == 0) {
    st->zmax[r] = zmax;
    st->sum[r] = sum;
    st->prefix[r] = 0;
    st->done[r] = 0;
  }
}

// one digit of the top-p boundary key (top_p_key): the first key, descending, whose inclusive mass exceeds limit
__global__ void __launch_bounds__(kFast) mass_pass_kernel(const float* __restrict__ in, int V, long long stride,
                                                          float inv_t, int shift, unsigned long long limit,
                                                          TruncState* __restrict__ st) {
  __shared__ FastReduce red;
  __shared__ unsigned long long h[256];
  const int r = blockIdx.y, b = blockIdx.x;
  if (st->nocut[r]) return;  // decided by an earlier pass: everything kept
  const float* row = in + r * stride;
  for (int i = threadIdx.x; i < 256; i += kFast) h[i] = 0;
  __syncthreads();
  const std::uint32_t prefix = st->prefix[r], mask = pass_mask(shift), kth = st->kth[r];
  const float zmax = st->zmax[r], inv_sum = 1.f / st->sum[r];
  const int chunk = (V + kParts - 1) / kParts;
  const int v1 = min(V, (b + 1) * chunk);
  std::uint32_t run_bin = 256;
  unsigned long long run = 0;
  for (int v = b * chunk + threadIdx.x; v < v1; v += kFast) {
    const float z = __fmul_rn(row[v], inv_t);
    const std::uint32_t key = fkey(z);
    if (key < kth || (key & mask) != prefix) continue;
    const unsigned long long q = fixed_mass(expf(z - zmax) * inv_sum);
    if (!q) continue;
    const std::uint32_t bin = (key >> shift) & 255u;
    if (bin != run_bin) {
      if (run) atomicAdd(&h[run_bin], run);
      run_bin = bin, run = 0;
    }
    run += q;
  }
  if (run) atomicAdd(&h[run_bin], run);
  __syncthreads();
  for (int i = threadIdx.x; i < 256; i += kFast)
    if (h[i]) atomicAdd(&st->mass[r][i], h[i]);
  if (!finish_part(&st->done[r], red)) return;
  for (int i = threadIdx.x; i < 256; i += kFast) {
    h[i] = const_cast<volatile unsigned long long&>(st->mass[r][i]);
    st->mass[r][i] = 0;
  }
  __syncthreads();
  if (threadIdx.x == 0) {
    unsigned long long above = st->above[r];
    int d = 255;
    bool hit = false;
    for (; d >= 0; --d) {
      if (above + h[d] > limit) {
        hit = true;
        break;
      }
      above += h[d];
    }
    st->above[r] = above;
    if (!hit) {
      st->nocut[r] = 1;
      st->thr[r] = kth;
    } else {
      const std::uint32_t p = prefix | (static_cast<std::uint32_t>(d) << shift);
      st->prefix[r] = p;
      if (shift == 0) st->thr[r] = p;
    }
    st->done[r] = 0;
  }
}

// kept = sum over keys >= thr of exp(z - zmax) / sum, when top-p cut the row (else 1)
__global__ void __launch_bounds__(kFast) trunc_kept_kernel(const float* __restrict__ in, int V, long long stride,
                                                           float inv_t, TruncState* __restrict__ st) {
  __shared__ FastReduce red;
  const int r = blockIdx.y, b = blockIdx.x;
  if (st->nocut[r]) return;
  const float* row = in + r * stride;
  const std::uint32_t thr = st->thr[r];
  const float zmax = st->zmax[r], sum = st->sum[r];
  const int chunk = (V + kParts - 1) / kParts;
  const int v1 = min(V, (b + 1) * chunk);
  float k = 0.f;
  for (int v = b * chunk + threadIdx.x; v < v1; v += kFast) {
    const float z = __fmul_rn(row[v], inv_t);
    if (fkey(z) >= thr) k += expf(z - zmax) / sum;
  }
  for (int o = 16; o > 0; o >>= 1) k += __shfl_xor_sync(0xffffffffu, k, o);
  if ((threadIdx.x & 31) == 0) red.f[threadIdx.x >> 5] = k;
  __syncthreads();
  if (threadIdx.x == 0) {
    float x = 0.f;
    for (int w = 0; w < kFastWarps; ++w) x += red.f[w];
    st->s[r][b] = x;
  }
  if (!finish_part(&st->done[r], red)) return;
  const float kept = fast_sum(const_cast<volatile float&>(st->s[r][threadIdx.x]), red);
  if (threadIdx.x == 0) {
    st->kept[r] = kept;
    st->done[r] = 0;
  }
}

__global__ void __launch_bounds__(kFast) trunc_write_kernel(const float* __restrict__ in, int V, long long stride,
                                                            float inv_t, const TruncState* __restrict__ st,
                                                            float* __restrict__ probs) {
  const int r = blockIdx.y, b = blockIdx.x;
  const float* row = in + r * stride;
  float* pr = probs + static_cast<long long>(r) * V;
  const std::uint32_t thr = st->thr[r];
  const float zmax = st->zmax[r], sum = st->sum[r], kept = st->kept[r];
  const int chunk = (V + kParts - 1) / kParts;
  const int v1 = min(V, (b + 1) * chunk);
  for (int v = b * chunk + threadIdx.x; v < v1; v += kFast) {
    const float z = __fmul_rn(row[v], inv_t);
    pr[v] = fkey(z) >= thr ? (expf(z - zmax) / sum) / kept : 0.f;
  }
}

template <class T>
T* symbol(const T& sym) {
  void* p = nullptr;
  CUDA_CHECK(cudaGetSymbolAddress(&p, sym));
  return static_cast<T*>(p);
}

template <Score S>
void launch_select(const float* in, int rows, int V, std::int64_t stride, float inv_t, unsigned long long seed,
                   unsigned long long counter, int entry, std::int32_t* out, cudaStream_t stream,
                   const std::uint32_t* thr = nullptr) {
  static SelectPartials* const part = symbol(g_select[0]);
  select_kernel<S><<<dim3(kParts, rows), kFast, 0, stream>>>(in, V, stride, inv_t, seed, counter, part + entry, out,
                                                              thr);
  CUDA_CHECK(cudaGetLastError());
}

void launch_softmax(const float* in, int rows, int V, std::int64_t stride, float inv_t, float* probs,
                    cudaStream_t stream) {
  static SoftmaxPartials* const part = symbol(g_softmax);
  softmax_stats_kernel<<<dim3(kParts, rows), kFast, 0, stream>>>(in, V, stride, inv_t, part);
  CUDA_CHECK(cudaGetLastError());
  softmax_write_kernel<<<dim3(kParts, rows), kFast, 0, stream>>>(in, V, stride, inv_t, part, probs);
  CUDA_CHECK(cudaGetLastError());
}

bool top_k_active(const SamplingParams& p, int V) { return p.top_k > 0 && p.top_k < V; }
// the multi-block paths: up to kFastRows rows; sampling without top-k / top-p
bool plain_sampling(const SamplingParams& p, int V) { return !top_k_active(p, V) && p.top_p >= 1.f; }

// The top-k / top-p selections of `rows` rows into g_trunc[entry]; returns it (thr per row: the lowest key kept).
TruncState* launch_truncation(const float* in, int rows, int V, std::int64_t stride, float inv_t,
                              const SamplingParams& p, int entry, cudaStream_t stream) {
  static TruncState* const states = symbol(g_trunc[0]);
  TruncState* st = states + entry;
  const dim3 grid(kParts, rows);
  trunc_init_kernel<<<1, 32, 0, stream>>>(rows, top_k_active(p, V) ? p.top_k : 0, st);
  if (top_k_active(p, V))
    for (int shift = 24; shift >= 0; shift -= 8)
      kth_pass_kernel<<<grid, kFast, 0, stream>>>(in, V, stride, inv_t, shift, st);
  trunc_stats_kernel<<<grid, kFast, 0, stream>>>(in, V, stride, inv_t, st);
  if (p.top_p < 1.f) {
    const unsigned long long limit = static_cast<unsigned long long>(static_cast<double>(p.top_p) * kMassOne);
    for (int shift = 24; shift >= 0; shift -= 8)
      mass_pass_kernel<<<grid, kFast, 0, stream>>>(in, V, stride, inv_t, shift, limit, st);
  }
  CUDA_CHECK(cudaGetLastError());
  return st;
}

void require(bool ok, const char* what) {
  if (!ok) throw std::invalid_argument(std::string("head: ") + what);
}

void check_rows(const void* p, int rows, int V, std::int64_t stride) {
  require(p != nullptr && rows >= 0 && V > 0 && stride >= V, "bad logits/probs arguments");
}

}  // namespace

void argmax(const float* logits, int rows, int V, std::int64_t stride, std::int32_t* out, cudaStream_t stream) {
  check_rows(logits, rows, V, stride);
  if (rows == 0) return;
  if (rows <= kFastRows)
    launch_select<Score::Logit>(logits, rows, V, stride, 1.f, 0, 0, 0, out, stream);
  else
    argmax_kernel<<<rows, kBlock, 0, stream>>>(logits, V, stride, out);
  CUDA_CHECK(cudaGetLastError());
}

void sampling_probs(const float* logits, int rows, int V, std::int64_t stride, const SamplingParams& params,
                    float* probs, cudaStream_t stream) {
  check_rows(logits, rows, V, stride);
  require(params.temperature > 0.f, "sampling_probs needs temperature > 0");
  if (rows == 0) return;
  const float inv_t = 1.0f / params.temperature;
  if (rows <= kFastRows && plain_sampling(params, V)) {
    launch_softmax(logits, rows, V, stride, inv_t, probs, stream);
    return;
  }
  if (rows <= kFastRows) {
    TruncState* st = launch_truncation(logits, rows, V, stride, inv_t, params, 1, stream);
    const dim3 grid(kParts, rows);
    if (params.top_p < 1.f) trunc_kept_kernel<<<grid, kFast, 0, stream>>>(logits, V, stride, inv_t, st);
    trunc_write_kernel<<<grid, kFast, 0, stream>>>(logits, V, stride, inv_t, st, probs);
    CUDA_CHECK(cudaGetLastError());
    return;
  }
  dist_kernel<1><<<rows, kBlock, 0, stream>>>(logits, V, stride, inv_t, params.top_k, params.top_p, 0ull, 0ull,
                                              nullptr, probs);
  CUDA_CHECK(cudaGetLastError());
}

void sample(const float* logits, int rows, int V, std::int64_t stride, const SamplingParams& params, SamplerRng& rng,
            std::int32_t* out, cudaStream_t stream) {
  check_rows(logits, rows, V, stride);
  if (rows == 0) return;
  if (params.temperature <= 0.f) {
    argmax(logits, rows, V, stride, out, stream);
    return;
  }
  const float inv_t = 1.0f / params.temperature;
  if (rows <= kFastRows && plain_sampling(params, V)) {
    launch_select<Score::GumbelLogit>(logits, rows, V, stride, inv_t, rng.seed, rng.counter++, 1, out, stream);
    return;
  }
  if (rows <= kFastRows) {
    const TruncState* st = launch_truncation(logits, rows, V, stride, inv_t, params, 0, stream);
    const auto* thr = reinterpret_cast<const std::uint32_t*>(reinterpret_cast<const char*>(st) + offsetof(TruncState, thr));
    launch_select<Score::GumbelLogit>(logits, rows, V, stride, inv_t, rng.seed, rng.counter++, 1, out, stream, thr);
    return;
  }
  dist_kernel<0><<<rows, kBlock, 0, stream>>>(logits, V, stride, inv_t, params.top_k, params.top_p, rng.seed,
                                              rng.counter++, out, nullptr);
  CUDA_CHECK(cudaGetLastError());
}

void sample_from_probs(const float* probs, int rows, int V, std::int64_t stride, SamplerRng& rng, std::int32_t* out,
                       cudaStream_t stream) {
  check_rows(probs, rows, V, stride);
  if (rows == 0) return;
  if (rows <= kFastRows)
    launch_select<Score::GumbelProb>(probs, rows, V, stride, 1.f, rng.seed, rng.counter++, 2, out, stream);
  else
    multinomial_kernel<<<rows, kBlock, 0, stream>>>(probs, V, stride, rng.seed, rng.counter++, out);
  CUDA_CHECK(cudaGetLastError());
}

void uniform(int n, SamplerRng& rng, float* out, cudaStream_t stream) {
  require(n >= 0 && (n == 0 || out != nullptr), "bad uniform arguments");
  if (n == 0) return;
  uniform_kernel<<<(n + 255) / 256, 256, 0, stream>>>(n, rng.seed, rng.counter++, out);
  CUDA_CHECK(cudaGetLastError());
}

}  // namespace ingine::kernels::head
