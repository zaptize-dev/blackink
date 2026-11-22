// Counter-based RNG of the draft kernels (host and device): Philox4x32-10 (Salmon et al. 2011).
#pragma once

#include <cuda_runtime.h>

#include <algorithm>
#include <bit>
#include <cmath>
#include <cstdint>

namespace ingine::kernels::draft {

struct Philox {
  static __host__ __device__ __forceinline__ void round(std::uint32_t (&c)[4], const std::uint32_t (&k)[2]) {
    const std::uint64_t p0 = static_cast<std::uint64_t>(0xD2511F53u) * c[0];
    const std::uint64_t p1 = static_cast<std::uint64_t>(0xCD9E8D57u) * c[2];
    const std::uint32_t hi0 = static_cast<std::uint32_t>(p0 >> 32), lo0 = static_cast<std::uint32_t>(p0);
    const std::uint32_t hi1 = static_cast<std::uint32_t>(p1 >> 32), lo1 = static_cast<std::uint32_t>(p1);
    c[0] = hi1 ^ c[1] ^ k[0];
    c[1] = lo1;
    c[2] = hi0 ^ c[3] ^ k[1];
    c[3] = lo0;
  }
  static __host__ __device__ __forceinline__ void generate(std::uint32_t (&c)[4], std::uint64_t seed) {
    std::uint32_t k[2] = {static_cast<std::uint32_t>(seed), static_cast<std::uint32_t>(seed >> 32)};
#pragma unroll
    for (int r = 0; r < 10; ++r) {
      round(c, k);
      k[0] += 0x9E3779B9u;
      k[1] += 0xBB67AE85u;
    }
  }
};

// Gumbel noise -log(E), E = -log(1 - v) ~ Exp(1), from 64 random bits k: v = k / 2^64 kept to 24 significant
// bits below its leading zeros (down to 2^-64; the last 24 bits get + 0.5), so v is never 0 or 1 and the noise
// is not capped (it reaches about 44): the drafts follow softmax(logits / T) to float precision. The same
// construction as gumbel() in kernels/head/head.cu.
__host__ __device__ __forceinline__ float gumbel_from_bits(std::uint64_t k) {
#ifdef __CUDA_ARCH__
  const int z = min(__clzll(static_cast<long long>(k)), 40);
#else
  const int z = std::min(std::countl_zero(k), 40);
#endif
  const float m = static_cast<float>(static_cast<std::uint32_t>(k >> (40 - z))) + (z == 40 ? 0.5f : 0.0f);
#ifdef __CUDA_ARCH__
  const float v = m * __int_as_float((127 - 24 - z) << 23);  // m * 2^-(24 + z)
#else
  const float v = m * std::bit_cast<float>((127 - 24 - z) << 23);
#endif
  const float e = v < 0.5f ? -log1pf(-v) : -logf(1.0f - v);
  return -logf(e);
}

// Gumbel noise for (seed, stream, a, b).
__host__ __device__ __forceinline__ float gumbel(std::uint64_t seed, std::uint64_t stream, std::uint32_t a,
                                                 std::uint32_t b) {
  std::uint32_t c[4] = {static_cast<std::uint32_t>(stream), static_cast<std::uint32_t>(stream >> 32), a, b};
  Philox::generate(c, seed);
  return gumbel_from_bits((static_cast<std::uint64_t>(c[0]) << 32) | c[1]);
}

}  // namespace ingine::kernels::draft
