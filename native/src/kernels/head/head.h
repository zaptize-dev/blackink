// Adapted from fngine (github.com/satellitedown/fngine, Apache-2.0) for ingine.
// Token selection over FP32 logits rows (vocabulary 200058): greedy argmax and engine.Engine's sampling
// (temperature, top-k, top-p, then a multinomial draw), plus the distribution and random draws the
// speculative acceptance (engine._accept) needs.
//
// Semantics (engine._probs): z = logits * (1 / temperature) (torch divides by a scalar via its reciprocal);
// top-k (k in [1, V)) keeps z >= the k-th largest z (ties kept, as masked_fill(logits < kth)); p = softmax(z);
// top-p (< 1) sorts p descending and drops every token whose exclusive cumulative probability exceeds top_p,
// then renormalises. Tokens tied with the last kept one are kept too (torch's unstable sort may split such
// ties; float ties at the boundary are the only difference). Draws use a counter-based RNG: a pure
// function of (seed, counter, row, index), so a seeded request is reproducible and graph replay is safe;
// every launcher that draws randomness uses rng.counter and then increments it.
//
// No workspace, no host synchronisation; all outputs are device memory on `stream`. Up to 8 rows, every entry
// point splits each row over 256 blocks (a few microseconds at V = 200058; top-k / top-p select their threshold
// key digit by digit, one launch per 8-bit digit, ~25 us of GPU time per row) and combines the slices through
// per-entry-point state in static device memory: calls of the same entry point must not run concurrently (one
// stream, as the Generator uses). Their draws equal the one-CTA kernels' (more than 8 rows) token for token.
#pragma once

#include <cuda_runtime.h>

#include <cstdint>

namespace ingine::kernels::head {

struct SamplerRng {
  std::uint64_t seed = 0;
  std::uint64_t counter = 0;
};

struct SamplingParams {
  float temperature = 1.0f;  // <= 0: greedy (sample() = argmax)
  int top_k = 0;             // <= 0 or >= V: disabled
  float top_p = 1.0f;        // >= 1: disabled
};

// out[r] = first index of the maximum of row r (torch.argmax). logits fp32 [rows, stride], V <= stride.
void argmax(const float* logits, int rows, int V, std::int64_t stride, std::int32_t* out, cudaStream_t stream);

// probs[r, v] = engine._probs(logits[r], params) (requires temperature > 0); probs fp32 [rows, V].
void sampling_probs(const float* logits, int rows, int V, std::int64_t stride, const SamplingParams& params,
                    float* probs, cudaStream_t stream);

// out[r] = one draw from engine._probs(logits[r]) (Gumbel-max over the kept tokens); temperature <= 0: argmax.
void sample(const float* logits, int rows, int V, std::int64_t stride, const SamplingParams& params,
            SamplerRng& rng, std::int32_t* out, cudaStream_t stream);

// out[r] = one draw from the non-negative weights probs[r] (torch.multinomial of the normalised row;
// an all-zero row gives 0).
void sample_from_probs(const float* probs, int rows, int V, std::int64_t stride, SamplerRng& rng,
                       std::int32_t* out, cudaStream_t stream);

// out[i] = uniform in [0, 1), i < n (24-bit resolution, like torch.rand for FP32).
void uniform(int n, SamplerRng& rng, float* out, cudaStream_t stream);

}  // namespace ingine::kernels::head
