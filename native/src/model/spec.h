// DSpark speculative decoding loop: C++ port of engine.Engine._generate_spec / _accept / _rollback.
//
// Per round: the drafter proposes 7 tokens after the anchor (the last sampled, not yet processed token at
// position P); only the leading drafts whose confidence (max softmax probability) exceeds
// options.spec_threshold are verified, in the smallest verify width of {2, 4, 8} holding anchor + them; the
// verify call runs the target's decode step of that width (history + taps); acceptance is the longest
// prefix equal to the target argmax (greedy) or speculative rejection sampling against the drafts' own
// distributions (sampled, exact for the target's distribution); rejected rows are rolled back (conv states
// from the call's history, ring slots from the backup), the accepted rows' taps become draft context, and
// the accepted drafts plus the correction/bonus token are emitted.
//
// The width choice, the backup, the acceptance and the rollback run on the device (two graphs with a switch
// over the verify widths), and so do the commit of the accepted rows and the next draft (Drafter::advance,
// bound to this loop's device buffers), so a round and the next draft are launched as a whole: one host
// synchronisation per round, after the acceptance, while the next draft already runs.
#pragma once

#include "kernels/head/head.h"
#include "model/drafter.h"
#include "model/runtime.h"

#include "ingine/generator.h"

#include <cuda_runtime.h>

#include <cstdint>
#include <functional>
#include <span>
#include <vector>

namespace ingine::model {

class SpecLoop {
 public:
  // Captures the verify graph (verify_select -> switch { save_verify + decode(m) }) for `rt` and `drafter`
  // (same stream; rt with the verify widths). Both must outlive the loop.
  SpecLoop(Runtime& rt, Drafter& drafter);
  ~SpecLoop();
  SpecLoop(const SpecLoop&) = delete;
  SpecLoop& operator=(const SpecLoop&) = delete;

  // Generates after a prefill: `token` is the first sampled token (not yet processed), at position
  // P = tokens_in_state.size(); the drafter must hold every position before P (drafter.length() == P).
  // Emits tokens through on_tokens (several per round; stops after a token in stop_ids, after
  // max_new_tokens tokens, or when on_tokens returns false). Appends every processed token to
  // tokens_in_state and sets stats.generated_tokens, spec_rounds and spec_accepted. Sampled acceptance draws
  // from `rng` (and seeds the drafter's Gumbel noise from it). The caller reserves BLOCK positions of
  // max_ctx for the verify calls (engine.generate: prompt + max_new_tokens + 8 <= max_ctx).
  void generate(std::int32_t token, int max_new_tokens, const Sampling& sampling, kernels::head::SamplerRng& rng,
                std::span<const std::int32_t> stop_ids,
                const std::function<bool(std::span<const std::int32_t>)>& on_tokens, GenerateStats& stats,
                std::vector<std::int32_t>& tokens_in_state);

 private:
  // The accept graph (width_select -> switch { accept + rollback + publish }) for this sampling setup.
  void prepare_accept(const Sampling& sampling);

  Runtime& rt_;
  Drafter& drafter_;
  Drafter::Draft draft_{};             // the drafter's fixed output buffers
  DeviceBuffer width_;                 // int32: the verify width chosen on the device
  MappedHostBuffer record_;            // VerifyRecord of the last round
  MappedHostBuffer forced_window_;     // forced replay: int32 [8], the forced tokens after the round's anchor
  GraphExec verify_, accept_;
  struct AcceptKey {
    bool greedy, forced;
    float temperature, top_p;
    int top_k;
    bool operator==(const AcceptKey&) const = default;
  };
  AcceptKey accept_key_{};
  cudaEvent_t done_ = nullptr;
};

}  // namespace ingine::model
