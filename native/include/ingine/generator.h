// The token generator interface the server drives (mirror of engine.Engine.generate). CUDA-free so the
// scheduler, the HTTP layer and their tests build without the model; ingine_model implements make_generator.
#pragma once

#include <array>
#include <cstdint>
#include <filesystem>
#include <functional>
#include <memory>
#include <optional>
#include <span>
#include <string>
#include <vector>

namespace ingine::model {

struct Sampling {
  float temperature = 1.0f;  // 0 => greedy
  float top_p = 1.0f;
  int top_k = 0;  // 0 => disabled
  std::optional<uint64_t> seed;
  // Teacher-forced replay (test/benchmark only; empty = off): generate() emits exactly these tokens
  // (max_new_tokens is capped at their count), running its real loop along them: plain decoding samples as
  // usual and then takes the forced token; speculative decoding accepts a draft iff it equals the forced token
  // and continues with the forced token after the accepted run. Fills GenerateStats::spec_trace.
  std::span<const int32_t> forced;
};

// How the prompt reused the state of earlier requests (reused_tokens were not prefilled again).
enum class Reuse {
  None,        // empty state, the whole prompt prefilled
  Extend,      // the prompt extends the tokens in the state (previous prompt + processed generated tokens)
  Checkpoint,  // restored the checkpoint taken after the previous prompt's prefill (a prefix of this prompt)
  Parked,      // restored a parked conversation's checkpoint (a prefix of this prompt; RuntimeOptions::park_mb)
};
inline const char* to_string(Reuse reuse) {
  switch (reuse) {
    case Reuse::Extend: return "extend";
    case Reuse::Checkpoint: return "checkpoint";
    case Reuse::Parked: return "parked";
    default: return "none";
  }
}

// One speculative round of a forced replay: verify width, accepted drafts, the drafts that would have been
// accepted had all 7 been verified, and GPU time of the round's verify (+ acceptance) and of its draft.
struct SpecRoundRecord {
  int width = 0, accepted = 0, accepted_all = 0;
  float verify_ms = 0, draft_ms = 0;
};

struct GenerateStats {
  int prompt_tokens = 0, reused_tokens = 0, generated_tokens = 0;
  Reuse reuse = Reuse::None;
  double prefill_seconds = 0, decode_seconds = 0;
  int spec_rounds = 0, spec_accepted = 0;
  std::array<int, 7> spec_accepted_per_pos{};
  std::vector<SpecRoundRecord> spec_trace;  // forced replay only
};

struct RuntimeOptions {
  std::filesystem::path model_dir, ablation;  // empty ablation: original model
  int max_ctx = 131072;
  // Tokens per prefill chunk. 256 keeps prefill scratch to ~110 MiB so 16 GB Blackwell GPUs have plenty of
  // headroom for 96K Q4 KV cache + DSpark MTP speculative decoding.
  int prefill_chunk = 384;
  bool dense_fp8 = true, kv_fp8 = false, kv_q4 = false, full_nvfp4 = false, spec = false;
  float spec_threshold = 0.8f;
  // Conversation parking: device memory (MiB) for the checkpoints of conversations displaced by another
  // one, least recently used evicted, all released when a device allocation fails; 0 disables; negative:
  // min(3072, free memory after start-up - 4096), at least 0.
  int park_mb = 0;
};

struct ImageSpan {
  int offset = 0;           // starting token index in prompt
  int n_tokens = 0;         // number of visual tokens
  std::vector<float> data;  // [n_tokens * 2304] row-major FP32
};

class Generator {
 public:
  virtual ~Generator() = default;
  virtual std::string model_id() const = 0;  // "inkling-2-mini-abliterated" with an ablation, else "inkling-2-mini"
  virtual int max_model_len() const = 0;     // = max_ctx
  // One request at a time. Emits sampled tokens in order through on_tokens (a speculative round emits several);
  // stops after a token in stop_ids, after max_new_tokens, or when on_tokens returns false (cancel).
  // Throws std::invalid_argument when the prompt does not fit max_ctx (reserve as engine.py does).
  virtual GenerateStats generate(std::span<const int32_t> prompt, int max_new_tokens, const Sampling& sampling,
                                 std::span<const int32_t> stop_ids,
                                 const std::function<bool(std::span<const int32_t>)>& on_tokens) = 0;
  virtual GenerateStats generate_multimodal(std::span<const int32_t> prompt,
                                            std::span<const ImageSpan> /*images*/,
                                            int max_new_tokens, const Sampling& sampling,
                                            std::span<const int32_t> stop_ids,
                                            const std::function<bool(std::span<const int32_t>)>& on_tokens) {
    return generate(prompt, max_new_tokens, sampling, stop_ids, on_tokens);
  }
};

std::unique_ptr<Generator> make_generator(const RuntimeOptions& options);  // the model (ingine_model)

}  // namespace ingine::model
