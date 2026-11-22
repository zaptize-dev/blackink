// Canned-output generator (mirror of engine/ingine/server.py FakeGenerator) for tests and `ingine-serve --fake`.
#pragma once

#include "ingine/generator.h"

#include <cstdint>
#include <functional>
#include <mutex>
#include <span>
#include <string>
#include <variant>
#include <vector>

namespace ingine::text {
class Tokenizer;
}

namespace ingine::engine {

// Replays canned token ids, honouring max_new_tokens, stop_ids and cancellation. responder(prompt_ids) returns
// the full output as token ids or as text (encoded with added tokens recognised); the default echoes the last
// user message as thinking + answer, or as a tool call when it reads `call NAME {json args}`. calls() records
// every request and how it ended.
class FakeGenerator final : public model::Generator {
 public:
  using Output = std::variant<std::vector<int32_t>, std::string>;
  using Responder = std::function<Output(std::span<const int32_t>)>;

  struct Call {
    std::vector<int32_t> prompt_ids;
    int max_new_tokens = 0;
    model::Sampling sampling;
    int yielded = 0;
    bool cancelled = false;
    bool done = false;
  };

  explicit FakeGenerator(const text::Tokenizer& tokenizer, std::string model_id = "Inkling-2-Mini-fake",
                         int max_model_len = 1 << 20, Responder responder = {}, double token_delay = 0.0);

  std::string model_id() const override { return model_id_; }
  int max_model_len() const override { return max_model_len_; }
  model::GenerateStats generate(std::span<const int32_t> prompt, int max_new_tokens,
                                const model::Sampling& sampling, std::span<const int32_t> stop_ids,
                                const std::function<bool(std::span<const int32_t>)>& on_tokens) override;

  std::string echo(std::span<const int32_t> prompt_ids) const;
  std::vector<Call> calls() const;

 private:
  const text::Tokenizer& tokenizer_;
  std::string model_id_;
  int max_model_len_;
  Responder responder_;
  double token_delay_;
  mutable std::mutex mutex_;
  std::vector<Call> calls_;
};

}  // namespace ingine::engine
