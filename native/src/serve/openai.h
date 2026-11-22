// OpenAI Chat Completions request parsing and JSON shapes, exactly as engine/ingine/server.py
// (Handler._parse_request, _num, _send_error, _delta_json).
#pragma once

#include "ingine/generator.h"
#include "text/chat.h"

#include <nlohmann/json.hpp>

#include <cstdint>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace ingine::serve {

using Json = nlohmann::ordered_json;

// Invalid request -> HTTP `status` with an OpenAI error body naming `param`.
struct BadRequest : std::runtime_error {
  BadRequest(const std::string& message, std::optional<std::string> param_ = std::nullopt, int status_ = 400)
      : std::runtime_error(message), param(std::move(param_)), status(status_) {}
  std::optional<std::string> param;
  int status;
};

struct ChatRequest {
  std::vector<int32_t> prompt_ids;   // full prompt, prefill included
  std::vector<int32_t> prefill_ids;  // tail of prompt_ids the model would otherwise have generated
  int max_new_tokens = 0;
  model::Sampling sampling;
  std::vector<std::string> stop;
  bool stream = false;
  bool include_usage = false;
};

// Validates `body` (a JSON object) and renders the prompt. Throws BadRequest or text::ChatError. Sampling fields
// the request omits (temperature, top_p, top_k) take their values from `defaults`.
ChatRequest parse_request(const Json& body, const text::ChatFormat& fmt, int max_model_len,
                          const model::Sampling& defaults = {});

// {"error": {"message", "type", "param", "code"}} with the type server.py maps from the status.
Json error_body(int status, std::string_view message, const std::optional<std::string>& param = std::nullopt);

// One streamed delta: reasoning_content, content or a tool_calls entry.
Json delta_json(const text::Delta& delta);

// json.dumps(ensure_ascii=False) equivalent (compact separators); invalid UTF-8 becomes U+FFFD.
std::string dump(const Json& value);

// Python truthiness of a JSON value.
bool truthy(const Json& value);

}  // namespace ingine::serve
