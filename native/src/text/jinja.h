// Adapted from fngine (github.com/satellitedown/fngine, Apache-2.0) for ingine.
// Jinja adapter over the vendored llama-jinja runtime: parse once, render with a JSON context. raise_exception
// is provided like transformers' chat-template environment.
#pragma once

#include <nlohmann/json.hpp>

#include <memory>
#include <stdexcept>
#include <string>

namespace ingine::text {

// Template failure. For raise_exception(msg), what() is exactly msg (jinja2.exceptions.TemplateError(msg));
// other runtime errors carry the interpreter's message.
struct TemplateError : std::runtime_error {
  using std::runtime_error::runtime_error;
};

// The parsed program is immutable; each render owns its evaluation context, so render() is thread-safe.
class JinjaTemplate {
 public:
  explicit JinjaTemplate(const std::string& source);
  std::string render(const nlohmann::ordered_json& context) const;

 private:
  struct Impl;
  std::shared_ptr<const Impl> impl_;
};

}  // namespace ingine::text
