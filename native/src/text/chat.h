// Inkling-2 chat format (port of engine/ingine/chat.py): prompt rendering through the checkpoint's
// chat_template.jinja and an incremental parser of the model's output into OpenAI-style deltas.
//
// Output grammar (tokenizer_config.json response_template)::
//
//     <|message_model|>[HEADER]<|content_thinking|>TEXT<|end_message|>          -> reasoning_content
//     <|message_model|>[HEADER]<|content_text|>TEXT<|end_message|>              -> content
//       (also <|content_tool_error|> and <|content_invoke_tool_text|> bodies -> content)
//     <|message_model|>NAME<|content_invoke_tool_json|>{"name":NAME,"args":{...}}<|end_message|>
//                                                                               -> tool call
//     ...<|content_model_end_sampling|>                                         -> end of turn
#pragma once

#include "text/tokenizer.h"

#include <nlohmann/json.hpp>

#include <cstdint>
#include <filesystem>
#include <functional>
#include <memory>
#include <optional>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace ingine::text {

inline constexpr std::string_view MESSAGE_MODEL = "<|message_model|>";
inline constexpr std::string_view MESSAGE_USER = "<|message_user|>";
inline constexpr std::string_view MESSAGE_SYSTEM = "<|message_system|>";
inline constexpr std::string_view MESSAGE_TOOL = "<|message_tool|>";
inline constexpr std::string_view CONTENT_TEXT = "<|content_text|>";
inline constexpr std::string_view CONTENT_THINKING = "<|content_thinking|>";
inline constexpr std::string_view CONTENT_INVOKE_TOOL_JSON = "<|content_invoke_tool_json|>";
inline constexpr std::string_view CONTENT_INVOKE_TOOL_TEXT = "<|content_invoke_tool_text|>";
inline constexpr std::string_view CONTENT_TOOL_ERROR = "<|content_tool_error|>";
inline constexpr std::string_view END_MESSAGE = "<|end_message|>";
inline constexpr std::string_view END_SAMPLING = "<|content_model_end_sampling|>";
inline constexpr std::string_view ENDOFTEXT = "<|endoftext|>";
inline constexpr std::string_view EFFORT_LEVELS[] = {"none", "minimal", "low", "medium", "high", "xhigh", "max"};

// Invalid chat request (HTTP 400); `param` names the offending request field. what() is the Python message.
struct ChatError : std::runtime_error {
  std::string param;
  explicit ChatError(const std::string& message, std::string param_name = "messages")
      : std::runtime_error(message), param(std::move(param_name)) {}
};

// The request values exactly as server.py passes them to ChatFormat.render (raw JSON, validated inside).
struct ChatOptions {
  nlohmann::ordered_json reasoning_effort;                                    // null: template default (0.9)
  nlohmann::ordered_json enable_thinking = true;                              // Python truthiness
  nlohmann::ordered_json template_kwargs = nlohmann::ordered_json::object();  // extra template variables
};

struct RenderedPrompt {
  std::vector<int32_t> ids;          // full prompt, prefill included
  std::string text;
  std::vector<int32_t> prefill_ids;  // tail of ids the model would otherwise have generated
};

struct ToolCallDelta {
  int index = 0;
  std::optional<std::string> id, name;  // set on the first delta of each call
  std::string arguments;
  bool operator==(const ToolCallDelta&) const = default;
};

// One output event; exactly one of the fields is non-empty.
struct Delta {
  std::string reasoning_content, content;
  std::optional<ToolCallDelta> tool_call;
  bool operator==(const Delta&) const = default;
};

struct ToolCall {
  std::string id, name, arguments;  // arguments: JSON object text
  bool operator==(const ToolCall&) const = default;
};

struct Completion {
  std::string reasoning_content, content;
  std::vector<ToolCall> tool_calls;
  std::optional<std::string> finish_reason;  // "stop" | "length" | "tool_calls"; nullopt while unfinished
};

// Coalesce adjacent events of the same kind (same tool call for argument deltas): chat.merge_deltas.
std::vector<Delta> merge_deltas(const std::vector<Delta>& deltas);

// chat.normalize_effort: null, a level name or a float in [0, 0.99]; throws ChatError(param "reasoning_effort").
nlohmann::ordered_json normalize_effort(const nlohmann::ordered_json& effort);
// chat.normalize_messages: OpenAI chat messages -> the shapes the checkpoint template expects.
nlohmann::ordered_json normalize_messages(const nlohmann::ordered_json& messages);

namespace detail {
struct FormatData;
}

// Incremental parser of generated token ids into Delta events (chat.StreamParser). One per request.
class StreamParser {
 public:
  std::vector<Delta> feed(int32_t token);
  std::vector<Delta> finish();  // end of generation: flush pending text/tool calls and fix finish_reason
  // True once no further tokens can change the output (stop token, stop string, or finish()).
  bool finished() const noexcept { return saw_stop_token_ || stopped_by_string_ || finish_reason_.has_value(); }
  Completion result() const;
  int num_tokens() const noexcept { return num_tokens_; }  // generated tokens (prefill excluded)
  const std::optional<std::string>& finish_reason() const noexcept { return finish_reason_; }

 private:
  friend class ChatFormat;
  enum class State { Outside, Header, Body };
  enum class Kind { None, Thinking, Text, ToolJson };
  struct Detokenizer {
    std::vector<int32_t> ids;
    std::size_t prefix = 0, read = 0;
  };
  struct ToolState {
    int index = 0;
    std::string id, header, body;
    std::optional<std::string> name;
    std::optional<std::size_t> args_start;
    std::string emitted;
  };

  StreamParser(std::shared_ptr<const detail::FormatData> fmt, std::vector<std::string> stop,
               std::function<std::string()> id_factory);
  std::string decode(std::span<const int32_t> ids) const;
  std::string detok_push(int32_t token);
  std::string detok_flush();
  void begin_body(Kind kind);
  void end_body(std::vector<Delta>& out);
  void route(const std::string& text, std::vector<Delta>& out);
  void emit_content(std::string_view text, std::vector<Delta>& out);
  void text(const std::string& text, std::vector<Delta>& out);
  void announce(ToolState& t, std::vector<Delta>& out);
  void tool_args(const ToolState& t, const std::string& args, std::vector<Delta>& out);
  void tool_text(const std::string& text, std::vector<Delta>& out);
  void close_tool(std::vector<Delta>& out);

  std::shared_ptr<const detail::FormatData> fmt_;
  std::vector<std::string> stop_;
  std::size_t hold_len_ = 0;
  std::function<std::string()> new_id_;
  State state_ = State::Outside;
  std::vector<int32_t> header_ids_;
  std::string header_;
  Kind kind_ = Kind::None;
  Detokenizer detok_;
  std::string hold_;  // content withheld while it may begin a stop string
  std::optional<ToolState> tool_;
  std::string reasoning_, content_;
  std::vector<ToolCall> tool_calls_;
  int num_tokens_ = 0;
  bool saw_stop_token_ = false;  // an end-of-turn / foreign-role token was generated
  bool stopped_by_string_ = false;
  std::optional<std::string> finish_reason_;
};

// Tokenizer-bound renderer and parser factory. Immutable after load except for the render cache (below); all
// methods are thread-safe.
class ChatFormat {
 public:
  static ChatFormat load(const std::filesystem::path& model_dir);  // tokenizer.json + chat_template.jinja

  // ChatFormat.render(messages, tools=tools, reasoning_effort=..., enable_thinking=..., **template_kwargs).
  // `tools` null = absent. Throws ChatError. The ids are tokenised incrementally against the most recent renders
  // (a few conversations): only the text after the last added token of the shared rendered prefix is encoded.
  RenderedPrompt render(const nlohmann::ordered_json& messages, const nlohmann::ordered_json& tools,
                        const ChatOptions& options) const;
  // Forget the cached renders (shared by copies of this ChatFormat): the next render tokenises from scratch.
  void clear_render_cache() const;
  // Parser primed with the prompt's prefill ids. id_factory makes tool-call ids (default "call_" + 24 hex).
  StreamParser parser(std::span<const int32_t> prefill_ids, std::vector<std::string> stop = {},
                      std::function<std::string()> id_factory = {}) const;
  // Non-streaming parse of a whole (possibly truncated) generation (ChatFormat.parse).
  Completion parse(std::span<const int32_t> ids, std::span<const int32_t> prefill_ids = {},
                   std::vector<std::string> stop = {}, std::function<std::string()> id_factory = {}) const;
  // End-of-turn ids the generator stops at (end sampling, endoftext, user/system/tool message), ascending.
  std::vector<int32_t> stop_ids() const;
  // <|message_model|><|content_text|>: appended to the prompt when thinking is disabled.
  std::vector<int32_t> no_thinking_prefill() const;
  // tokenizer.decode(ids, skip_special_tokens=False): lossy UTF-8.
  std::string decode(std::span<const int32_t> ids) const;
  const Tokenizer& tokenizer() const;

 private:
  explicit ChatFormat(std::shared_ptr<const detail::FormatData> data) : data_(std::move(data)) {}
  std::shared_ptr<const detail::FormatData> data_;
};

}  // namespace ingine::text
