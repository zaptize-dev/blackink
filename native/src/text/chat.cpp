#include "text/chat.h"

#include "text/jinja.h"
#include "text/pyjson.h"
#include "text/unicode.h"

#include <algorithm>
#include <cstring>
#include <deque>
#include <fstream>
#include <mutex>
#include <random>
#include <sstream>
#include <unordered_map>
#include <unordered_set>

namespace ingine::text {

using Json = nlohmann::ordered_json;
namespace uni = unicode;

namespace detail {
// Recent rendered prompts (text, ids, added-token boundaries) for incremental tokenisation: consecutive requests
// of a conversation share a long prefix of rendered text, and only the text after the last added token in that
// prefix needs encoding. Shared by request threads.
struct RenderCache {
  struct Entry {
    std::string text;
    std::vector<int32_t> ids;
    std::vector<TokenBoundary> boundaries;
  };
  static constexpr std::size_t kEntries = 4;  // concurrent conversations kept
  std::mutex mu;
  std::deque<std::shared_ptr<const Entry>> entries;  // most recent first
};

struct FormatData {
  Tokenizer tokenizer;
  JinjaTemplate chat_template;
  int32_t message_model = -1, message_user = -1, message_system = -1, message_tool = -1;
  int32_t content_text = -1, end_message = -1, end_sampling = -1, endoftext = -1;
  std::unordered_set<int32_t> stop_ids;
  std::unordered_map<int32_t, int> content_kind;  // tag id -> StreamParser kind (1 thinking, 2 text, 3 tool json)
  std::unique_ptr<RenderCache> cache = std::make_unique<RenderCache>();
};
}  // namespace detail

namespace {

constexpr std::string_view kReplacementChar = "\xEF\xBF\xBD";

int32_t require_id(const Tokenizer& tok, std::string_view name) {
  const auto id = tok.token_id(name);
  if (!id) throw std::invalid_argument("tokenizer.json lacks " + std::string(name));
  return *id;
}

std::string read_file(const std::filesystem::path& path) {
  std::ifstream in(path, std::ios::binary);
  if (!in) throw std::runtime_error("cannot open " + path.string());
  std::ostringstream ss;
  ss << in.rdbuf();
  return ss.str();
}

const Json* get(const Json& object, const char* key) {
  if (!object.is_object()) return nullptr;
  const auto it = object.find(key);
  return it == object.end() ? nullptr : &*it;
}

bool is_none(const Json* v) { return v == nullptr || v->is_null(); }

Json normalize_tool_call(const Json& tc) {
  const Json* function = get(tc, "function");
  if (!tc.is_object() || function == nullptr || !function->is_object()) {
    throw ChatError("tool_calls entries must be objects with a 'function' object");
  }
  Json fn = *function;
  const Json* name = get(fn, "name");
  if (name == nullptr || !name->is_string()) throw ChatError("tool call function name must be a string");
  const std::string& fname = name->get_ref<const std::string&>();
  const Json* arguments = get(fn, "arguments");
  Json args;
  if (is_none(arguments) || (arguments->is_string() && arguments->get_ref<const std::string&>().empty())) {
    args = Json::object();
  } else if (arguments->is_string()) {
    try {
      args = py::json_loads(arguments->get_ref<const std::string&>());
    } catch (const py::JsonDecodeError& e) {
      throw ChatError("tool call arguments for " + py::str_repr(fname) + " are not valid JSON: " + e.what());
    }
  } else {
    args = *arguments;
  }
  if (!args.is_object()) throw ChatError("tool call arguments for " + py::str_repr(fname) + " must be a JSON object");
  fn["arguments"] = std::move(args);
  Json out = tc;
  out["function"] = std::move(fn);
  return out;
}

// Re-matching chat._TOOL_PREFIX: \s*\{\s*"name"\s*:\s*("(?:[^"\\]|\\.)*")\s*,\s*"args"\s*:\s* (re.S, Python \s).
struct PrefixMatch {
  std::string name_literal;
  std::size_t end = 0;
};

std::optional<PrefixMatch> match_tool_prefix(std::string_view s) {
  std::size_t pos = 0;
  const auto skip_ws = [&] {
    while (pos < s.size()) {
      const auto cp = uni::utf8_codepoint_at(s, pos, "tool call");
      if (!uni::is_py_space(cp.value)) break;
      pos += cp.length;
    }
  };
  const auto literal = [&](std::string_view lit) {
    if (s.compare(pos, lit.size(), lit) != 0 || pos + lit.size() > s.size()) return false;
    pos += lit.size();
    return true;
  };
  skip_ws();
  if (!literal("{")) return std::nullopt;
  skip_ws();
  if (!literal("\"name\"")) return std::nullopt;
  skip_ws();
  if (!literal(":")) return std::nullopt;
  skip_ws();
  const std::size_t start = pos;
  if (!literal("\"")) return std::nullopt;
  while (true) {
    if (pos >= s.size()) return std::nullopt;
    if (s[pos] == '"') break;
    if (s[pos] == '\\') {
      if (pos + 1 >= s.size()) return std::nullopt;
      pos += 1 + uni::utf8_codepoint_at(s, pos + 1, "tool call").length;
    } else {
      pos += uni::utf8_codepoint_at(s, pos, "tool call").length;
    }
  }
  ++pos;
  PrefixMatch m;
  m.name_literal = std::string(s.substr(start, pos - start));
  skip_ws();
  if (!literal(",")) return std::nullopt;
  skip_ws();
  if (!literal("\"args\"")) return std::nullopt;
  skip_ws();
  if (!literal(":")) return std::nullopt;
  skip_ws();
  m.end = pos;
  return m;
}

// chat._args_candidate: drop trailing blanks and the outer `}`.
std::string args_candidate(std::string_view tail) {
  tail = uni::py_rstrip(tail);
  if (!tail.empty() && tail.back() == '}') tail.remove_suffix(1);
  return std::string(tail);
}

std::string random_call_id() {
  thread_local std::mt19937_64 rng{std::random_device{}() ^ (static_cast<std::uint64_t>(std::random_device{}()) << 32)};
  static constexpr char kHex[] = "0123456789abcdef";
  std::string id = "call_";
  for (int i = 0; i < 24; ++i) id.push_back(kHex[rng() & 15]);
  return id;
}

// Length of the common prefix of a and b (memcmp in blocks, then bytewise in the differing block).
std::size_t common_prefix(std::string_view a, std::string_view b) {
  const std::size_t n = std::min(a.size(), b.size());
  constexpr std::size_t kBlock = 4096;
  std::size_t i = 0;
  while (i + kBlock <= n && std::memcmp(a.data() + i, b.data() + i, kBlock) == 0) i += kBlock;
  while (i < n && a[i] == b[i]) ++i;
  return i;
}

// tokenizer.encode(text, true), reusing the ids of a cached render that shares a prefix with text: the ids up to
// the last added token whose end b satisfies b + max_added - 1 <= p (p = common prefix length) are exact, because
// every added-token match starting before b reads only bytes before p, and BPE never crosses an added token.
std::vector<int32_t> encode_prompt(const detail::FormatData& fmt, const std::string& text) {
  detail::RenderCache& cache = *fmt.cache;
  std::deque<std::shared_ptr<const detail::RenderCache::Entry>> entries;
  {
    std::lock_guard lock(cache.mu);
    entries = cache.entries;
  }
  std::shared_ptr<const detail::RenderCache::Entry> base;
  std::size_t common = 0;
  for (const auto& e : entries) {
    const std::size_t p = common_prefix(e->text, text);
    if (p > common || !base) base = e, common = p;
  }

  auto entry = std::make_shared<detail::RenderCache::Entry>();
  entry->text = text;
  if (base && common == text.size() && base->text.size() == text.size()) {
    entry->ids = base->ids;
    entry->boundaries = base->boundaries;
  } else {
    std::size_t byte = 0, tokens = 0, kept = 0;
    if (base) {
      const std::size_t reach = fmt.tokenizer.max_added_token_bytes();
      for (const TokenBoundary& b : base->boundaries) {
        if (b.byte + reach > common + 1) break;
        byte = b.byte, tokens = b.tokens, ++kept;
      }
    }
    entry->ids.reserve(tokens + (text.size() - byte) / 3 + 1);
    if (base) {
      entry->ids.assign(base->ids.begin(), base->ids.begin() + static_cast<std::ptrdiff_t>(tokens));
      entry->boundaries.assign(base->boundaries.begin(), base->boundaries.begin() + static_cast<std::ptrdiff_t>(kept));
    }
    fmt.tokenizer.encode_special(std::string_view(text).substr(byte), entry->ids, &entry->boundaries, byte);
  }
  std::vector<int32_t> ids = entry->ids;
  {
    std::lock_guard lock(cache.mu);
    // A render extending most of the one it reused supersedes it (the conversation moved on); other
    // conversations keep their entries.
    if (base && common >= base->text.size() - base->text.size() / 10) {
      cache.entries.erase(std::remove(cache.entries.begin(), cache.entries.end(), base), cache.entries.end());
    }
    cache.entries.push_front(std::move(entry));
    while (cache.entries.size() > detail::RenderCache::kEntries) cache.entries.pop_back();
  }
  return ids;
}

}  // namespace

// ---- request normalisation ------------------------------------------------------------------------------------

Json normalize_effort(const Json& effort) {
  if (effort.is_null()) return effort;
  if (effort.is_boolean()) throw ChatError("invalid reasoning_effort: " + py::repr(effort), "reasoning_effort");
  Json value = effort;
  if (effort.is_string()) {
    const std::string key = uni::py_lower(uni::py_strip(effort.get_ref<const std::string&>()));
    for (const std::string_view level : EFFORT_LEVELS) {
      if (key == level) return Json(key);
    }
    const auto parsed = py::parse_float(key);
    if (!parsed) {
      throw ChatError("invalid reasoning_effort " + py::repr(effort) +
                          ": expected one of none, minimal, low, medium, high, xhigh, max or a number in [0, 0.99]",
                      "reasoning_effort");
    }
    value = Json(*parsed);
  }
  if (!value.is_number() || !(value.get<double>() >= 0.0 && value.get<double>() <= 0.99)) {
    throw ChatError("reasoning_effort must be in [0.0, 0.99], got " + py::repr(value), "reasoning_effort");
  }
  return Json(value.get<double>());
}

Json normalize_messages(const Json& messages) {
  if (!messages.is_array() || messages.empty()) throw ChatError("'messages' must be a non-empty list");
  Json out = Json::array();
  for (const Json& message : messages) {
    const Json* role = get(message, "role");
    if (!message.is_object() || role == nullptr || !role->is_string()) {
      throw ChatError("each message must be an object with a string 'role'");
    }
    Json m = message;
    if (m["role"] == "developer") m["role"] = "system";
    const std::string r = m["role"].get<std::string>();
    if (r != "system" && r != "user" && r != "assistant" && r != "tool") {
      throw ChatError("unsupported message role " + py::str_repr(r));
    }
    const Json* content = get(m, "content");
    if (is_none(content)) {
      m.erase("content");
    } else if (content->is_array()) {
      for (const Json& part : *content) {
        if (part.is_object()) {
          const Json* type = get(part, "type");
          if (!is_none(type) && !(*type == "text" || *type == "input_text")) {
            throw ChatError("unsupported content part type " + py::repr(*type) + " (text only)");
          }
        }
        if (!part.is_string() && !part.is_object()) throw ChatError("content parts must be strings or objects");
      }
    } else if (!content->is_string()) {
      throw ChatError("message content must be a string, a list of parts, or null");
    }
    if (r == "assistant") {
      const Json* rc = get(m, "reasoning_content");
      const Json* reasoning = get(m, "reasoning");
      if (!(rc != nullptr && py::truthy(*rc)) && reasoning != nullptr && reasoning->is_string()) {
        Json copy = *reasoning;
        m["reasoning_content"] = std::move(copy);
      }
      m.erase("reasoning");
      const Json* calls = get(m, "tool_calls");
      if (calls != nullptr && py::truthy(*calls)) {
        if (!calls->is_array()) throw ChatError("'tool_calls' must be a list");
        Json normalized = Json::array();
        for (const Json& tc : *calls) normalized.push_back(normalize_tool_call(tc));
        m["tool_calls"] = std::move(normalized);
      } else {
        m.erase("tool_calls");
      }
    }
    out.push_back(std::move(m));
  }
  return out;
}

std::vector<Delta> merge_deltas(const std::vector<Delta>& deltas) {
  std::vector<Delta> out;
  for (const Delta& d : deltas) {
    if (!out.empty()) {
      Delta& last = out.back();
      if (!d.reasoning_content.empty() && !last.reasoning_content.empty()) {
        last.reasoning_content += d.reasoning_content;
        continue;
      }
      if (!d.content.empty() && !last.content.empty()) {
        last.content += d.content;
        continue;
      }
      if (d.tool_call && last.tool_call && !d.tool_call->id && d.tool_call->index == last.tool_call->index) {
        last.tool_call->arguments += d.tool_call->arguments;
        continue;
      }
    }
    out.push_back(d);
  }
  return out;
}

// ---- ChatFormat -----------------------------------------------------------------------------------------------

ChatFormat ChatFormat::load(const std::filesystem::path& model_dir) {
  auto data = std::make_shared<detail::FormatData>(detail::FormatData{
      Tokenizer::load(model_dir), JinjaTemplate(read_file(model_dir / "chat_template.jinja"))});
  const Tokenizer& tok = data->tokenizer;
  data->message_model = require_id(tok, MESSAGE_MODEL);
  data->message_user = require_id(tok, MESSAGE_USER);
  data->message_system = require_id(tok, MESSAGE_SYSTEM);
  data->message_tool = require_id(tok, MESSAGE_TOOL);
  data->content_text = require_id(tok, CONTENT_TEXT);
  data->end_message = require_id(tok, END_MESSAGE);
  data->end_sampling = require_id(tok, END_SAMPLING);
  data->endoftext = require_id(tok, ENDOFTEXT);
  data->stop_ids = {data->end_sampling, data->endoftext, data->message_user, data->message_system,
                    data->message_tool};
  data->content_kind = {{require_id(tok, CONTENT_THINKING), 1},
                        {data->content_text, 2},
                        {require_id(tok, CONTENT_TOOL_ERROR), 2},
                        {require_id(tok, CONTENT_INVOKE_TOOL_TEXT), 2},
                        {require_id(tok, CONTENT_INVOKE_TOOL_JSON), 3}};
  return ChatFormat(std::move(data));
}

const Tokenizer& ChatFormat::tokenizer() const { return data_->tokenizer; }

void ChatFormat::clear_render_cache() const {
  std::lock_guard lock(data_->cache->mu);
  data_->cache->entries.clear();
}

std::vector<int32_t> ChatFormat::stop_ids() const {
  std::vector<int32_t> ids(data_->stop_ids.begin(), data_->stop_ids.end());
  std::sort(ids.begin(), ids.end());
  return ids;
}

std::vector<int32_t> ChatFormat::no_thinking_prefill() const { return {data_->message_model, data_->content_text}; }

std::string ChatFormat::decode(std::span<const int32_t> ids) const { return data_->tokenizer.decode_lossy(ids); }

RenderedPrompt ChatFormat::render(const Json& messages, const Json& tools, const ChatOptions& options) const {
  Json msgs = normalize_messages(messages);
  if (!tools.is_null() && !tools.is_array()) throw ChatError("'tools' must be a list", "tools");
  const Json effort = normalize_effort(options.reasoning_effort);

  // tokenizer.apply_chat_template(msgs, tools=tools or None, tokenize=False, reasoning_effort=effort, **kwargs)
  Json context = Json::object();
  if (!options.template_kwargs.is_null() && !options.template_kwargs.is_object()) {
    throw ChatError("chat template error: template kwargs must be an object");
  }
  if (options.template_kwargs.is_object()) {
    for (const auto& [key, value] : options.template_kwargs.items()) {
      if (key == "messages" || key == "conversation" || key == "tools" || key == "tokenize" ||
          key == "reasoning_effort" || key == "enable_thinking") {
        throw ChatError("chat template error: got multiple values for keyword argument '" + key + "'");
      }
      if (key == "chat_template" || key == "continue_final_message" || key == "sanitize_control_tokens" ||
          key == "return_assistant_tokens_mask") {
        throw ChatError("chat template error: chat_template_kwargs key '" + key + "' is not supported");
      }
      // apply_chat_template parameters that cannot change a tokenize=False render of this template.
      if (key == "documents" || key == "add_generation_prompt" || key == "padding" || key == "truncation" ||
          key == "max_length" || key == "return_tensors" || key == "return_dict" || key == "tokenizer_kwargs") {
        continue;
      }
      context[key] = value;
    }
  }
  Json tool_schemas = nullptr;
  if (tools.is_array() && !tools.empty()) {
    for (const Json& tool : tools) {
      if (!tool.is_object()) {
        throw ChatError(
            "chat template error: Tools should either be a JSON schema, or a callable function with type hints and "
            "a docstring suitable for auto-conversion to a schema.");
      }
    }
    tool_schemas = tools;
  }
  // transformers: drop None-valued message keys, then alias reasoning fields (reasoning_content, thinking,
  // reasoning: the first non-None one becomes reasoning_content and thinking).
  for (Json& m : msgs) {
    Json cleaned = Json::object();
    for (auto it = m.begin(); it != m.end(); ++it) {
      if (!it->is_null()) cleaned[it.key()] = std::move(*it);
    }
    for (const char* key : {"reasoning_content", "thinking", "reasoning"}) {
      if (const Json* r = get(cleaned, key)) {
        Json reasoning = *r;
        if (!cleaned.contains("thinking")) {
          Json aliased = Json::object();
          aliased["thinking"] = reasoning;
          for (auto it = cleaned.begin(); it != cleaned.end(); ++it) aliased[it.key()] = std::move(*it);
          cleaned = std::move(aliased);
        }
        cleaned["reasoning_content"] = std::move(reasoning);
        break;
      }
    }
    m = std::move(cleaned);
  }
  context["messages"] = std::move(msgs);
  context["tools"] = std::move(tool_schemas);
  context["documents"] = nullptr;
  context["add_generation_prompt"] = false;
  context["reasoning_effort"] = effort;

  RenderedPrompt prompt;
  try {
    prompt.text = data_->chat_template.render(context);
  } catch (const TemplateError& e) {
    throw ChatError(std::string("chat template error: ") + e.what());
  }
  if (!py::truthy(options.enable_thinking)) {
    prompt.text += MESSAGE_MODEL;
    prompt.text += CONTENT_TEXT;
    prompt.prefill_ids = no_thinking_prefill();
  }
  prompt.ids = encode_prompt(*data_, prompt.text);
  return prompt;
}

StreamParser ChatFormat::parser(std::span<const int32_t> prefill_ids, std::vector<std::string> stop,
                                std::function<std::string()> id_factory) const {
  StreamParser p(data_, std::move(stop), std::move(id_factory));
  for (const int32_t t : prefill_ids) p.feed(t);
  p.num_tokens_ = 0;  // count generated tokens only
  return p;
}

Completion ChatFormat::parse(std::span<const int32_t> ids, std::span<const int32_t> prefill_ids,
                             std::vector<std::string> stop, std::function<std::string()> id_factory) const {
  StreamParser p = parser(prefill_ids, std::move(stop), std::move(id_factory));
  for (const int32_t t : ids) {
    if (p.finished()) break;
    p.feed(t);
  }
  p.finish();
  return p.result();
}

// ---- StreamParser ---------------------------------------------------------------------------------------------

StreamParser::StreamParser(std::shared_ptr<const detail::FormatData> fmt, std::vector<std::string> stop,
                           std::function<std::string()> id_factory)
    : fmt_(std::move(fmt)), new_id_(id_factory ? std::move(id_factory) : std::function<std::string()>(random_call_id)) {
  std::size_t longest = 1;
  bool any = false;
  for (auto& s : stop) {
    if (s.empty()) continue;
    const std::size_t n = uni::utf8_length(s);
    longest = any ? std::max(longest, n) : n;
    any = true;
    stop_.push_back(std::move(s));
  }
  hold_len_ = longest - 1;
}

Completion StreamParser::result() const {
  return Completion{reasoning_, content_, tool_calls_, finish_reason_};
}

std::string StreamParser::decode(std::span<const int32_t> ids) const { return fmt_->tokenizer.decode_lossy(ids); }

std::vector<Delta> StreamParser::feed(int32_t tok) {
  std::vector<Delta> out;
  if (finished()) return out;
  ++num_tokens_;
  const detail::FormatData& fmt = *fmt_;
  if (fmt.stop_ids.contains(tok)) {
    end_body(out);
    saw_stop_token_ = true;
    return out;
  }
  if (const auto kind = fmt.content_kind.find(tok); kind != fmt.content_kind.end()) {
    if (state_ == State::Body) {
      end_body(out);  // missing <|end_message|>: keep the header, start a new body
    } else if (state_ == State::Header) {
      header_ = decode(header_ids_);
    } else {
      header_.clear();
    }
    header_ids_.clear();
    begin_body(static_cast<Kind>(kind->second));
    return out;
  }
  if (fmt.tokenizer.is_special(tok)) {
    if (tok == fmt.message_model) {
      end_body(out);
      state_ = State::Header;
      header_ids_.clear();
    } else if (tok == fmt.end_message) {
      end_body(out);
      state_ = State::Outside;
    }
    // Other special tokens (content_xml/content_image tags, unused ids) carry no text. A message opened with an
    // unknown content tag stays in Header state and is discarded at its end.
    return out;
  }
  if (state_ == State::Header) {
    header_ids_.push_back(tok);
  } else if (state_ == State::Body) {
    const std::string text = detok_push(tok);
    if (!text.empty()) route(text, out);
  }
  return out;  // text outside any message is ignored, as in the checkpoint's response_template
}

std::vector<Delta> StreamParser::finish() {
  std::vector<Delta> out;
  if (finish_reason_) return out;
  if (!(saw_stop_token_ || stopped_by_string_)) end_body(out);
  if (stopped_by_string_) {
    finish_reason_ = "stop";
  } else if (saw_stop_token_) {
    finish_reason_ = tool_calls_.empty() ? "stop" : "tool_calls";
  } else {
    finish_reason_ = "length";
  }
  return out;
}

// Incremental decode of one message body; holds back incomplete UTF-8 sequences (chat._Detokenizer).
std::string StreamParser::detok_push(int32_t tok) {
  Detokenizer& d = detok_;
  d.ids.push_back(tok);
  const std::span<const int32_t> ids(d.ids);
  const std::string prefix_text = d.read > d.prefix ? decode(ids.subspan(d.prefix, d.read - d.prefix)) : "";
  const std::string new_text = decode(ids.subspan(d.prefix));
  const std::size_t prefix_len = uni::utf8_length(prefix_text);
  if (uni::utf8_length(new_text) > prefix_len && !new_text.ends_with(kReplacementChar)) {
    d.prefix = d.read;
    d.read = d.ids.size();
    return new_text.substr(uni::utf8_offset(new_text, prefix_len));
  }
  return {};
}

std::string StreamParser::detok_flush() {
  Detokenizer& d = detok_;
  std::string out;
  if (d.read < d.ids.size()) {
    const std::span<const int32_t> ids(d.ids);
    const std::string prefix_text = d.read > d.prefix ? decode(ids.subspan(d.prefix, d.read - d.prefix)) : "";
    const std::string all = decode(ids.subspan(d.prefix));
    out = all.substr(uni::utf8_offset(all, uni::utf8_length(prefix_text)));
  }
  d.ids.clear();
  d.prefix = d.read = 0;
  return out;
}

void StreamParser::begin_body(Kind kind) {
  state_ = State::Body;
  kind_ = kind;
  if (kind == Kind::ToolJson) {
    tool_ = ToolState{};
    tool_->index = static_cast<int>(tool_calls_.size());
    tool_->id = new_id_();
    tool_->header = std::string(uni::py_strip(header_));
  }
}

void StreamParser::end_body(std::vector<Delta>& out) {
  if (state_ != State::Body) {
    if (state_ == State::Header) state_ = State::Outside;
    return;
  }
  const std::string tail = detok_flush();
  if (!tail.empty()) route(tail, out);
  if (stopped_by_string_) return;
  if (kind_ == Kind::Text && !hold_.empty()) {
    emit_content(hold_, out);
    hold_.clear();
  } else if (kind_ == Kind::ToolJson && tool_) {
    close_tool(out);
  }
  state_ = State::Outside;
  kind_ = Kind::None;
}

void StreamParser::route(const std::string& text, std::vector<Delta>& out) {
  if (kind_ == Kind::Thinking) {
    reasoning_ += text;
    out.push_back(Delta{.reasoning_content = text});
  } else if (kind_ == Kind::Text) {
    this->text(text, out);
  } else if (kind_ == Kind::ToolJson) {
    tool_text(text, out);
  }
}

void StreamParser::emit_content(std::string_view text, std::vector<Delta>& out) {
  if (text.empty()) return;
  content_ += text;
  out.push_back(Delta{.content = std::string(text)});
}

void StreamParser::text(const std::string& text, std::vector<Delta>& out) {
  if (stop_.empty()) {
    emit_content(text, out);
    return;
  }
  const std::string buf = hold_ + text;
  std::size_t first = std::string::npos;
  for (const std::string& s : stop_) first = std::min(first, buf.find(s));
  if (first != std::string::npos) {
    emit_content(std::string_view(buf).substr(0, first), out);
    hold_.clear();
    stopped_by_string_ = true;
    return;
  }
  const std::size_t len = uni::utf8_length(buf);
  const std::size_t keep = std::min(hold_len_, len);
  const std::size_t cut = uni::utf8_offset(buf, len - keep);
  emit_content(std::string_view(buf).substr(0, cut), out);
  hold_ = buf.substr(cut);
}

void StreamParser::announce(ToolState& t, std::vector<Delta>& out) {
  tool_calls_.push_back(ToolCall{t.id, t.name.value_or(""), ""});
  out.push_back(Delta{.tool_call = ToolCallDelta{t.index, t.id, t.name, ""}});
}

void StreamParser::tool_args(const ToolState& t, const std::string& args, std::vector<Delta>& out) {
  if (args.empty()) return;
  tool_calls_[static_cast<std::size_t>(t.index)].arguments += args;
  out.push_back(Delta{.tool_call = ToolCallDelta{t.index, std::nullopt, std::nullopt, args}});
}

void StreamParser::tool_text(const std::string& text, std::vector<Delta>& out) {
  ToolState& t = *tool_;
  t.body += text;
  if (!t.args_start) {
    const auto m = match_tool_prefix(t.body);
    if (!m) return;
    std::string name;
    try {
      const Json loaded = py::json_loads(m->name_literal);
      name = loaded.is_string() ? loaded.get<std::string>() : std::string();
    } catch (const py::JsonDecodeError&) {
      name.clear();
    }
    if (name.empty()) name = t.header;
    t.name = name;
    if (name.empty()) return;
    t.args_start = m->end;
    announce(t, out);
  }
  const std::string cand = args_candidate(std::string_view(t.body).substr(*t.args_start));
  if (cand.size() > t.emitted.size() && cand.starts_with(t.emitted)) {
    tool_args(t, cand.substr(t.emitted.size()), out);
    t.emitted = cand;
  }
}

void StreamParser::close_tool(std::vector<Delta>& out) {
  ToolState t = std::move(*tool_);
  tool_.reset();
  if (t.args_start) {
    const std::string cand = args_candidate(std::string_view(t.body).substr(*t.args_start));
    if (cand.size() > t.emitted.size() && cand.starts_with(t.emitted)) {
      tool_args(t, cand.substr(t.emitted.size()), out);
    }
    return;
  }
  // The body never matched the canonical prefix: parse it whole.
  std::string name = t.header;
  std::string arguments(uni::py_strip(t.body));
  std::optional<Json> obj;
  try {
    obj = py::json_loads(t.body);
  } catch (const py::JsonDecodeError&) {
    obj.reset();
  }
  if (obj && obj->is_object()) {
    const Json* n = get(*obj, "name");
    if (n != nullptr && n->is_string() && !n->get_ref<const std::string&>().empty()) name = n->get<std::string>();
    const Json* args = get(*obj, "args");
    arguments = py::json_dumps(args != nullptr ? *args : Json::object());
  }
  if (name.empty()) return;
  t.name = name;
  announce(t, out);
  tool_args(t, arguments, out);
}

}  // namespace ingine::text
