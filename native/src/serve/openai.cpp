#include "serve/openai.h"

#include <algorithm>
#include <array>
#include <climits>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <limits>
#include <sstream>
#include <string>

namespace ingine::serve {
namespace {

struct Bound {
  double value;
  const char* text;  // Python's str() of the bound, as printed in the error message
};

const Json* get(const Json& body, const char* key) {
  auto it = body.find(key);
  return it == body.end() || it->is_null() ? nullptr : &*it;
}

std::string range_message(const char* key, const std::optional<Bound>& lo, const std::optional<Bound>& hi) {
  return std::string("'") + key + "' must be in [" + (lo ? lo->text : "None") + ", " + (hi ? hi->text : "None") +
         "]";
}

int64_t saturate(double v) {
  if (v >= 9.2233720368547758e18) return std::numeric_limits<int64_t>::max();
  if (v <= -9.2233720368547758e18) return std::numeric_limits<int64_t>::min();
  return static_cast<int64_t>(v);
}

// server._num(body, key, int, lo, hi)
std::optional<int64_t> num_int(const Json& body, const char* key, std::optional<Bound> lo = std::nullopt,
                               std::optional<Bound> hi = std::nullopt) {
  const Json* v = get(body, key);
  if (!v) return std::nullopt;
  if (v->is_boolean() || !v->is_number() ||
      (v->is_number_float() && std::trunc(v->get<double>()) != v->get<double>())) {
    throw BadRequest(std::string("'") + key + "' must be an integer", key);
  }
  int64_t value;
  if (v->is_number_unsigned()) {
    value = int64_t(std::min<uint64_t>(v->get<uint64_t>(), uint64_t(std::numeric_limits<int64_t>::max())));
  } else if (v->is_number_integer()) {
    value = v->get<int64_t>();
  } else {
    value = saturate(v->get<double>());
  }
  if ((lo && double(value) < lo->value) || (hi && double(value) > hi->value)) {
    throw BadRequest(range_message(key, lo, hi), key);
  }
  return value;
}

// server._num(body, key, float, lo, hi)
std::optional<double> num_float(const Json& body, const char* key, std::optional<Bound> lo, std::optional<Bound> hi) {
  const Json* v = get(body, key);
  if (!v) return std::nullopt;
  if (v->is_boolean() || !v->is_number()) throw BadRequest(std::string("'") + key + "' must be a number", key);
  const double value = v->get<double>();
  if ((lo && value < lo->value) || (hi && value > hi->value)) throw BadRequest(range_message(key, lo, hi), key);
  return value;
}

int clamp_int(int64_t v) { return int(std::clamp<int64_t>(v, INT_MIN, INT_MAX)); }

}  // namespace

bool truthy(const Json& value) {
  switch (value.type()) {
    case Json::value_t::null: return false;
    case Json::value_t::boolean: return value.get<bool>();
    case Json::value_t::number_integer: return value.get<int64_t>() != 0;
    case Json::value_t::number_unsigned: return value.get<uint64_t>() != 0;
    case Json::value_t::number_float: return value.get<double>() != 0.0;
    case Json::value_t::string: return !value.get_ref<const std::string&>().empty();
    case Json::value_t::array:
    case Json::value_t::object: return !value.empty();
    default: return true;
  }
}

constexpr int32_t kImagePlaceholderId = 200054;  // <|unused_200054|>

std::vector<uint8_t> decode_base64(std::string_view b64) {
  static constexpr auto kTable = []() {
    std::array<int, 256> t{};
    t.fill(-1);
    for (int i = 0; i < 26; ++i) t[static_cast<unsigned char>('A' + i)] = i;
    for (int i = 0; i < 26; ++i) t[static_cast<unsigned char>('a' + i)] = 26 + i;
    for (int i = 0; i < 10; ++i) t[static_cast<unsigned char>('0' + i)] = 52 + i;
    t[static_cast<unsigned char>('+')] = 62;
    t[static_cast<unsigned char>('/')] = 63;
    return t;
  }();
  std::vector<uint8_t> out;
  out.reserve((b64.size() * 3) / 4);
  int val = 0, valb = -8;
  for (unsigned char c : b64) {
    if (c == '=' || c == '\r' || c == '\n' || c == ' ') continue;
    const int d = kTable[c];
    if (d == -1) throw BadRequest("invalid base64 image payload", "messages");
    val = (val << 6) + d;
    valb += 6;
    if (valb >= 0) {
      out.push_back(static_cast<uint8_t>((val >> valb) & 0xff));
      valb -= 8;
    }
  }
  return out;
}

std::vector<uint8_t> load_image_bytes(std::string_view url) {
  if (url.rfind("data:", 0) == 0) {
    const std::size_t comma = url.find(',');
    if (comma == std::string_view::npos) throw BadRequest("malformed data: image URL", "messages");
    const std::string_view meta = url.substr(0, comma);
    if (meta.find(";base64") == std::string_view::npos) {
      throw BadRequest("only base64 data: image URLs are supported", "messages");
    }
    return decode_base64(url.substr(comma + 1));
  }
  if (url.rfind("file://", 0) == 0 || (!url.empty() && url[0] == '/')) {
    const std::string path(url.rfind("file://", 0) == 0 ? url.substr(7) : url);
    std::ifstream f(path, std::ios::binary | std::ios::ate);
    if (!f) throw BadRequest("cannot open image file: " + path, "messages");
    const auto sz = f.tellg();
    f.seekg(0, std::ios::beg);
    std::vector<uint8_t> out(static_cast<std::size_t>(sz));
    f.read(reinterpret_cast<char*>(out.data()), sz);
    return out;
  }
  throw BadRequest("image_url must be a data:image/...;base64,... URL", "messages");
}

std::vector<std::string> extract_image_urls(const Json& messages) {
  std::vector<std::string> urls;
  if (!messages.is_array()) return urls;
  for (const auto& msg : messages) {
    if (!msg.is_object()) continue;
    const auto it = msg.find("content");
    if (it == msg.end() || !it->is_array()) continue;
    for (const auto& part : *it) {
      if (!part.is_object()) continue;
      const std::string type = part.contains("type") && part["type"].is_string() ? part["type"].get<std::string>() : "";
      if (type != "image_url" && type != "image" && type != "input_image") continue;
      if (part.contains("image_url")) {
        const auto& iu = part["image_url"];
        if (iu.is_string()) urls.push_back(iu.get<std::string>());
        else if (iu.is_object() && iu.contains("url") && iu["url"].is_string()) urls.push_back(iu["url"].get<std::string>());
      } else if (part.contains("url") && part["url"].is_string()) {
        urls.push_back(part["url"].get<std::string>());
      } else if (part.contains("image") && part["image"].is_string()) {
        urls.push_back(part["image"].get<std::string>());
      }
    }
  }
  return urls;
}

ChatRequest parse_request(const Json& body, const text::ChatFormat& fmt, int max_model_len,
                          const model::Sampling& defaults, const vision::VisionEncoder* vision) {
  if (num_int(body, "n", Bound{1, "1"}, Bound{1, "1"}).value_or(1) != 1) {
    throw BadRequest("only n=1 is supported", "n");
  }
  Json kwargs = Json::object();
  if (auto it = body.find("chat_template_kwargs"); it != body.end() && truthy(*it)) {
    if (!it->is_object()) {
      throw BadRequest("'chat_template_kwargs' must be an object", "chat_template_kwargs");
    }
    kwargs = *it;
  }
  // body.get("reasoning_effort", kwargs.pop("reasoning_effort", None))
  Json effort = kwargs.contains("reasoning_effort") ? kwargs["reasoning_effort"] : Json();
  kwargs.erase("reasoning_effort");
  if (auto it = body.find("reasoning_effort"); it != body.end()) effort = *it;
  // kwargs.pop("enable_thinking", body.get("enable_thinking", True))
  Json enable_thinking = true;
  if (auto it = body.find("enable_thinking"); it != body.end()) enable_thinking = *it;
  if (kwargs.contains("enable_thinking")) {
    enable_thinking = kwargs["enable_thinking"];
    kwargs.erase("enable_thinking");
  }
  if (!enable_thinking.is_boolean()) throw BadRequest("'enable_thinking' must be a boolean", "enable_thinking");
  if (effort.is_string()) {
    const std::string& eff_str = effort.get_ref<const std::string&>();
    if (eff_str == "default" || eff_str.empty()) {
      effort = Json();
    } else if (eff_str == "off") {
      enable_thinking = false;
      effort = Json("none");
    }
  }
  bool reasoning_from_default = false;
  if (effort.is_null() && enable_thinking.get<bool>()) {
    effort = Json("medium");
    reasoning_from_default = true;
  }

  text::ChatOptions options;
  options.reasoning_effort = std::move(effort);
  options.enable_thinking = enable_thinking.get<bool>();
  options.template_kwargs = std::move(kwargs);
  const auto messages = body.find("messages");
  const auto tools = body.find("tools");
  static const Json kAbsent;  // null: no messages/tools key (passed by reference, not a deep copy of the body)
  text::RenderedPrompt prompt = fmt.render(messages == body.end() ? kAbsent : *messages,
                                           tools == body.end() ? kAbsent : *tools, options);

  std::vector<model::ImageSpan> image_spans;
  if (messages != body.end()) {
    const std::vector<std::string> image_urls = extract_image_urls(*messages);
    if (!image_urls.empty()) {
      if (!vision) throw BadRequest("vision encoder is not enabled on this server", "messages");
      std::vector<vision::ImageEmbeddings> encoded;
      encoded.reserve(image_urls.size());
      for (const std::string& url : image_urls) {
        const std::vector<uint8_t> bytes = load_image_bytes(url);
        encoded.push_back(vision->encode_image_bytes(bytes, 256));
      }
      std::vector<int32_t> expanded;
      expanded.reserve(prompt.ids.size() + encoded.size() * 256);
      std::size_t img_idx = 0;
      for (int32_t id : prompt.ids) {
        if (id == kImagePlaceholderId && img_idx < encoded.size()) {
          auto& emb = encoded[img_idx++];
          const int offset = int(expanded.size());
          expanded.insert(expanded.end(), std::size_t(emb.n_tokens), kImagePlaceholderId);
          image_spans.push_back(model::ImageSpan{offset, emb.n_tokens, std::move(emb.data)});
        } else {
          expanded.push_back(id);
        }
      }
      prompt.ids = std::move(expanded);
    }
  }

  const int64_t room = int64_t(max_model_len) - int64_t(prompt.ids.size());
  if (room <= 0) {
    throw BadRequest("prompt has " + std::to_string(prompt.ids.size()) + " tokens; the model context is " +
                         std::to_string(max_model_len),
                     "messages");
  }
  std::optional<int64_t> max_tokens;
  if (const Json* mct = get(body, "max_completion_tokens");
      !(mct && mct->is_number_integer() && mct->get<int64_t>() == -1)) {
    max_tokens = num_int(body, "max_completion_tokens", Bound{1, "1"});
  }
  if (!max_tokens) {
    if (const Json* mt = get(body, "max_tokens"); !(mt && mt->is_number_integer() && mt->get<int64_t>() == -1)) {
      max_tokens = num_int(body, "max_tokens", Bound{1, "1"});
    }
  }
  const Json* np = get(body, "n_predict");
  const bool pre_encode = (np && np->is_number_integer() && np->get<int64_t>() == 0);
  if (!max_tokens && np && np->is_number_integer() && np->get<int64_t>() > 0) {
    max_tokens = np->get<int64_t>();
  }

  ChatRequest req;
  req.images = std::move(image_spans);
  if (!enable_thinking.get<bool>()) {
    req.reasoning_effort = "none (0.0)";
    req.reasoning_source = "harness";
  } else if (options.reasoning_effort.is_string()) {
    const std::string& e = options.reasoning_effort.get_ref<const std::string&>();
    if (e == "none") req.reasoning_effort = "none (0.0)";
    else if (e == "minimal") req.reasoning_effort = "minimal (0.1)";
    else if (e == "low") req.reasoning_effort = "low (0.2)";
    else if (e == "medium") req.reasoning_effort = "medium (0.7)";
    else if (e == "high") req.reasoning_effort = "high (0.9)";
    else if (e == "xhigh") req.reasoning_effort = "xhigh (0.99)";
    else if (e == "max") req.reasoning_effort = "max (0.99)";
    else req.reasoning_effort = e;
    req.reasoning_source = reasoning_from_default ? "default" : "harness";
  } else if (options.reasoning_effort.is_number()) {
    std::ostringstream oss;
    oss << options.reasoning_effort.get<double>();
    req.reasoning_effort = oss.str();
    req.reasoning_source = "harness";
  }
  req.sampling.temperature =
      float(num_float(body, "temperature", Bound{0.0, "0.0"}, Bound{2.0, "2.0"}).value_or(defaults.temperature));
  req.sampling.top_p = float(num_float(body, "top_p", Bound{0.0, "0.0"}, Bound{1.0, "1.0"}).value_or(defaults.top_p));
  req.sampling.top_k =
      clamp_int(std::max<int64_t>(0, num_int(body, "top_k", Bound{-1, "-1"}).value_or(defaults.top_k)));
  if (auto seed = num_int(body, "seed")) req.sampling.seed = static_cast<uint64_t>(*seed);
  if (req.sampling.top_p <= 0.0f) throw BadRequest("'top_p' must be in (0, 1]", "top_p");

  if (const Json* stop = get(body, "stop")) {
    if (stop->is_string()) {
      req.stop.push_back(stop->get<std::string>());
    } else if (stop->is_array() &&
               std::all_of(stop->begin(), stop->end(), [](const Json& s) { return s.is_string(); })) {
      for (const Json& s : *stop) req.stop.push_back(s.get<std::string>());
    } else {
      throw BadRequest("'stop' must be a string or a list of strings", "stop");
    }
  }
  if (auto it = body.find("stream"); it != body.end()) {
    if (!it->is_boolean()) throw BadRequest("'stream' must be a boolean", "stream");
    req.stream = it->get<bool>();
  }
  if (auto it = body.find("stream_options"); it != body.end() && truthy(*it)) {
    if (!it->is_object()) throw BadRequest("'stream_options' must be an object", "stream_options");
    if (auto u = it->find("include_usage"); u != it->end()) req.include_usage = truthy(*u);
  }
  req.max_new_tokens = pre_encode ? 0 : int(std::min(max_tokens.value_or(room), room));
  req.prompt_ids = std::move(prompt.ids);
  req.prefill_ids = std::move(prompt.prefill_ids);
  return req;
}

Json error_body(int status, std::string_view message, const std::optional<std::string>& param) {
  const char* kind = status == 400 ? "invalid_request_error"
                     : status == 404 ? "not_found_error"
                     : status == 503 ? "server_busy"
                                     : "server_error";
  return Json{{"error",
               {{"message", message}, {"type", kind}, {"param", param ? Json(*param) : Json()}, {"code", status}}}};
}

Json delta_json(const text::Delta& d) {
  if (!d.reasoning_content.empty()) return Json{{"reasoning_content", d.reasoning_content}};
  if (!d.content.empty()) return Json{{"content", d.content}};
  const text::ToolCallDelta& tc = *d.tool_call;
  if (tc.id) {
    return Json{{"tool_calls",
                 Json::array({Json{{"index", tc.index},
                                   {"id", *tc.id},
                                   {"type", "function"},
                                   {"function", {{"name", tc.name ? Json(*tc.name) : Json()},
                                                 {"arguments", tc.arguments}}}}})}};
  }
  return Json{{"tool_calls", Json::array({Json{{"index", tc.index}, {"function", {{"arguments", tc.arguments}}}}})}};
}

std::string dump(const Json& value) { return value.dump(-1, ' ', false, Json::error_handler_t::replace); }

}  // namespace ingine::serve
