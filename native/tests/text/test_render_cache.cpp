// Render cache (incremental tokenisation): every render through a warm ChatFormat must equal a cold render of
// the same request (text, ids, prefill ids, or the same ChatError). Covers growing conversations (message
// prefixes in order), edited last messages (shared prefix ending at, just after and inside special tokens, so
// the boundary rule b + max_added - 1 <= p decides), repeated renders, conversations interleaved round-robin
// (2, and more than the cache holds), and concurrent renders from several threads.
#include "text/chat.h"
#include "text/unicode.h"
#include "text_test.h"

#include <atomic>
#include <thread>

using namespace ingine::text;
using Json = nlohmann::ordered_json;

namespace {

struct Request {
  std::string label;
  Json messages, tools;
  ChatOptions options;
};

struct Output {
  std::string text;
  std::vector<int32_t> ids, prefill;
  std::string error;
  bool operator==(const Output&) const = default;
};

Output render(const ChatFormat& fmt, const Request& r) {
  Output o;
  try {
    RenderedPrompt p = fmt.render(r.messages, r.tools, r.options);
    o.text = std::move(p.text);
    o.ids = std::move(p.ids);
    o.prefill = std::move(p.prefill_ids);
  } catch (const ChatError& e) {
    o.error = std::string(e.what()) + " [" + e.param + "]";
  }
  return o;
}

Output cold_render(const ChatFormat& cold, const Request& r) {
  cold.clear_render_cache();
  return render(cold, r);
}

// A conversation of tool use, in the shape the agent sends: system, user, then assistant turns with reasoning
// and tool calls answered by tool messages (named by tool_call_id), some text containing special-token text.
Json conversation(int seed, int turns) {
  Json m = Json::array();
  m.push_back({{"role", "system"}, {"content", "You are agent #" + std::to_string(seed) + ". Use the tools."}});
  m.push_back({{"role", "user"}, {"content", "Task " + std::to_string(seed) + ": inspect the repository, fix the bug "
                                                 "and report — résumé 😀 中文."}});
  for (int t = 0; t < turns; ++t) {
    const std::string id = "call_" + std::to_string(seed) + "_" + std::to_string(t);
    Json call = {{"id", id},
                 {"type", "function"},
                 {"function",
                  {{"name", t % 2 ? "search" : "get_weather"},
                   {"arguments", t % 2 ? "{\"q\": \"step " + std::to_string(t) + "\", \"n\": " + std::to_string(t) + "}"
                                       : "{\"city\": \"Paris\", \"unit\": \"c\"}"}}}};
    m.push_back({{"role", "assistant"},
                 {"content", t % 3 == 0 ? "" : "Checking step " + std::to_string(t) + "."},
                 {"reasoning_content", "Thinking about step " + std::to_string(t) + " of task " + std::to_string(seed)},
                 {"tool_calls", Json::array({call})}});
    std::string result = "result " + std::to_string(t) + ":\n";
    for (int k = 0; k < 6 + t; ++k) result += "  line " + std::to_string(k) + " of output\n";
    if (t % 4 == 1) result += "literal <|end_message|> and <|message_user|> inside a tool result\n";
    m.push_back({{"role", "tool"}, {"tool_call_id", id}, {"content", result}});
    if (t % 5 == 4) m.push_back({{"role", "user"}, {"content", "Continue with step " + std::to_string(t + 1)}});
  }
  m.push_back({{"role", "assistant"}, {"content", "Done for task " + std::to_string(seed) + "."}});
  return m;
}

Json tools_list() {
  return Json::parse(R"([
    {"type": "function", "function": {"name": "get_weather", "description": "Weather for a city.",
      "parameters": {"type": "object", "properties": {"city": {"type": "string"}, "unit": {"type": "string"}},
                     "required": ["city"]}}},
    {"type": "function", "function": {"name": "search", "description": "Search the web.",
      "parameters": {"type": "object", "properties": {"q": {"type": "string"}, "n": {"type": "integer"}}}}}])");
}

// The requests a client sends during a conversation: messages[:k] for k = 1..n.
std::vector<Request> prefixes(const std::string& label, const Json& messages, const Json& tools,
                              const ChatOptions& options) {
  std::vector<Request> out;
  for (std::size_t k = 1; k <= messages.size(); ++k) {
    Json m = Json::array();
    for (std::size_t i = 0; i < k; ++i) m.push_back(messages[i]);
    out.push_back({label + "[:" + std::to_string(k) + "]", std::move(m), tools, options});
  }
  return out;
}

// Variants of a request with its last message's content edited, so the shared rendered prefix ends at a special
// token boundary, just after it, inside the content, or where an edit completes / breaks a special token.
std::vector<Request> edits(const Request& r) {
  std::vector<Request> out;
  const Json& last = r.messages.back();
  if (!last.contains("content") || !last["content"].is_string()) return out;
  const std::string content = last["content"].get<std::string>();
  // Cut points on character boundaries (the request text must stay valid UTF-8).
  const std::size_t chars = unicode::utf8_length(content);
  const auto at = [&](std::size_t ch) { return unicode::utf8_offset(content, ch); };
  const auto with = [&](const std::string& note, std::string text) {
    Request e = r;
    e.label += " edit:" + note;
    e.messages.back()["content"] = std::move(text);
    out.push_back(std::move(e));
  };
  with("append", content + " (more)");
  with("first char", "X" + content.substr(at(1)));  // differs right after <|content_text|>
  with("middle", content.substr(0, at(chars / 2)) + "~" + content.substr(at(chars / 2)));
  with("truncate", content.substr(0, at(chars / 3)));
  with("partial token", content + "<|end_mess");
  with("complete token", content + "<|end_message|>tail");
  with("empty", "");
  return out;
}

}  // namespace

int main() {
  const auto dir = test::model_dir();
  if (dir.empty()) return test::kSkip;
  const ChatFormat warm = ChatFormat::load(dir);
  const ChatFormat cold = ChatFormat::load(dir);  // separate cache, cleared before every render
  test::Checker c;
  int renders = 0;

  const auto check = [&](const Request& r, const Output& want) {
    const Output got = render(warm, r);
    ++renders;
    if (!c.expect(got == want, r.label + ": warm render differs from cold")) {
      std::fprintf(stderr, "  text %s, ids %s, prefill %s, error %s|%s\n", got.text == want.text ? "ok" : "DIFF",
                   got.ids == want.ids ? "ok" : "DIFF", got.prefill == want.prefill ? "ok" : "DIFF",
                   got.error.c_str(), want.error.c_str());
    }
  };
  const auto check_cold = [&](const Request& r) { check(r, cold_render(cold, r)); };

  // 1. Golden chat requests (tests/golden/text/chat.json): the message prefixes of every request in order, a
  //    repeated render, and edits of the last message.
  for (const Json& g : test::load_golden("chat.json")) {
    if (!g["messages"].is_array() || g["messages"].empty()) continue;
    ChatOptions options{g["reasoning_effort"], g["enable_thinking"], g["kwargs"]};
    warm.clear_render_cache();
    for (const Request& r : prefixes(g["note"].get<std::string>(), g["messages"], g["tools"], options)) {
      const Output want = cold_render(cold, r);
      check(r, want);
      check(r, want);  // identical text: full cache hit
      for (const Request& e : edits(r)) {
        check_cold(e);
        check(r, want);  // back to the unedited request
      }
    }
  }

  // 2. Long tool-use conversations: growth in order, with edits, effort levels and the no-thinking prefill.
  std::vector<std::vector<Request>> convs;
  for (int s = 0; s < 6; ++s) {
    ChatOptions options;
    options.reasoning_effort = s % 3 == 0 ? Json() : Json(s % 3 == 1 ? "low" : "high");
    options.enable_thinking = s % 2 == 0;
    convs.push_back(prefixes("conv" + std::to_string(s), conversation(s, 8 + s), tools_list(), options));
  }
  std::vector<std::vector<Output>> want(convs.size());
  for (std::size_t s = 0; s < convs.size(); ++s) {
    for (const Request& r : convs[s]) want[s].push_back(cold_render(cold, r));
  }
  for (std::size_t s = 0; s < convs.size(); ++s) {
    warm.clear_render_cache();
    for (std::size_t k = 0; k < convs[s].size(); ++k) {
      check(convs[s][k], want[s][k]);
      if (k % 3 == 2) {
        for (const Request& e : edits(convs[s][k])) check_cold(e);
      }
    }
  }

  // 3. Interleaved round-robin: two conversations (both stay cached), then all six (more than the cache holds,
  //    so entries are evicted and conversations come back cold).
  for (const std::size_t n : {std::size_t{2}, convs.size()}) {
    warm.clear_render_cache();
    for (std::size_t k = 0;; ++k) {
      bool any = false;
      for (std::size_t s = 0; s < n; ++s) {
        if (k >= convs[s].size()) continue;
        any = true;
        check(convs[s][k], want[s][k]);
      }
      if (!any) break;
    }
  }

  // 4. Concurrent: each thread replays one conversation (in order, with a repeat) three times against the
  //    shared cache, while the others do the same.
  std::atomic<int> mismatches{0}, concurrent{0};
  warm.clear_render_cache();
  std::vector<std::thread> threads;
  for (std::size_t t = 0; t < 8; ++t) {
    threads.emplace_back([&, t] {
      const std::size_t s = t % convs.size();
      for (int round = 0; round < 3; ++round) {
        for (std::size_t k = 0; k < convs[s].size(); ++k) {
          for (int rep = 0; rep < 2; ++rep) {
            if (!(render(warm, convs[s][k]) == want[s][k])) ++mismatches;
            ++concurrent;
          }
        }
      }
    });
  }
  for (auto& th : threads) th.join();
  c.expect(mismatches.load() == 0, "concurrent warm renders differ from cold: " + std::to_string(mismatches.load()));

  std::printf("%d sequential and %d concurrent renders compared with cold renders\n", renders, concurrent.load());
  return c.finish("text_render_cache");
}
