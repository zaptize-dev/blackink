// OpenAI-compatible server end to end against FakeGenerator (CPU only, real tokenizer/template): port of
// engine/tests/test_server.py.
#include "engine/fake_generator.h"
#include "serve/chat_server.h"
#include "text/chat.h"

#include <httplib.h>
#include <nlohmann/json.hpp>

#include <arpa/inet.h>
#include <netinet/in.h>
#include <sys/socket.h>
#include <unistd.h>

#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <functional>
#include <map>
#include <memory>
#include <mutex>
#include <optional>
#include <set>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

namespace {

using Json = nlohmann::ordered_json;
using ingine::engine::FakeGenerator;
using ingine::serve::ChatServer;
using ingine::text::ChatFormat;

int g_failures = 0;

#define CHECK(cond)                                                              \
  do {                                                                           \
    if (!(cond)) {                                                               \
      std::fprintf(stderr, "%s:%d: check failed: %s\n", __FILE__, __LINE__, #cond); \
      ++g_failures;                                                              \
      throw std::runtime_error("check failed");                                  \
    }                                                                            \
  } while (0)

const Json MSGS = Json::array({{{"role", "user"}, {"content", "Hi 🎉"}}});
const std::string THINK_TEXT =
    "<|message_model|><|content_thinking|>Let me think… 𓀀 ok<|end_message|>"
    "<|message_model|><|content_text|>Hello, wörld 𝔘! STOP here<|end_message|>"
    "<|content_model_end_sampling|>";
const std::string TOOL_TEXT =
    "<|message_model|><|content_thinking|>Need the weather.<|end_message|>"
    "<|message_model|>get_weather<|content_invoke_tool_json|>{\"name\":\"get_weather\",\"args\":"
    "{\"city\":\"São Paulo\",\"days\":3,\"opts\":{\"units\":\"metric\"}}}<|end_message|>"
    "<|message_model|>search<|content_invoke_tool_json|>{\"name\":\"search\",\"args\":{\"q\":\"rain\"}}"
    "<|end_message|><|content_model_end_sampling|>";
std::string long_text() {
  std::string s = "<|message_model|><|content_text|>";
  for (int i = 0; i < 3000; ++i) s += "word ";
  return s + "<|end_message|><|content_model_end_sampling|>";
}
const Json TOOLS = Json::array(
    {{{"type", "function"}, {"function", {{"name", "get_weather"}, {"parameters", {{"type", "object"}}}}}},
     {{"type", "function"}, {"function", {{"name", "search"}, {"parameters", {{"type", "object"}}}}}}});
const std::set<std::string> FINISH = {"stop", "length", "tool_calls"};

const ChatFormat* g_fmt = nullptr;

ingine::text::RenderedPrompt render(const Json& messages, Json effort = Json(), bool enable_thinking = true) {
  ingine::text::ChatOptions options;
  options.reasoning_effort = std::move(effort);
  options.enable_thinking = enable_thinking;
  return g_fmt->render(messages, Json(), options);
}

struct Served {
  std::unique_ptr<FakeGenerator> gen;
  std::unique_ptr<ChatServer> srv;
  int port() const { return srv->port(); }
  ~Served() {
    if (srv) srv->stop();
  }
};

std::unique_ptr<Served> serve(std::string text = THINK_TEXT, double delay = 0.0, int max_queue = 4,
                              int max_model_len = 0, FakeGenerator::Responder responder = {},
                              ingine::model::Sampling sampling = {}) {
  auto s = std::make_unique<Served>();
  if (!responder) responder = [text](std::span<const int32_t>) { return FakeGenerator::Output(text); };
  s->gen = std::make_unique<FakeGenerator>(g_fmt->tokenizer(), "inkling-test", 1 << 20, responder, delay);
  ingine::serve::ServerOptions options;
  options.max_queue = max_queue;
  options.max_model_len = max_model_len;
  options.sampling = sampling;
  s->srv = std::make_unique<ChatServer>(*s->gen, *g_fmt, options);
  if (!s->srv->bind("127.0.0.1", 0)) throw std::runtime_error("bind failed");
  s->srv->start();
  return s;
}

struct Response {
  int status = 0;
  httplib::Headers headers;
  std::string body;
};

Response request(int port, const std::string& method, const std::string& path, const std::string& body = "") {
  httplib::Client cli("127.0.0.1", port);
  cli.set_read_timeout(20, 0);
  httplib::Result r = method == "GET" ? cli.Get(path) : cli.Post(path, body, "application/json");
  if (!r) throw std::runtime_error("request failed: " + httplib::to_string(r.error()));
  return {r->status, r->headers, r->body};
}

std::vector<Json> parse_sse(const std::string& data) {
  std::vector<Json> events;
  size_t pos = 0;
  while (pos < data.size()) {
    size_t end = data.find("\n\n", pos);
    if (end == std::string::npos) end = data.size();
    const std::string block = data.substr(pos, end - pos);
    pos = end + 2;
    if (block.find_first_not_of(" \t\r\n") == std::string::npos || block.starts_with(":")) continue;
    CHECK(block.starts_with("data: "));
    const std::string payload = block.substr(6);
    events.push_back(payload == "[DONE]" ? Json("[DONE]") : Json::parse(payload));
  }
  return events;
}

struct Chat {
  int status;
  httplib::Headers headers;
  Json body;  // response object, or the SSE events of a successful stream
};

Chat chat(int port, const Json& extra) {
  Json body{{"messages", MSGS}};
  for (auto& [k, v] : extra.items()) body[k] = v;
  Response r = request(port, "POST", "/v1/chat/completions", body.dump());
  const bool stream = extra.contains("stream") && extra["stream"] == true;
  return {r.status, r.headers, stream && r.status == 200 ? Json(parse_sse(r.body)) : Json::parse(r.body)};
}

std::string header(const httplib::Headers& h, const std::string& key) {
  auto it = h.find(key);
  return it == h.end() ? "" : it->second;
}

int raw_request(int port, const Json& extra) {
  Json body{{"messages", MSGS}};
  for (auto& [k, v] : extra.items()) body[k] = v;
  const std::string data = body.dump();
  const int s = ::socket(AF_INET, SOCK_STREAM, 0);
  sockaddr_in addr{};
  addr.sin_family = AF_INET;
  addr.sin_port = htons(uint16_t(port));
  addr.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
  if (::connect(s, reinterpret_cast<sockaddr*>(&addr), sizeof addr) != 0) throw std::runtime_error("connect failed");
  const std::string req = "POST /v1/chat/completions HTTP/1.1\r\nHost: x\r\nContent-Type: application/json\r\n"
                          "Content-Length: " +
                          std::to_string(data.size()) + "\r\n\r\n" + data;
  CHECK(::send(s, req.data(), req.size(), MSG_NOSIGNAL) == ssize_t(req.size()));
  return s;
}

std::pair<std::string, std::string> read_raw_response(int s) {
  timeval tv{10, 0};
  ::setsockopt(s, SOL_SOCKET, SO_RCVTIMEO, &tv, sizeof tv);
  std::string buf;
  char tmp[65536];
  while (buf.find("\r\n\r\n") == std::string::npos) {
    const ssize_t n = ::recv(s, tmp, sizeof tmp, 0);
    if (n <= 0) throw std::runtime_error("connection closed");
    buf.append(tmp, size_t(n));
  }
  const size_t split = buf.find("\r\n\r\n");
  std::string head = buf.substr(0, split), body = buf.substr(split + 4);
  std::string lower = head;
  for (char& c : lower) c = char(std::tolower(static_cast<unsigned char>(c)));
  const size_t cl = lower.find("content-length:");
  if (cl == std::string::npos) throw std::runtime_error("no content-length");
  const size_t length = std::stoul(head.substr(cl + 15));
  while (body.size() < length) {
    const ssize_t n = ::recv(s, tmp, sizeof tmp, 0);
    if (n <= 0) throw std::runtime_error("connection closed");
    body.append(tmp, size_t(n));
  }
  return {head, body};
}

void wait_for(const std::function<bool()>& cond, double timeout = 10.0) {
  const auto t0 = std::chrono::steady_clock::now();
  while (std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count() < timeout) {
    if (cond()) return;
    std::this_thread::sleep_for(std::chrono::milliseconds(10));
  }
  CHECK(!"condition not reached");
}

Json health(int port) { return Json::parse(request(port, "GET", "/health").body); }

// ---- OpenAI schema checks ---------------------------------------------------------------------------------

void check_usage(const Json& u) {
  CHECK(u.size() == 3 && u.contains("prompt_tokens") && u.contains("completion_tokens") && u.contains("total_tokens"));
  for (auto& [k, v] : u.items()) CHECK(v.is_number_integer() && v.get<int64_t>() >= 0);
  CHECK(u["total_tokens"] == u["prompt_tokens"].get<int>() + u["completion_tokens"].get<int>());
}

void check_completion(const Json& r) {
  CHECK(r["id"].is_string() && r["id"].get<std::string>().starts_with("chatcmpl-"));
  CHECK(r["object"] == "chat.completion" && r["created"].is_number_integer() && r["model"] == "inkling-test");
  CHECK(r["choices"].size() == 1);
  const Json& c = r["choices"][0];
  CHECK(c["index"] == 0 && FINISH.contains(c["finish_reason"].get<std::string>()) && c["logprobs"].is_null());
  const Json& m = c["message"];
  CHECK(m["role"] == "assistant");
  CHECK(m["content"].is_null() || m["content"].is_string());
  CHECK(m["reasoning_content"].is_null() || m["reasoning_content"].is_string());
  if (m.contains("tool_calls")) {
    CHECK(!m["tool_calls"].empty());
    for (const Json& tc : m["tool_calls"]) {
      CHECK(tc.size() == 3 && tc["type"] == "function" && tc["id"].is_string());
      CHECK(tc["function"].size() == 2 && tc["function"]["name"].is_string() &&
            tc["function"]["arguments"].is_string());
    }
  }
  check_usage(r["usage"]);
}

struct CallOut {
  std::string id, name, arguments;
  bool operator==(const CallOut&) const = default;
};
struct Stream {
  std::string reasoning, content;
  std::vector<CallOut> calls;
  std::string finish;
  Json usage;
};

// Validates a chunk stream and reassembles it.
Stream check_chunks(Json events, bool include_usage = false) {
  CHECK(events.back() == "[DONE]");
  Json chunks = Json::array();
  for (size_t i = 0; i + 1 < events.size(); ++i) chunks.push_back(events[i]);
  std::set<std::string> ids;
  for (const Json& c : chunks) ids.insert(c["id"].get<std::string>());
  CHECK(ids.size() == 1 && ids.begin()->starts_with("chatcmpl-"));
  Stream out;
  if (include_usage) {
    Json last = chunks.back();
    chunks.erase(chunks.size() - 1);
    CHECK(last["choices"] == Json::array() && last["object"] == "chat.completion.chunk");
    check_usage(last["usage"]);
    out.usage = last["usage"];
  }
  CHECK((chunks[0]["choices"][0]["delta"] == Json{{"role", "assistant"}, {"content", ""}}));
  std::map<int, CallOut> calls;
  bool finished = false;
  for (size_t i = 0; i < chunks.size(); ++i) {
    const Json& c = chunks[i];
    CHECK(c["object"] == "chat.completion.chunk" && c["created"].is_number_integer());
    CHECK(c["model"] == "inkling-test");
    CHECK(include_usage ? (c.contains("usage") && c["usage"].is_null()) : !c.contains("usage"));
    CHECK(c["choices"].size() == 1);
    const Json& ch = c["choices"][0];
    CHECK(ch["index"] == 0 && ch["logprobs"].is_null());
    const Json& d = ch["delta"];
    for (auto& [k, v] : d.items()) {
      CHECK(k == "role" || k == "content" || k == "reasoning_content" || k == "tool_calls");
    }
    if (!ch["finish_reason"].is_null()) {
      CHECK(i == chunks.size() - 1 && FINISH.contains(ch["finish_reason"].get<std::string>()) && d.empty());
      out.finish = ch["finish_reason"];
      finished = true;
      continue;
    }
    if (i > 0) CHECK(d.size() == 1 && !d.contains("role"));
    if (d.contains("reasoning_content")) out.reasoning += d["reasoning_content"].get<std::string>();
    if (d.contains("content") && d["content"].is_string()) out.content += d["content"].get<std::string>();
    if (d.contains("tool_calls")) {
      for (const Json& tc : d["tool_calls"]) {
        CHECK(tc["index"].is_number_integer());
        const int index = tc["index"];
        if (tc.contains("id")) {
          CHECK(!calls.contains(index) && tc["type"] == "function");
          CHECK(tc["function"]["name"].is_string() && !tc["function"]["name"].get<std::string>().empty());
          calls[index] = CallOut{tc["id"], tc["function"]["name"], ""};
        } else {
          CHECK(tc.size() == 2 && tc.contains("function") && tc["function"].size() == 1 &&
                tc["function"].contains("arguments"));
        }
        calls[index].arguments += tc["function"]["arguments"].get<std::string>();
      }
    }
  }
  CHECK(finished);
  CHECK((out.reasoning + out.content).find("\xEF\xBF\xBD") == std::string::npos);
  for (auto& [k, v] : calls) out.calls.push_back(v);
  return out;
}

int encoded_len(const std::string& text) { return int(g_fmt->tokenizer().encode(text, true).size()); }

// ---- tests ---------------------------------------------------------------------------------------------------

void test_models_and_health() {
  auto s = serve();
  Response r = request(s->port(), "GET", "/v1/models");
  Json body = Json::parse(r.body);
  CHECK(r.status == 200 && body["object"] == "list");
  CHECK(body["data"].size() == 1);
  const Json& m = body["data"][0];
  CHECK(m["id"] == "inkling-test" && m["object"] == "model" && m["created"].is_number_integer());
  CHECK(m["owned_by"].is_string() && m["max_model_len"] == (1 << 20));
  CHECK(Json::parse(request(s->port(), "GET", "/v1/models/inkling-test").body) == m);
  Json h = health(s->port());
  CHECK(h["status"] == "ok" && h["model"] == "inkling-test" && h["waiting"] == 0 && h["running"] == false &&
        h["max_queue"] == 4);
  CHECK(request(s->port(), "GET", "/health/").status == 200);
  Response nope = request(s->port(), "GET", "/nope");
  CHECK(nope.status == 404 && Json::parse(nope.body)["error"]["type"] == "not_found_error");
  CHECK(Json::parse(nope.body)["error"]["message"] == "no route for GET /nope");
  CHECK(request(s->port(), "POST", "/v1/completions", "{}").status == 404);
}

void test_non_streaming() {
  auto s = serve();
  Chat r = chat(s->port(), {{"model", "whatever"}});
  CHECK(r.status == 200);
  check_completion(r.body);
  const Json& m = r.body["choices"][0]["message"];
  CHECK(m["reasoning_content"] == "Let me think… 𓀀 ok");
  CHECK(m["content"] == "Hello, wörld 𝔘! STOP here" && !m.contains("tool_calls"));
  CHECK(m["refusal"].is_null() && r.body["system_fingerprint"].is_null());
  CHECK(r.body["choices"][0]["finish_reason"] == "stop");
  const auto calls = s->gen->calls();
  const auto ids = render(MSGS).ids;
  CHECK(r.body["usage"]["prompt_tokens"] == ids.size() && calls[0].prompt_ids.size() == ids.size());
  CHECK(r.body["usage"]["completion_tokens"] == encoded_len(THINK_TEXT));
  CHECK(calls[0].prompt_ids == ids);
}

void test_streaming_matches_non_streaming(bool include_usage) {
  auto s = serve();
  Chat ref = chat(s->port(), Json::object());
  Chat r = chat(s->port(), {{"stream", true}, {"stream_options", {{"include_usage", include_usage}}}});
  CHECK(r.status == 200 && header(r.headers, "Content-Type").starts_with("text/event-stream"));
  Stream st = check_chunks(r.body, include_usage);
  const Json& m = ref.body["choices"][0]["message"];
  CHECK(Json(st.reasoning) == m["reasoning_content"] && Json(st.content) == m["content"] && st.calls.empty() &&
        st.finish == "stop");
  if (include_usage) CHECK(st.usage == ref.body["usage"]);
}

void test_tool_call_streaming() {
  auto s = serve(TOOL_TEXT, 0.003);  // paced so deltas are not coalesced into one chunk
  Chat r = chat(s->port(), {{"tools", TOOLS}});
  check_completion(r.body);
  const Json& m = r.body["choices"][0]["message"];
  CHECK(r.body["choices"][0]["finish_reason"] == "tool_calls" && m["content"].is_null());
  CHECK(m["tool_calls"].size() == 2);
  CHECK(m["tool_calls"][0]["function"]["name"] == "get_weather");
  CHECK((Json::parse(m["tool_calls"][0]["function"]["arguments"].get<std::string>()) ==
         Json{{"city", "São Paulo"}, {"days", 3}, {"opts", {{"units", "metric"}}}}));
  CHECK(m["tool_calls"][1]["function"]["name"] == "search");
  CHECK((Json::parse(m["tool_calls"][1]["function"]["arguments"].get<std::string>()) == Json{{"q", "rain"}}));
  Chat sr = chat(s->port(), {{"tools", TOOLS}, {"stream", true}, {"stream_options", {{"include_usage", true}}}});
  Stream st = check_chunks(sr.body, true);
  CHECK(st.finish == "tool_calls" && st.reasoning == "Need the weather." && st.content.empty());
  CHECK(st.calls.size() == 2);
  for (size_t i = 0; i < 2; ++i) {
    CHECK(Json(st.calls[i].name) == m["tool_calls"][i]["function"]["name"]);
    CHECK(Json(st.calls[i].arguments) == m["tool_calls"][i]["function"]["arguments"]);
  }
  CHECK(st.calls[0].id != st.calls[1].id && st.usage == r.body["usage"]);
  int arg_chunks = 0;
  for (size_t i = 0; i + 2 < sr.body.size(); ++i) {
    const Json& d = sr.body[i]["choices"][0]["delta"];
    if (!d.contains("tool_calls")) continue;
    for (const Json& tc : d["tool_calls"]) arg_chunks += !tc.contains("id") && tc["index"] == 0;
  }
  CHECK(arg_chunks > 1);  // arguments really stream
}

void test_tool_round_trip_prompt() {
  Json msgs = MSGS;
  msgs.push_back({{"role", "assistant"},
                  {"content", nullptr},
                  {"reasoning_content", "r"},
                  {"tool_calls",
                   Json::array({{{"id", "call_a"},
                                 {"type", "function"},
                                 {"function", {{"name", "get_weather"}, {"arguments", "{\"city\":\"Paris\"}"}}}}})}});
  msgs.push_back({{"role", "tool"}, {"tool_call_id", "call_a"}, {"content", "sunny"}});
  auto s = serve();
  Chat r = chat(s->port(), {{"messages", msgs}, {"tools", TOOLS}});
  CHECK(r.status == 200);
  const std::string text = g_fmt->tokenizer().decode(s->gen->calls()[0].prompt_ids);
  CHECK(text.find("<|message_model|>get_weather<|content_invoke_tool_json|>{\"name\":\"get_weather\",\"args\":"
                  "{\"city\":\"Paris\"}}<|end_message|><|content_model_end_sampling|>"
                  "<|message_tool|>get_weather<|content_text|>sunny<|end_message|>") != std::string::npos);
}

void test_thinking_controls() {
  auto s = serve("Plain answer.<|end_message|><|content_model_end_sampling|>");
  for (const Json& body : {Json{{"chat_template_kwargs", {{"enable_thinking", false}}}}, Json{{"enable_thinking", false}}}) {
    Json req = body;
    req["reasoning_effort"] = 0.3;
    Chat r = chat(s->port(), req);
    CHECK(r.status == 200);
    const Json& m = r.body["choices"][0]["message"];
    CHECK(m["content"] == "Plain answer." && m["reasoning_content"].is_null());
    const auto want = render(MSGS, 0.3, false);
    CHECK(s->gen->calls().back().prompt_ids == want.ids);
    CHECK(r.body["usage"]["prompt_tokens"] == want.ids.size());
  }
  chat(s->port(), {{"reasoning_effort", "max"}});
  CHECK(s->gen->calls().back().prompt_ids == render(MSGS, "max").ids);
  chat(s->port(), {{"chat_template_kwargs", {{"reasoning_effort", "low"}}}});
  CHECK(s->gen->calls().back().prompt_ids == render(MSGS, "low").ids);
}

void test_sampling_and_max_tokens() {
  auto s = serve();
  Chat r = chat(s->port(), {{"temperature", 0.6}, {"top_p", 0.95}, {"top_k", 20}, {"seed", 7}, {"max_tokens", 5}});
  CHECK(r.status == 200 && r.body["choices"][0]["finish_reason"] == "length");
  CHECK(r.body["usage"]["completion_tokens"] == 5);
  auto call = s->gen->calls().back();
  CHECK(call.sampling.temperature == 0.6f && call.sampling.top_p == 0.95f && call.sampling.top_k == 20 &&
        call.sampling.seed == uint64_t(7));
  CHECK(call.max_new_tokens == 5);
  chat(s->port(), {{"max_completion_tokens", 3}, {"max_tokens", 9}, {"top_k", -1}});
  call = s->gen->calls().back();
  CHECK(call.max_new_tokens == 3 && call.sampling.top_k == 0 && !call.sampling.seed);
  CHECK(call.sampling.temperature == 1.0f && call.sampling.top_p == 1.0f);
  Chat sr = chat(s->port(), {{"stream", true}, {"max_tokens", 4}});
  CHECK(check_chunks(sr.body).finish == "length");
}

// ServerOptions::sampling (ingine-serve --temperature / --top-p / --top-k) fills the fields a request omits;
// fields the request sets win.
void test_sampling_defaults() {
  ingine::model::Sampling defaults;
  defaults.temperature = 0.7f;
  defaults.top_p = 0.95f;
  defaults.top_k = 40;
  auto s = serve(THINK_TEXT, 0.0, 4, 0, {}, defaults);
  chat(s->port(), {{"max_tokens", 2}});
  auto call = s->gen->calls().back();
  CHECK(call.sampling.temperature == 0.7f && call.sampling.top_p == 0.95f && call.sampling.top_k == 40);
  chat(s->port(), {{"max_tokens", 2}, {"temperature", 1.0}, {"top_p", 1.0}, {"top_k", 0}});
  call = s->gen->calls().back();
  CHECK(call.sampling.temperature == 1.0f && call.sampling.top_p == 1.0f && call.sampling.top_k == 0);
  chat(s->port(), {{"max_tokens", 2}, {"top_p", 0.5}});
  call = s->gen->calls().back();
  CHECK(call.sampling.temperature == 0.7f && call.sampling.top_p == 0.5f && call.sampling.top_k == 40);
}

void test_context_limit() {
  const int n = int(render(MSGS).ids.size());
  {
    auto s = serve(THINK_TEXT, 0.0, 4, n + 6);
    Chat r = chat(s->port(), Json::object());
    CHECK(r.body["choices"][0]["finish_reason"] == "length" && s->gen->calls().back().max_new_tokens == 6);
    chat(s->port(), {{"max_tokens", 100}});
    CHECK(s->gen->calls().back().max_new_tokens == 6);
  }
  auto s = serve(THINK_TEXT, 0.0, 4, n);
  Chat r = chat(s->port(), Json::object());
  CHECK(r.status == 400 && r.body["error"]["param"] == "messages");
}

void test_stop_strings_cancel_generation(bool stream) {
  std::string text = "<|message_model|><|content_text|>alpha beta<END> gamma";
  for (int i = 0; i < 200; ++i) text += " delta";
  text += "<|end_message|>";
  auto s = serve(text, 0.002);
  Chat r = chat(s->port(), {{"stop", Json::array({"<END>", "never"})},
                            {"stream", stream},
                            {"stream_options", {{"include_usage", true}}}});
  CHECK(r.status == 200);
  std::string content, finish;
  Json usage;
  if (stream) {
    Stream st = check_chunks(r.body, true);
    content = st.content, finish = st.finish, usage = st.usage;
  } else {
    check_completion(r.body);
    content = r.body["choices"][0]["message"]["content"];
    finish = r.body["choices"][0]["finish_reason"];
    usage = r.body["usage"];
  }
  CHECK(content == "alpha beta" && finish == "stop");
  CHECK(usage["completion_tokens"].get<int>() < 20);
  wait_for([&] { return s->gen->calls().back().done; });
  CHECK(s->gen->calls().back().cancelled && s->gen->calls().back().yielded < 50);
  Chat r2 = chat(s->port(), {{"stop", "beta"}});
  CHECK(r2.body["choices"][0]["message"]["content"] == "alpha ");
}

void test_disconnect_cancels_generation(bool stream) {
  auto s = serve(long_text(), 0.002);
  const int sock = raw_request(s->port(), {{"stream", stream}});
  wait_for([&] {
    auto calls = s->gen->calls();
    return !calls.empty() && calls.back().yielded > 20;
  });
  if (stream) {
    char buf[4096];
    const ssize_t n = ::recv(sock, buf, sizeof buf, 0);
    CHECK(n > 0 && std::string(buf, size_t(n)).starts_with("HTTP/1.1 200"));
  }
  ::close(sock);
  wait_for([&] { return s->gen->calls().back().done; });
  CHECK(s->gen->calls().back().cancelled && s->gen->calls().back().yielded < 2000);
  wait_for([&] { return !health(s->port())["running"].get<bool>(); });
  Chat r = chat(s->port(), {{"max_tokens", 3}});  // server still healthy
  CHECK(r.status == 200 && r.body["usage"]["completion_tokens"] == 3);
}

void test_queue_full_503_and_withdraw() {
  auto s = serve(long_text(), 0.005, 1);
  const int port = s->port();
  const int a = raw_request(port, {{"stream", true}});
  wait_for([&] { return health(port)["running"].get<bool>(); });
  const int b = raw_request(port, Json::object());
  wait_for([&] { return health(port)["waiting"] == 1; });
  Chat r = chat(port, Json::object());
  CHECK(r.status == 503 && r.body["error"]["code"] == 503 && r.body["error"]["message"].is_string());
  CHECK(r.body["error"]["type"] == "server_busy");
  CHECK(header(r.headers, "Retry-After") == "1");
  ::close(b);  // a queued request whose client leaves is withdrawn without ever running
  wait_for([&] { return health(port)["waiting"] == 0; });
  const int c = raw_request(port, {{"max_tokens", 2}});
  wait_for([&] { return health(port)["waiting"] == 1; });
  ::close(a);
  auto [head, body] = read_raw_response(c);
  ::close(c);
  CHECK(head.starts_with("HTTP/1.1 200"));
  CHECK(Json::parse(body)["usage"]["completion_tokens"] == 2);
  const auto calls = s->gen->calls();
  CHECK(calls.size() == 2);  // a and c ran; b never reached the generator
  CHECK(calls[0].cancelled && calls[0].yielded < 2000);
}

void test_fifo_order() {
  auto s = serve(THINK_TEXT, 0.001, 8);
  std::mutex mutex;
  std::vector<std::pair<int, int>> order;
  std::vector<std::thread> threads;
  for (int i = 0; i < 5; ++i) {
    threads.emplace_back([&, i] {
      Chat r = chat(s->port(), {{"messages", Json::array({{{"role", "user"}, {"content", "req " + std::to_string(i)}}})},
                                {"max_tokens", 4}});
      std::lock_guard lock(mutex);
      order.emplace_back(i, r.status);
    });
    wait_for([&] { return int(s->gen->calls().size()) + health(s->port())["waiting"].get<int>() >= i + 1; });
  }
  for (auto& t : threads) t.join();
  std::sort(order.begin(), order.end());
  for (int i = 0; i < 5; ++i) CHECK(order[size_t(i)] == std::make_pair(i, 200));
  const auto calls = s->gen->calls();
  CHECK(calls.size() == 5);
  for (int i = 0; i < 5; ++i) {
    CHECK(g_fmt->tokenizer().decode(calls[size_t(i)].prompt_ids).find("req " + std::to_string(i)) != std::string::npos);
  }
}

void test_generator_failure() {
  auto s = serve(THINK_TEXT, 0.0, 4, 0,
                 [](std::span<const int32_t>) -> FakeGenerator::Output { throw std::runtime_error("kaboom"); });
  Chat r = chat(s->port(), Json::object());
  CHECK(r.status == 500 && r.body["error"]["message"].get<std::string>().find("kaboom") != std::string::npos);
  CHECK(r.body["error"]["type"] == "server_error" && r.body["error"]["code"] == 500);
  Chat sr = chat(s->port(), {{"stream", true}});
  CHECK(sr.status == 200 && sr.body.back() == "[DONE]");
  CHECK(sr.body[sr.body.size() - 2]["error"]["message"].get<std::string>().find("kaboom") != std::string::npos);
  wait_for([&] { return !health(s->port())["running"].get<bool>(); });
}

void test_bad_requests() {
  const std::vector<std::pair<Json, std::string>> cases = {
      {{{"n", 2}}, "n"},
      {{{"temperature", -1}}, "temperature"},
      {{{"temperature", "hot"}}, "temperature"},
      {{{"top_p", 0}}, "top_p"},
      {{{"max_tokens", 0}}, "max_tokens"},
      {{{"max_tokens", 1.5}}, "max_tokens"},
      {{{"seed", true}}, "seed"},
      {{{"stream", "yes"}}, "stream"},
      {{{"stop", 5}}, "stop"},
      {{{"messages", Json::array()}}, "messages"},
      {{{"messages", Json::array({{{"role", "user"},
                                   {"content", Json::array({{{"type", "image_url"}, {"image_url", {{"url", "x"}}}}})}}})}},
       "messages"},
      {{{"reasoning_effort", 2}}, "reasoning_effort"},
      {{{"chat_template_kwargs", {{"enable_thinking", "no"}}}}, "enable_thinking"},
  };
  auto s = serve();
  for (const auto& [body, param] : cases) {
    Chat r = chat(s->port(), body);
    if (!(r.status == 400 && r.body["error"]["param"] == param && r.body["error"]["type"] == "invalid_request_error")) {
      std::fprintf(stderr, "bad request %s -> %d %s\n", body.dump().c_str(), r.status, r.body.dump().c_str());
    }
    CHECK(r.status == 400 && r.body["error"]["param"] == param && r.body["error"]["type"] == "invalid_request_error");
    CHECK(r.body["error"]["code"] == 400);
  }
  Chat r = chat(s->port(), {{"temperature", -1}});
  CHECK(r.body["error"]["message"] == "'temperature' must be in [0.0, 2.0]");
  r = chat(s->port(), {{"max_tokens", 0}});
  CHECK(r.body["error"]["message"] == "'max_tokens' must be in [1, None]");
  CHECK(s->gen->calls().empty());
}

void test_invalid_json() {
  auto s = serve();
  Response r = request(s->port(), "POST", "/v1/chat/completions", "{nope");
  CHECK(r.status == 400 && Json::parse(r.body)["error"]["message"].get<std::string>().find("invalid JSON") !=
                               std::string::npos);
  r = request(s->port(), "POST", "/v1/chat/completions", "[1]");
  CHECK(r.status == 400 && Json::parse(r.body)["error"]["message"] == "request body must be a JSON object");
}

// The default responder: echo, tool call, thinking disabled.
void test_echo() {
  auto gen = std::make_unique<FakeGenerator>(g_fmt->tokenizer(), "inkling-test");
  ChatServer srv(*gen, *g_fmt);
  CHECK(srv.bind("127.0.0.1", 0));
  srv.start();
  Chat r = chat(srv.port(), Json::object());
  const Json& m = r.body["choices"][0]["message"];
  CHECK(m["content"] == "You said: Hi 🎉" && m["reasoning_content"] == "The user wrote 4 characters.");
  r = chat(srv.port(), {{"enable_thinking", false}});
  CHECK(r.body["choices"][0]["message"]["content"] == "You said: Hi 🎉");
  CHECK(r.body["choices"][0]["message"]["reasoning_content"].is_null());
  r = chat(srv.port(), {{"messages", Json::array({{{"role", "user"}, {"content", "call get_weather {\"city\": \"Paris\"}"}}})},
                        {"tools", TOOLS}});
  const Json& t = r.body["choices"][0]["message"];
  CHECK(r.body["choices"][0]["finish_reason"] == "tool_calls" && t["reasoning_content"] == "Calling get_weather.");
  CHECK(t["tool_calls"][0]["function"]["name"] == "get_weather");
  CHECK((Json::parse(t["tool_calls"][0]["function"]["arguments"].get<std::string>()) == Json{{"city", "Paris"}}));
  srv.stop();
}

}  // namespace

int main() {
  const char* env = std::getenv("INGINE_MODEL_DIR");
  const std::filesystem::path model_dir = env && *env ? env : INGINE_MODEL_DIR;
  if (!std::filesystem::exists(model_dir / "tokenizer.json") || !std::filesystem::exists(model_dir / "chat_template.jinja")) {
    std::printf("skip: no tokenizer/chat template in %s\n", model_dir.c_str());
    return 77;
  }
  const ChatFormat fmt = ChatFormat::load(model_dir);
  g_fmt = &fmt;
  const std::vector<std::pair<std::string, std::function<void()>>> tests = {
      {"models_and_health", test_models_and_health},
      {"non_streaming", test_non_streaming},
      {"streaming_matches_non_streaming[False]", [] { test_streaming_matches_non_streaming(false); }},
      {"streaming_matches_non_streaming[True]", [] { test_streaming_matches_non_streaming(true); }},
      {"tool_call_streaming", test_tool_call_streaming},
      {"tool_round_trip_prompt", test_tool_round_trip_prompt},
      {"thinking_controls", test_thinking_controls},
      {"sampling_and_max_tokens", test_sampling_and_max_tokens},
      {"sampling_defaults", test_sampling_defaults},
      {"context_limit", test_context_limit},
      {"stop_strings_cancel_generation[False]", [] { test_stop_strings_cancel_generation(false); }},
      {"stop_strings_cancel_generation[True]", [] { test_stop_strings_cancel_generation(true); }},
      {"disconnect_cancels_generation[False]", [] { test_disconnect_cancels_generation(false); }},
      {"disconnect_cancels_generation[True]", [] { test_disconnect_cancels_generation(true); }},
      {"queue_full_503_and_withdraw", test_queue_full_503_and_withdraw},
      {"fifo_order", test_fifo_order},
      {"generator_failure", test_generator_failure},
      {"bad_requests", test_bad_requests},
      {"invalid_json", test_invalid_json},
      {"echo", test_echo},
  };
  int failed = 0;
  for (const auto& [name, fn] : tests) {
    const auto t0 = std::chrono::steady_clock::now();
    try {
      fn();
      std::printf("PASS %s (%.2fs)\n", name.c_str(),
                  std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count());
    } catch (const std::exception& e) {
      ++failed;
      std::printf("FAIL %s: %s\n", name.c_str(), e.what());
    }
    std::fflush(stdout);
  }
  std::printf("%d/%zu passed\n", int(tests.size()) - failed, tests.size());
  return failed ? 1 : 0;
}
