// Renders captured chat requests (traces/<dir>/<task>/NNNN-*.json, key "request") through ChatFormat.
//
//   text_traces [--dump FILE] [--check FILE] [--bench] DIR...
//
// Default (the CTest run): render every request and check the incremental path (render cache) against a cold
// render for every consecutive pair of a task; skipped (77) when no trace directory exists.
//   --dump FILE   write every rendered prompt (name, text, ids or the ChatError) to FILE
//   --check FILE  compare against a dump from another build (byte for byte)
//   --bench       time render stages at the sizes of the captured requests and a synthetic 200K conversation
#include "text/chat.h"
#include "text/jinja.h"
#include "text_test.h"

#include <algorithm>
#include <chrono>
#include <cstring>
#include <map>

using namespace ingine::text;
using Json = nlohmann::ordered_json;
namespace fs = std::filesystem;

namespace {

struct Request {
  std::string name;  // <dir>/<task>/<file>
  std::string task;  // conversation key (consecutive files of one task)
  std::string body;  // the request object as JSON text (what the server parses)
};

struct Output {
  std::string text;
  std::vector<int32_t> ids, prefill;
  std::string error;
  bool operator==(const Output&) const = default;
};

std::string read_file(const fs::path& path) {
  std::ifstream in(path, std::ios::binary);
  std::stringstream ss;
  ss << in.rdbuf();
  return ss.str();
}

std::vector<Request> load_requests(const std::vector<std::string>& dirs) {
  std::vector<Request> out;
  for (const auto& dir : dirs) {
    if (!fs::is_directory(dir)) continue;
    std::vector<fs::path> files;
    for (const auto& task : fs::directory_iterator(dir)) {
      if (!task.is_directory()) continue;
      for (const auto& f : fs::directory_iterator(task.path())) {
        if (f.path().extension() == ".json") files.push_back(f.path());
      }
    }
    std::sort(files.begin(), files.end());
    for (const auto& f : files) {
      const Json trace = Json::parse(read_file(f));
      if (!trace.contains("request")) continue;
      out.push_back({fs::path(dir).filename().string() + "/" + f.parent_path().filename().string() + "/" +
                         f.filename().string(),
                     dir + "/" + f.parent_path().filename().string(), trace["request"].dump()});
    }
  }
  return out;
}

// server/openai.cpp parse_request: the ChatFormat inputs of a request body.
struct Inputs {
  Json messages, tools;
  ChatOptions options;
};

Inputs inputs_of(const Json& body) {
  Inputs in;
  Json kwargs = body.contains("chat_template_kwargs") && body["chat_template_kwargs"].is_object()
                    ? body["chat_template_kwargs"]
                    : Json::object();
  Json effort = kwargs.contains("reasoning_effort") ? kwargs["reasoning_effort"] : Json();
  kwargs.erase("reasoning_effort");
  if (body.contains("reasoning_effort")) effort = body["reasoning_effort"];
  Json enable_thinking = body.contains("enable_thinking") ? body["enable_thinking"] : Json(true);
  if (kwargs.contains("enable_thinking")) enable_thinking = kwargs["enable_thinking"], kwargs.erase("enable_thinking");
  in.options.reasoning_effort = effort;
  in.options.enable_thinking = enable_thinking;
  in.options.template_kwargs = kwargs;
  in.messages = body.contains("messages") ? body["messages"] : Json();
  in.tools = body.contains("tools") ? body["tools"] : Json();
  return in;
}

Output render(const ChatFormat& fmt, const Inputs& in) {
  Output o;
  try {
    RenderedPrompt p = fmt.render(in.messages, in.tools, in.options);
    o.text = std::move(p.text);
    o.ids = std::move(p.ids);
    o.prefill = std::move(p.prefill_ids);
  } catch (const ChatError& e) {
    o.error = std::string(e.what()) + " [" + e.param + "]";
  }
  return o;
}

void put_str(std::ostream& os, const std::string& s) {
  const uint64_t n = s.size();
  os.write(reinterpret_cast<const char*>(&n), 8);
  os.write(s.data(), static_cast<std::streamsize>(n));
}
void put_ids(std::ostream& os, const std::vector<int32_t>& v) {
  const uint64_t n = v.size();
  os.write(reinterpret_cast<const char*>(&n), 8);
  os.write(reinterpret_cast<const char*>(v.data()), static_cast<std::streamsize>(n * 4));
}
bool get_str(std::istream& is, std::string& s) {
  uint64_t n = 0;
  if (!is.read(reinterpret_cast<char*>(&n), 8)) return false;
  s.resize(n);
  return static_cast<bool>(is.read(s.data(), static_cast<std::streamsize>(n)));
}
bool get_ids(std::istream& is, std::vector<int32_t>& v) {
  uint64_t n = 0;
  if (!is.read(reinterpret_cast<char*>(&n), 8)) return false;
  v.resize(n);
  return static_cast<bool>(is.read(reinterpret_cast<char*>(v.data()), static_cast<std::streamsize>(n * 4)));
}

using Clock = std::chrono::steady_clock;
double ms_since(Clock::time_point t) { return std::chrono::duration<double, std::milli>(Clock::now() - t).count(); }

template <class F>
double best_ms(int reps, F&& f) {
  double best = 1e30;
  for (int i = 0; i < reps; ++i) {
    const auto t = Clock::now();
    f();
    best = std::min(best, ms_since(t));
  }
  return best;
}

// The render cache exists in the new code only; the tool also builds against older trees for --dump.
template <class F>
constexpr bool kHasRenderCache = requires(const F& f) { f.clear_render_cache(); };

template <class F>
void clear_cache(const F& fmt) {
  if constexpr (kHasRenderCache<F>) fmt.clear_render_cache();
}

// Stage timings of one request: server body parse, normalize_messages, JSON -> template values (an empty
// template over the same context), full cold render, tokenisation of the rendered text, and the warm render
// of a follow-up (the same request rendered after its predecessor in the conversation).
void bench_one(const ChatFormat& fmt, const std::string& label, const std::string& body_text,
               const std::string* previous_body) {
  const Json body = Json::parse(body_text);
  const Inputs in = inputs_of(body);
  const Output o = render(fmt, in);
  static const JinjaTemplate empty("");
  const int reps = 7;
  const double parse = best_ms(reps, [&] { (void)Json::parse(body_text); });
  const double normalize = best_ms(reps, [&] { (void)normalize_messages(in.messages); });
  const double convert = best_ms(reps, [&] {
    Json ctx = Json::object();
    ctx["messages"] = normalize_messages(in.messages);
    ctx["tools"] = in.tools;
    (void)empty.render(ctx);
  });
  const double tokenize = best_ms(reps, [&] { (void)fmt.tokenizer().encode(o.text, true); });
  const double cold = best_ms(reps, [&] {
    clear_cache(fmt);
    (void)render(fmt, in);
  });
  double warm = -1;
  if (kHasRenderCache<ChatFormat> && previous_body != nullptr) {
    const Inputs prev = inputs_of(Json::parse(*previous_body));
    double best = 1e30;
    for (int i = 0; i < reps; ++i) {
      clear_cache(fmt);
      (void)render(fmt, prev);
      const auto t = Clock::now();
      (void)render(fmt, in);
      best = std::min(best, ms_since(t));
    }
    warm = best;
  }
  const double jinja = cold - tokenize - convert;
  std::printf("%-34s %7zu tok %4zu msg %7.2f parse %6.2f normalize %6.2f convert %7.2f jinja %6.2f tokenize "
              "%7.2f cold %7.2f warm\n",
              label.c_str(), o.ids.size(), in.messages.size(), parse, normalize, convert, jinja, tokenize, cold, warm);
}

}  // namespace

int main(int argc, char** argv) {
  std::string dump, check;
  bool bench = false;
  std::vector<std::string> dirs;
  for (int i = 1; i < argc; ++i) {
    if (!std::strcmp(argv[i], "--dump") && i + 1 < argc) {
      dump = argv[++i];
    } else if (!std::strcmp(argv[i], "--check") && i + 1 < argc) {
      check = argv[++i];
    } else if (!std::strcmp(argv[i], "--bench")) {
      bench = true;
    } else {
      dirs.emplace_back(argv[i]);
    }
  }
  const auto dir = test::model_dir();
  if (dir.empty()) return test::kSkip;
  const std::vector<Request> requests = load_requests(dirs);
  if (requests.empty()) {
    std::printf("skip: no captured requests\n");
    return test::kSkip;
  }
  const ChatFormat fmt = ChatFormat::load(dir);
  test::Checker c;

  if (bench) {
    // Representative captured requests: closest to 9K, 44K and 89K prompt tokens, then a synthetic ~200K one.
    std::vector<std::pair<std::size_t, std::size_t>> sizes;  // (tokens, index)
    for (std::size_t i = 0; i < requests.size(); ++i) {
      const Output o = render(fmt, inputs_of(Json::parse(requests[i].body)));
      sizes.emplace_back(o.ids.size(), i);
    }
    for (const std::size_t target : {9000u, 44000u, 89000u}) {
      const auto best = std::min_element(sizes.begin(), sizes.end(), [&](const auto& a, const auto& b) {
        return std::llabs(static_cast<long long>(a.first) - static_cast<long long>(target)) <
               std::llabs(static_cast<long long>(b.first) - static_cast<long long>(target));
      });
      const Request& r = requests[best->second];
      const bool has_prev = best->second > 0 && requests[best->second - 1].task == r.task;
      bench_one(fmt, r.name, r.body, has_prev ? &requests[best->second - 1].body : nullptr);
    }
    // Synthetic: the largest request with its conversation (after the first message) repeated to ~200K tokens;
    // the follow-up appends one more user message.
    const auto largest = std::max_element(sizes.begin(), sizes.end());
    Json body = Json::parse(requests[largest->second].body);
    Json& msgs = body["messages"];
    const Json base = msgs;
    const std::size_t per_copy = largest->first;
    for (std::size_t tokens = per_copy; tokens < 200000; tokens += per_copy) {
      for (std::size_t m = 1; m < base.size(); ++m) msgs.push_back(base[m]);
    }
    Json next = body;
    next["messages"].push_back(Json{{"role", "user"}, {"content", "Continue."}});
    const std::string first = body.dump();
    bench_one(fmt, "synthetic", next.dump(), &first);
    return 0;
  }

  std::vector<Output> outputs;
  outputs.reserve(requests.size());
  int pairs = 0;
  for (std::size_t i = 0; i < requests.size(); ++i) {
    const Inputs in = inputs_of(Json::parse(requests[i].body));
    clear_cache(fmt);
    const Output cold = render(fmt, in);
    if constexpr (kHasRenderCache<ChatFormat>) {
      if (cold.error.empty()) {
        c.expect(cold.ids == fmt.tokenizer().encode(cold.text, true), "cold ids != encode(text): " + requests[i].name);
      }
      // Incremental: render the predecessor, then this request with the cache primed; compare with cold.
      if (i > 0 && requests[i - 1].task == requests[i].task) {
        clear_cache(fmt);
        (void)render(fmt, inputs_of(Json::parse(requests[i - 1].body)));
        const Output warm = render(fmt, in);
        c.expect(warm == cold, "incremental render differs from cold: " + requests[i].name);
        ++pairs;
      }
      // Repeated render of the same request (full cache hit).
      c.expect(render(fmt, in) == cold, "repeated render differs: " + requests[i].name);
    }
    outputs.push_back(cold);
  }
  if constexpr (kHasRenderCache<ChatFormat>) {
    // Conversations interleaved round-robin (several cache entries live at once), without clearing.
    std::map<std::string, std::vector<std::size_t>> by_task;
    for (std::size_t i = 0; i < requests.size(); ++i) by_task[requests[i].task].push_back(i);
    clear_cache(fmt);
    int interleaved = 0;
    for (std::size_t step = 0;; ++step) {
      bool any = false;
      for (const auto& [task, list] : by_task) {
        if (step >= list.size()) continue;
        any = true;
        const std::size_t i = list[step];
        c.expect(render(fmt, inputs_of(Json::parse(requests[i].body))) == outputs[i],
                 "interleaved render differs: " + requests[i].name);
        ++interleaved;
      }
      if (!any) break;
    }
    std::printf("%zu requests, %d consecutive pairs and %d interleaved renders checked against cold renders\n",
                requests.size(), pairs, interleaved);
  }

  if (!dump.empty()) {
    std::ofstream os(dump, std::ios::binary);
    for (std::size_t i = 0; i < requests.size(); ++i) {
      put_str(os, requests[i].name);
      put_str(os, outputs[i].text);
      put_ids(os, outputs[i].ids);
      put_ids(os, outputs[i].prefill);
      put_str(os, outputs[i].error);
    }
    std::printf("wrote %s\n", dump.c_str());
  }
  if (!check.empty()) {
    std::ifstream is(check, std::ios::binary);
    std::map<std::string, Output> ref;
    std::string name;
    while (get_str(is, name)) {
      Output o;
      get_str(is, o.text);
      get_ids(is, o.ids);
      get_ids(is, o.prefill);
      get_str(is, o.error);
      ref.emplace(name, std::move(o));
    }
    c.expect(ref.size() == requests.size(), "reference has " + std::to_string(ref.size()) + " requests");
    for (std::size_t i = 0; i < requests.size(); ++i) {
      const auto it = ref.find(requests[i].name);
      if (!c.expect(it != ref.end(), "missing in reference: " + requests[i].name)) continue;
      c.expect(it->second.text == outputs[i].text, "text differs: " + requests[i].name);
      c.expect(it->second.ids == outputs[i].ids, "ids differ: " + requests[i].name);
      c.expect(it->second.prefill == outputs[i].prefill, "prefill differs: " + requests[i].name);
      c.expect(it->second.error == outputs[i].error, "error differs: " + requests[i].name);
    }
  }
  return c.finish("text_traces");
}
