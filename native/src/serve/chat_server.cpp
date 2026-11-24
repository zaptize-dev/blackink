#include "serve/chat_server.h"

#include "engine/log.h"
#include "serve/http_transport.h"
#include "serve/monitor_ui.h"
#include "serve/sys_metrics.h"
#include "serve/webui.h"

#include <spdlog/spdlog.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <ctime>
#include <functional>
#include <optional>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

namespace ingine::serve {
namespace {

using engine::Clock;

constexpr auto kPollInterval = std::chrono::milliseconds(50);  // between client-disconnect checks

// self.path.split("?", 1)[0].rstrip("/")
std::string route_path(const httplib::Request& req) {
  std::string path = req.target.substr(0, req.target.find('?'));
  while (!path.empty() && path.back() == '/') path.pop_back();
  return path;
}

void send_json(httplib::Response& res, int status, const Json& body) {
  res.status = status;
  res.set_content(dump(body), "application/json");
}

void send_error(httplib::Response& res, int status, std::string_view message,
                const std::optional<std::string>& param = std::nullopt) {
  send_json(res, status, error_body(status, message, param));
}

std::string sse(std::string_view payload) {
  std::string out;
  out.reserve(payload.size() + 8);
  out.append("data: ").append(payload).append("\n\n");
  return out;
}

double seconds(Clock::duration d) { return std::chrono::duration<double>(d).count(); }

// One request's state on its request thread.
struct Exchange {
  Exchange(std::shared_ptr<engine::Job> job_, text::StreamParser parser_, ChatRequest req_)
      : job(std::move(job_)), parser(std::move(parser_)), req(std::move(req_)), created(std::time(nullptr)) {}

  std::shared_ptr<engine::Job> job;
  text::StreamParser parser;
  ChatRequest req;
  int64_t created;
  std::optional<std::string> failure;
  std::optional<Clock::time_point> t_first;
};

// Drains the job's output into the parser until it reports a finish reason; `gone()` is polled every
// kPollInterval and ends the exchange with ClientDisconnected. `emit` receives every batch's deltas.
void pump(Exchange& x, const std::function<bool()>& gone, const std::function<void(std::vector<text::Delta>&)>& emit) {
  Clock::time_point last_check = Clock::now();
  for (;;) {
    engine::JobBatch batch = x.job->take(kPollInterval);
    const Clock::time_point now = Clock::now();
    if (now - last_check >= kPollInterval) {
      last_check = now;
      if (gone()) throw ClientDisconnected();
    }
    if (batch.empty()) continue;
    std::vector<text::Delta> deltas;
    for (int32_t tok : batch.tokens) {
      if (x.parser.finished()) continue;
      if (!x.t_first) x.t_first = Clock::now();
      std::vector<text::Delta> d = x.parser.feed(tok);
      deltas.insert(deltas.end(), std::make_move_iterator(d.begin()), std::make_move_iterator(d.end()));
      if (x.parser.finished()) x.job->cancel();  // stop string / stop token: nothing more to generate
    }
    if (batch.failure) x.failure = std::move(batch.failure);
    if (x.parser.finished() || batch.end) {
      std::vector<text::Delta> d = x.parser.finish();
      deltas.insert(deltas.end(), std::make_move_iterator(d.begin()), std::make_move_iterator(d.end()));
    }
    if (!deltas.empty()) emit(deltas);
    if (x.parser.finish_reason()) break;
  }
}

Json usage(const Exchange& x) {
  const int prompt = int(x.job->prompt_ids.size());
  const int completion = x.parser.num_tokens();
  return Json{{"prompt_tokens", prompt}, {"completion_tokens", completion}, {"total_tokens", prompt + completion}};
}

Json timings(const Exchange& x) {
  const Clock::time_point now = Clock::now();
  const engine::Job& job = *x.job;
  const int cache_n = std::max(0, job.reused_tokens());
  const int prompt_n = std::max(0, int(job.prompt_ids.size()) - cache_n);
  const int predicted_n = x.parser.num_tokens();
  const Clock::time_point t0 = job.t_start().value_or(job.t_submit);
  const double prompt_ms = x.t_first ? std::max(0.0, seconds(*x.t_first - t0) * 1000.0) : 0.0;
  const double predicted_ms =
      (x.t_first && predicted_n > 1) ? std::max(0.0, seconds(now - *x.t_first) * 1000.0) : 0.0;
  const double predicted_per_second =
      (predicted_ms > 0.0 && predicted_n > 1) ? (double(predicted_n - 1) * 1000.0 / predicted_ms) : 0.0;
  return Json{{"cache_n", cache_n},
              {"prompt_n", prompt_n},
              {"prompt_ms", prompt_ms},
              {"predicted_n", predicted_n},
              {"predicted_ms", predicted_ms},
              {"predicted_per_second", predicted_per_second}};
}

std::string format_prometheus_metrics(const engine::SchedulerMetrics& m) {
  struct Item {
    const char* name;
    const char* description;
    double value;
  };
  const std::vector<Item> counters = {
      {"prompt_tokens_total", "Number of prompt tokens processed, excluding cached tokens",
       double(m.prompt_tokens_total)},
      {"prompt_tokens_cached_total", "Number of prompt tokens reused from the cache",
       double(m.prompt_tokens_cached_total)},
      {"prompt_seconds_total", "Total time spent processing prompts", m.prompt_seconds_total},
      {"tokens_predicted_total", "Number of generation tokens processed", double(m.tokens_predicted_total)},
      {"tokens_predicted_seconds_total", "Total time spent generating tokens", m.tokens_predicted_seconds_total},
      {"n_decode_total",
       "Total number of llama_decode() calls, excluding speculative decoding and multimodal decoding",
       double(m.n_decode_total)},
      {"n_tokens_max", "Largest observed sequence length (prompt + generation)", double(m.n_tokens_max)},
      {"spec_decode_num_draft_tokens_total", "Speculative: Total draft tokens generated",
       double(m.spec_decode_num_draft_tokens_total)},
      {"spec_decode_num_accepted_tokens_total", "Speculative: Total draft tokens accepted by the target model",
       double(m.spec_decode_num_accepted_tokens_total)},
      {"spec_decode_num_drafts_total", "Speculative: Total speculative decoding verification steps",
       double(m.spec_decode_num_drafts_total)},
  };
  const double prompt_tps =
      m.prompt_seconds_total > 1e-9 ? double(m.prompt_tokens_total) / m.prompt_seconds_total : 0.0;
  const double predicted_tps = m.tokens_predicted_seconds_total > 1e-9
                                   ? double(m.tokens_predicted_total) / m.tokens_predicted_seconds_total
                                   : 0.0;
  const std::vector<Item> gauges = {
      {"prompt_tokens_seconds", "Average prompt throughput in tokens/s", prompt_tps},
      {"predicted_tokens_seconds", "Average generation throughput in tokens/s", predicted_tps},
      {"requests_processing", "Number of requests processing", double(m.requests_processing)},
      {"requests_deferred", "Number of requests deferred", double(m.requests_deferred)},
      {"n_busy_slots_per_decode", "Average number of busy slots per llama_decode() call",
       m.n_decode_total > 0 ? 1.0 : 0.0},
  };

  std::ostringstream prometheus;
  auto add_items = [&prometheus](const char* type, const std::vector<Item>& items) {
    for (const auto& item : items) {
      prometheus << "# HELP llamacpp:" << item.name << " " << item.description << "\n"
                 << "# TYPE llamacpp:" << item.name << " " << type << "\n"
                 << "llamacpp:" << item.name << " " << item.value << "\n";
    }
  };
  add_items("counter", counters);
  add_items("gauge", gauges);

  if (m.spec_decode_num_drafts_total > 0) {
    prometheus << "# HELP llamacpp:spec_decode_num_accepted_tokens_per_pos_total"
                  " Accepted tokens per draft position\n"
               << "# TYPE llamacpp:spec_decode_num_accepted_tokens_per_pos_total counter\n";
    for (std::size_t i = 0; i < m.spec_accepted_per_pos.size(); ++i) {
      prometheus << "llamacpp:spec_decode_num_accepted_tokens_per_pos_total{position=\"" << i << "\"} "
                 << m.spec_accepted_per_pos[i] << "\n";
    }
  }
  return prometheus.str();
}

void log_done(const Exchange& x) {
  const Clock::time_point now = Clock::now();
  const engine::Job& job = *x.job;
  const double queued = seconds(job.t_start().value_or(now) - job.t_submit);
  const int n = x.parser.num_tokens();
  const std::string rate =
      x.t_first && n > 1 ? fmt::format("{:.1f} tok/s", (n - 1) / std::max(seconds(now - *x.t_first), 1e-9)) : "-";
  const std::string ttft = x.t_first ? fmt::format("{:.3f}s", seconds(*x.t_first - job.t_submit)) : "-";
  const auto& finish = x.parser.finish_reason();
  engine::server_log().info("{}: prompt={} completion={} finish={} queued={:.3f}s ttft={} decode={} (T={:.2f},p={:.2f},k={}){}", job.id,
                            job.prompt_ids.size(), n, finish ? *finish : std::string("None"), queued, ttft, rate,
                            x.req.sampling.temperature, x.req.sampling.top_p, x.req.sampling.top_k,
                            x.failure ? " error=" + *x.failure : std::string());
}

void log_disconnected(const Exchange& x) {
  engine::server_log().info("{}: client disconnected after {} tokens; cancelled", x.job->id, x.parser.num_tokens());
}

Json chunk(const Exchange& x, const std::string& model_id, Json delta, const std::optional<std::string>& finish) {
  Json c{{"id", x.job->id},
         {"object", "chat.completion.chunk"},
         {"created", x.created},
         {"model", model_id},
         {"system_fingerprint", nullptr},
         {"x_tokens", x.parser.num_tokens()},
         {"choices", Json::array({Json{{"index", 0},
                                       {"delta", std::move(delta)},
                                       {"logprobs", nullptr},
                                       {"finish_reason", finish ? Json(*finish) : Json()}}})}};
  if (finish || x.parser.num_tokens() > 0) c["timings"] = timings(x);
  if (x.req.include_usage) c["usage"] = nullptr;
  return c;
}

struct StreamState {
  explicit StreamState(std::shared_ptr<Exchange> x_) : x(std::move(x_)) {}
  std::shared_ptr<Exchange> x;
  std::atomic<bool> cancelled{false};
  std::atomic<bool> started{false};
};

}  // namespace

ChatServer::ChatServer(model::Generator& generator, const text::ChatFormat& fmt, ServerOptions options)
    : fmt_(fmt),
      model_id_(options.model_id.empty() ? generator.model_id() : options.model_id),
      max_model_len_(options.max_model_len ? options.max_model_len : generator.max_model_len()),
      sampling_defaults_(options.sampling),
      created_(std::time(nullptr)),
      scheduler_(generator, fmt.stop_ids(), options.max_queue) {
  if (!options.model_dir.empty() &&
      std::filesystem::exists(options.model_dir / "model-00001-of-00001.safetensors")) {
    vision_ = std::make_unique<vision::VisionEncoder>(options.model_dir);
  }
  // A pool thread per connection, as ThreadingHTTPServer: the running request, the waiting ones, the ones
  // rejected with 503 and /health probes must all be served concurrently.
  server_.new_task_queue = [] { return new httplib::ThreadPool(8, 1024); };
  server_.set_socket_options(configure_http_server_socket);
  register_routes();
}

ChatServer::~ChatServer() { stop(); }

void ChatServer::register_routes() {
  server_.set_logger([](const httplib::Request& req, const httplib::Response& res) {
    engine::server_log().debug("{} \"{} {} {}\" {} -", req.remote_addr, req.method, req.target, req.version,
                               res.status);
  });
  server_.Get(R"(.*)", [this](const httplib::Request& req, httplib::Response& res) { handle_get(req, res); });
  server_.Post(R"(.*)", [this](const httplib::Request& req, httplib::Response& res) { handle_post(req, res); });
}

Json ChatServer::model_card() const {
  return Json{{"id", model_id_},
              {"object", "model"},
              {"created", created_},
              {"owned_by", "ingine"},
              {"max_model_len", max_model_len_},
              {"in_cache", true},
              {"path", model_id_},
              {"status", {{"value", "loaded"}}},
              {"architecture", {{"input_modalities", Json::array({"text"})}}}};
}

void ChatServer::handle_get(const httplib::Request& req, httplib::Response& res) {
  const std::string path = route_path(req);
  if (path == "/health") {
    const engine::SchedulerStats st = scheduler_.stats();
    send_json(res, 200,
              Json{{"status", "ok"},
                   {"model", model_id_},
                   {"waiting", st.waiting},
                   {"running", st.running},
                   {"max_queue", st.max_queue}});
  } else if (path == "/v1/models" || path == "/models") {
    send_json(res, 200,
              Json{{"object", "list"},
                   {"data", Json::array({model_card()})},
                   {"models", Json::array({Json{{"name", model_id_},
                                                {"model", model_id_},
                                                {"capabilities", Json::array({"reasoning", "tools", "completion"})},
                                                {"details", {{"format", "safetensors"},
                                                             {"family", "inkling"},
                                                             {"parameter_size", "20B-A1.8B"},
                                                             {"quantization_level", "NVFP4"}}}}})}});
  } else if (path == "/v1/models/" + model_id_) {
    send_json(res, 200, model_card());
  } else if (path == "/props") {
    const engine::SchedulerStats st = scheduler_.stats();
    send_json(
        res, 200,
        Json{{"default_generation_settings",
              {{"id", 0},
               {"id_task", -1},
               {"n_ctx", max_model_len_},
               {"speculative", true},
               {"is_processing", st.running},
               {"params",
                {{"n_predict", 32768},
                 {"seed", 4294967295ULL},
                 {"temperature", sampling_defaults_.temperature},
                 {"dynatemp_range", 0.0},
                 {"dynatemp_exponent", 1.0},
                 {"top_k", sampling_defaults_.top_k},
                 {"top_p", sampling_defaults_.top_p},
                 {"min_p", 0.0},
                 {"top_n_sigma", -1.0},
                 {"xtc_probability", 0.0},
                 {"xtc_threshold", 0.1},
                 {"typ_p", 1.0},
                 {"typical_p", 1.0},
                 {"repeat_last_n", 64},
                 {"repeat_penalty", 1.0},
                 {"presence_penalty", 0.0},
                 {"frequency_penalty", 0.0},
                 {"dry_multiplier", 0.0},
                 {"dry_base", 1.75},
                 {"dry_allowed_length", 2},
                 {"dry_penalty_last_n", 64},
                 {"dry_sequence_breakers", Json::array({"\n", ":", "\"", "*"})},
                 {"mirostat", 0},
                 {"mirostat_tau", 5.0},
                 {"mirostat_eta", 0.1},
                 {"stop", Json::array()},
                 {"max_tokens", -1},
                 {"n_keep", 0},
                 {"n_discard", 0},
                 {"ignore_eos", false},
                 {"stream", true},
                 {"logit_bias", Json::array()},
                 {"n_probs", 0},
                 {"min_keep", 0},
                 {"grammar", ""},
                 {"grammar_lazy", false},
                 {"grammar_triggers", Json::array()},
                 {"preserved_tokens", Json::array()},
                 {"chat_format", "Content-only"},
                 {"reasoning_format", "deepseek"},
                 {"reasoning_in_content", false},
                 {"generation_prompt", ""},
                 {"samplers", Json::array({"top_k", "top_p", "temperature"})},
                 {"backend_sampling", true},
                 {"speculative.n_max", 7},
                 {"speculative.n_min", 0},
                 {"speculative.p_min", 0.8},
                 {"timings_per_token", true},
                 {"post_sampling_probs", false},
                 {"lora", Json::array()}}},
               {"prompt", ""},
               {"next_token",
                {{"has_next_token", false},
                 {"has_new_line", false},
                 {"n_remain", 0},
                 {"n_decoded", 0},
                 {"stopping_word", ""}}}}},
             {"total_slots", 1},
             {"model_path", model_id_},
             {"role", "model"},
             {"modalities", {{"vision", bool(vision_)}, {"audio", false}, {"video", false}}},
             {"chat_template", "inkling-2-mini"},
             {"chat_template_caps",
              {{"reasoning_effort_levels", Json::array({"minimal", "low", "medium", "high", "xhigh", "max"})}}},
             {"bos_token", "<|beginoftext|>"},
             {"eos_token", "<|content_model_end_sampling|>"},
             {"build_info", "BlackInk (sm_120a NVFP4 + 96K Q4 KV + DSpark MTP)"}});
  } else if (path == "/slots") {
    const engine::SchedulerStats st = scheduler_.stats();
    send_json(res, 200,
              Json::array({Json{{"id", 0},
                                {"id_task", 0},
                                {"n_ctx", max_model_len_},
                                {"speculative", true},
                                {"is_processing", st.running}}}));
  } else if (path == "/tools") {
    send_json(res, 200, Json::array());
  } else if (path == "/api/metrics") {
    const engine::SchedulerMetrics m = scheduler_.metrics();
    const SysSnapshot s = sample_sys_metrics();
    const double prompt_tps =
        m.prompt_seconds_total > 1e-9 ? double(m.prompt_tokens_total) / m.prompt_seconds_total : 0.0;
    const double predicted_tps = m.tokens_predicted_seconds_total > 1e-9
                                     ? double(m.tokens_predicted_total) / m.tokens_predicted_seconds_total
                                     : 0.0;
    Json per_pos = Json::array();
    for (uint64_t v : m.spec_accepted_per_pos) per_pos.push_back(v);
    Json reqs = Json::array();
    for (const auto& r : m.recent_requests) {
      reqs.push_back(Json{{"timestamp_ms", r.timestamp_ms},
                          {"id", r.id},
                          {"status", r.status},
                          {"prompt_tokens", r.prompt_tokens},
                          {"reused_tokens", r.reused_tokens},
                          {"output_tokens", r.output_tokens},
                          {"decode_tps", r.decode_tps},
                          {"duration_s", r.duration_s},
                          {"ttft_ms", r.ttft_ms},
                          {"spec_rounds", r.spec_rounds},
                          {"spec_accepted", r.spec_accepted}});
    }
    std::string reasoning_effort;
    std::string reasoning_source;
    {
      std::lock_guard<std::mutex> lk(reasoning_mu_);
      reasoning_effort = reasoning_effort_;
      reasoning_source = reasoning_source_;
    }
    send_json(res, 200,
              Json{{"model",
                    {{"model_id", model_id_},
                     {"max_model_len", max_model_len_},
                     {"reasoning_effort", reasoning_effort},
                     {"reasoning_source", reasoning_source},
                     {"state", m.state},
                     {"requests_processing", m.requests_processing},
                     {"requests_deferred", m.requests_deferred},
                     {"active_prompt_tokens", m.active_prompt_tokens},
                     {"active_output_tokens", m.active_output_tokens},
                     {"active_live_tps", m.active_live_tps},
                     {"last_prefill_tps", m.last_prefill_tps},
                     {"context_used", m.context_used},
                     {"prompt_tokens_total", m.prompt_tokens_total},
                     {"prompt_tokens_cached_total", m.prompt_tokens_cached_total},
                     {"tokens_predicted_total", m.tokens_predicted_total},
                     {"prompt_tokens_seconds", prompt_tps},
                     {"predicted_tokens_seconds", predicted_tps},
                     {"spec_decode_num_drafts_total", m.spec_decode_num_drafts_total},
                     {"spec_decode_num_accepted_tokens_total", m.spec_decode_num_accepted_tokens_total},
                     {"spec_accepted_per_pos", std::move(per_pos)},
                     {"recent_requests", std::move(reqs)}}},
                   {"system",
                    {{"gpu_available", s.gpu_available},
                     {"gpu_name", s.gpu_name},
                     {"gpu_util_pct", s.gpu_util_pct},
                     {"mem_util_pct", s.mem_util_pct},
                     {"vram_used_gb", s.vram_used_gb},
                     {"vram_total_gb", s.vram_total_gb},
                     {"gpu_temp_c", s.gpu_temp_c},
                     {"power_w", s.power_w},
                     {"power_limit_w", s.power_limit_w},
                     {"pcie_gen", s.pcie_gen},
                     {"pcie_width", s.pcie_width},
                     {"pcie_tx_mbs", s.pcie_tx_mbs},
                     {"pcie_rx_mbs", s.pcie_rx_mbs},
                     {"cpu_util_pct", s.cpu_util_pct},
                     {"cpu_threads", s.cpu_threads},
                     {"ram_used_gb", s.ram_used_gb},
                     {"ram_total_gb", s.ram_total_gb}}}});
  } else if (path == "/metrics") {
    res.status = 200;
    res.set_header("Cache-Control", "no-cache");
    res.set_content(std::string(kMonitorHtml), "text/html; charset=utf-8");
  } else if (const WebUiAsset* asset = find_webui_asset(path)) {
    if (req.has_header("If-None-Match") && req.get_header_value("If-None-Match") == asset->etag) {
      res.status = 304;
      return;
    }
    res.status = 200;
    res.set_header("Content-Encoding", "gzip");
    res.set_header("ETag", std::string(asset->etag));
    res.set_header("Cache-Control", asset->immutable ? "public, max-age=31536000, immutable" : "no-cache");
    res.set_content(reinterpret_cast<const char*>(asset->gzip_data), asset->gzip_size, std::string(asset->mime_type));
  } else {
    send_error(res, 404, "no route for GET " + path);
  }
}

void ChatServer::handle_post(const httplib::Request& req, httplib::Response& res) {
  const std::string path = route_path(req);
  if (path == "/v1/streams/lookup") {
    send_json(res, 200, Json::array());
    return;
  }
  if (path == "/v1/chat/completions/control") {
    send_json(res, 200, Json::object());
    return;
  }
  if (path != "/v1/chat/completions") {
    send_error(res, 404, "no route for POST " + path);
    return;
  }
  if (!req.has_header("Content-Length")) {
    res.set_header("Connection", "close");
    send_error(res, 411, "Content-Length required");
    return;
  }
  ChatRequest request;
  try {
    Json body;
    try {
      body = Json::parse(req.body);
    } catch (const Json::parse_error& e) {
      send_error(res, 400, std::string("invalid JSON body: ") + e.what());
      return;
    }
    if (!body.is_object()) throw BadRequest("request body must be a JSON object");
    request = parse_request(body, fmt_, max_model_len_, sampling_defaults_, vision_.get());
    {
      std::lock_guard<std::mutex> lk(reasoning_mu_);
      reasoning_effort_ = request.reasoning_effort;
      reasoning_source_ = request.reasoning_source;
    }
  } catch (const BadRequest& e) {
    send_error(res, e.status, e.what(), e.param);
    return;
  } catch (const text::ChatError& e) {
    send_error(res, 400, e.what(), e.param);
    return;
  }
  complete(req, res, std::move(request));
}

void ChatServer::complete(const httplib::Request& req, httplib::Response& res, ChatRequest request) {
  if (request.max_new_tokens == 0 && !request.stream) {
    const int prompt_n = int(request.prompt_ids.size());
    send_json(res, 200,
              Json{{"id", "chatcmpl-preencode"},
                   {"object", "chat.completion"},
                   {"created", std::time(nullptr)},
                   {"model", model_id_},
                   {"system_fingerprint", nullptr},
                   {"choices", Json::array({Json{{"index", 0},
                                                 {"message",
                                                  {{"role", "assistant"},
                                                   {"content", ""},
                                                   {"reasoning_content", nullptr},
                                                   {"refusal", nullptr}}},
                                                 {"logprobs", nullptr},
                                                 {"finish_reason", "length"}}})},
                   {"usage", {{"prompt_tokens", prompt_n}, {"completion_tokens", 0}, {"total_tokens", prompt_n}}},
                   {"timings", {{"cache_n", 0}, {"prompt_n", prompt_n}, {"predicted_n", 0}}}});
    return;
  }
  auto job = std::make_shared<engine::Job>(request.prompt_ids, request.max_new_tokens, request.sampling,
                                           std::move(request.images));
  if (!scheduler_.submit(job)) {
    res.set_header("Retry-After", "1");
    send_error(res, 503, "server busy: request queue is full");
    return;
  }
  text::StreamParser parser = fmt_.parser(request.prefill_ids, request.stop);
  auto x = std::make_shared<Exchange>(std::move(job), std::move(parser), std::move(request));

  if (!x->req.stream) {
    try {
      pump(*x, [&req] { return req.is_connection_closed(); }, [](std::vector<text::Delta>&) {});
    } catch (const ClientDisconnected&) {
      scheduler_.withdraw(x->job);
      log_disconnected(*x);
      res.set_header("Connection", "close");
      send_error(res, 500, "client disconnected");
      return;
    }
    log_done(*x);
    if (x->failure) {
      send_error(res, 500, *x->failure);
      return;
    }
    const text::Completion result = x->parser.result();
    Json message{{"role", "assistant"},
                 {"content", !result.content.empty() || result.tool_calls.empty() ? Json(result.content) : Json()},
                 {"reasoning_content", result.reasoning_content.empty() ? Json() : Json(result.reasoning_content)},
                 {"refusal", nullptr}};
    if (!result.tool_calls.empty()) {
      Json calls = Json::array();
      for (const text::ToolCall& tc : result.tool_calls) {
        calls.push_back(Json{{"id", tc.id},
                             {"type", "function"},
                             {"function", {{"name", tc.name}, {"arguments", tc.arguments}}}});
      }
      message["tool_calls"] = std::move(calls);
    }
    send_json(res, 200,
              Json{{"id", x->job->id},
                   {"object", "chat.completion"},
                   {"created", x->created},
                   {"model", model_id_},
                   {"system_fingerprint", nullptr},
                   {"choices", Json::array({Json{{"index", 0},
                                                 {"message", std::move(message)},
                                                 {"logprobs", nullptr},
                                                 {"finish_reason", result.finish_reason ? Json(*result.finish_reason)
                                                                                        : Json()}}})},
                   {"usage", usage(*x)},
                   {"timings", timings(*x)}});
    return;
  }

  auto state = std::make_shared<StreamState>(std::move(x));
  res.set_header("Cache-Control", "no-cache");
  res.set_chunked_content_provider(
      "text/event-stream",
      [this, state](size_t, httplib::DataSink& sink) -> bool {
        if (state->started.exchange(true, std::memory_order_acq_rel)) {
          sink.done();
          return true;
        }
        Exchange& x = *state->x;
        SseTransport transport(sink, state->cancelled);
        try {
          transport.write(sse(dump(chunk(x, model_id_, Json{{"role", "assistant"}, {"content", ""}}, std::nullopt))));
          pump(x, [&] { return transport.poll(); },
               [&](std::vector<text::Delta>& deltas) {
                 for (const text::Delta& d : text::merge_deltas(deltas)) {
                   transport.write(sse(dump(chunk(x, model_id_, delta_json(d), std::nullopt))));
                 }
               });
        } catch (const ClientDisconnected&) {
          scheduler_.withdraw(x.job);
          log_disconnected(x);
          return false;
        }
        log_done(x);
        try {
          if (x.failure) {
            transport.write(sse(dump(error_body(500, *x.failure))));
          } else {
            transport.write(sse(dump(chunk(x, model_id_, Json::object(), x.parser.finish_reason()))));
            if (x.req.include_usage) {
              transport.write(sse(dump(Json{{"id", x.job->id},
                                            {"object", "chat.completion.chunk"},
                                            {"created", x.created},
                                            {"model", model_id_},
                                            {"system_fingerprint", nullptr},
                                            {"choices", Json::array()},
                                            {"usage", usage(x)}})));
            }
          }
          transport.write(sse("[DONE]"));
        } catch (const ClientDisconnected&) {
          return false;
        }
        sink.done();
        return true;
      },
      [this, state](bool success) {
        // The client left before or while streaming: the job must not keep (or start) generating.
        if (!success || !state->started.load(std::memory_order_acquire)) scheduler_.withdraw(state->x->job);
      });
}

bool ChatServer::bind(const std::string& host, int port) {
  if (port == 0) {
    port_ = server_.bind_to_any_port(host);
    return port_ > 0;
  }
  port_ = port;
  return server_.bind_to_port(host, port);
}

bool ChatServer::listen() { return server_.listen_after_bind(); }

void ChatServer::start() {
  thread_ = std::thread([this] { server_.listen_after_bind(); });
  server_.wait_until_ready();
}

void ChatServer::stop() {
  scheduler_.close();
  server_.stop();
  if (thread_.joinable()) thread_.join();
}

}  // namespace ingine::serve
