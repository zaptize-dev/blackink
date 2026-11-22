#include "serve/chat_server.h"

#include "engine/log.h"
#include "serve/http_transport.h"
#include "serve/webui.h"

#include <spdlog/spdlog.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <ctime>
#include <functional>
#include <optional>
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

void log_done(const Exchange& x) {
  const Clock::time_point now = Clock::now();
  const engine::Job& job = *x.job;
  const double queued = seconds(job.t_start().value_or(now) - job.t_submit);
  const int n = x.parser.num_tokens();
  const std::string rate =
      x.t_first && n > 1 ? fmt::format("{:.1f} tok/s", (n - 1) / std::max(seconds(now - *x.t_first), 1e-9)) : "-";
  const std::string ttft = x.t_first ? fmt::format("{:.3f}s", seconds(*x.t_first - job.t_submit)) : "-";
  const auto& finish = x.parser.finish_reason();
  engine::server_log().info("{}: prompt={} completion={} finish={} queued={:.3f}s ttft={} decode={}{}", job.id,
                            job.prompt_ids.size(), n, finish ? *finish : std::string("None"), queued, ttft, rate,
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
         {"choices", Json::array({Json{{"index", 0},
                                       {"delta", std::move(delta)},
                                       {"logprobs", nullptr},
                                       {"finish_reason", finish ? Json(*finish) : Json()}}})}};
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
              {"max_model_len", max_model_len_}};
}

void ChatServer::handle_get(const httplib::Request& req, httplib::Response& res) {
  const std::string path = route_path(req);
  if (path.empty() || path == "/chat" || path == "/ui") {
    res.status = 200;
    res.set_content(std::string(kWebUiHtml), "text/html; charset=utf-8");
  } else if (path == "/health") {
    const engine::SchedulerStats st = scheduler_.stats();
    send_json(res, 200,
              Json{{"status", "ok"},
                   {"model", model_id_},
                   {"waiting", st.waiting},
                   {"running", st.running},
                   {"max_queue", st.max_queue}});
  } else if (path == "/v1/models") {
    send_json(res, 200, Json{{"object", "list"}, {"data", Json::array({model_card()})}});
  } else if (path == "/v1/models/" + model_id_) {
    send_json(res, 200, model_card());
  } else {
    send_error(res, 404, "no route for GET " + path);
  }
}

void ChatServer::handle_post(const httplib::Request& req, httplib::Response& res) {
  const std::string path = route_path(req);
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
    request = parse_request(body, fmt_, max_model_len_, sampling_defaults_);
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
  auto job = std::make_shared<engine::Job>(request.prompt_ids, request.max_new_tokens, request.sampling);
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
                   {"usage", usage(*x)}});
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
