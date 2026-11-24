// ingine-serve: the OpenAI-compatible chat completions server (flags and defaults of `python -m ingine.server`).
#include "engine/fake_generator.h"
#include "engine/log.h"
#include "ingine/generator.h"
#include "serve/chat_server.h"
#include "serve/cli.h"
#include "text/chat.h"

#include <nlohmann/json.hpp>
#include <spdlog/spdlog.h>

#include <unistd.h>

#include <algorithm>
#include <cerrno>
#include <csignal>
#include <exception>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <memory>
#include <string>
#include <thread>

namespace {

int g_stop_pipe[2] = {-1, -1};

void handle_signal(int) {
  const char c = 's';
  (void)!::write(g_stop_pipe[1], &c, 1);
}

// tokenizer_config.json "model_max_length" (the Python server's fallback context limit), else 1 << 20.
int tokenizer_max_length(const std::filesystem::path& model_dir) {
  std::ifstream in(model_dir / "tokenizer_config.json");
  if (!in) return 1 << 20;
  try {
    const auto cfg = nlohmann::json::parse(in);
    if (auto it = cfg.find("model_max_length"); it != cfg.end() && it->is_number()) {
      return int(std::min<double>(it->get<double>(), 2147483647.0));
    }
  } catch (const nlohmann::json::exception&) {
  }
  return 1 << 20;
}

// Sampling of requests that omit temperature / top_p / top_k: generation_config.json's values where present,
// else ingine's defaults (temperature 1, no top-p, no top-k).
ingine::model::Sampling generation_defaults(const std::filesystem::path& model_dir) {
  ingine::model::Sampling s;
  std::ifstream in(model_dir / "generation_config.json");
  if (!in) return s;
  try {
    const auto cfg = nlohmann::json::parse(in);
    if (auto it = cfg.find("temperature"); it != cfg.end() && it->is_number()) s.temperature = it->get<float>();
    if (auto it = cfg.find("top_p"); it != cfg.end() && it->is_number()) s.top_p = it->get<float>();
    if (auto it = cfg.find("top_k"); it != cfg.end() && it->is_number_integer()) s.top_k = std::max(0, it->get<int>());
  } catch (const nlohmann::json::exception&) {
  }
  return s;
}

}  // namespace

int main(int argc, char** argv) {
  std::string model_dir, host = "0.0.0.0", ablation, backend = "fast", model_id, log_level = "INFO";
  int port = 8090, max_ctx = 98304, max_queue = 4, max_model_len = 0, park_mb = 0;
  bool engine = false, fake = false, dense_fp8 = false, kv_fp8 = false, kv_q4 = false, full_nvfp4 = false, spec = false;
  double spec_threshold = 0.8, fake_delay = 0.02;
  // sampling defaults; negative: from generation_config.json
  double temperature = -1, top_p = -1;
  int top_k = -1;

  ingine::serve::ArgParser ap("ingine-serve", "OpenAI-compatible chat completions server for BlackInk (ingine).");
  ap.option("--model-dir", "", "checkpoint dir (tokenizer + chat template)", &model_dir, {}, true);
  ap.option("--host", "", "", &host);
  ap.option("--port", "", "", &port);
  ap.flag("--engine", "serve the model with ingine", &engine);
  ap.flag("--fake", "serve canned outputs (FakeGenerator)", &fake);
  ap.exclusive({"--engine", "--fake"}, true);
  ap.option("--ablation", "", "ablation file (safetensors) applied at every visit", &ablation);
  ap.option("--max-ctx", "", "positions held by the full-attention caches", &max_ctx);
  ap.option("--backend", "{fast}", "decode step implementation", &backend, {"fast"});
  ap.flag("--dense-fp8", "FP8 E4M3 dense projections and head (row scales)", &dense_fp8);
  ap.flag("--full-nvfp4", "transcode MXFP8 experts (layers 20-23) to NVFP4 and quantize embed to FP8 for 16 GB GPUs",
          &full_nvfp4);
  ap.flag("--spec", "DSpark speculative decoding (mtp.safetensors drafter)", &spec);
  ap.flag("--kv-fp8", "FP8 E4M3 full-attention KV caches (half the memory; needed for 262144 positions on 32 GB)",
          &kv_fp8);
  ap.flag("--kv-q4", "Q4 4-bit full-attention KV caches (quarter the memory; 131072 positions in 1 GiB for 16 GB GPUs)",
          &kv_q4);
  ap.option("--spec-threshold", "", "verify the leading drafts whose draft probability exceeds this (0 = always all 7)",
            &spec_threshold);
  ap.option("--park-mb", "",
            "device memory (MiB) for conversations displaced by another one, released when memory runs short, "
            "0 = off (default: min(3072, free memory after start-up - 4096))",
            &park_mb);
  ap.option("--fake-delay", "", "seconds per fake token", &fake_delay);
  ap.option("--model-id", "", "model name reported by the API (default: generator.model_id)", &model_id);
  ap.option("--max-queue", "", "requests allowed to wait behind the running one", &max_queue);
  ap.option("--max-model-len", "",
            "context limit (default: generator.max_model_len or tokenizer model_max_length)", &max_model_len);
  ap.option("--log-level", "", "", &log_level);
  ap.option("--temperature", "", "temperature of requests that omit it (default: generation_config.json, else 1)",
            &temperature);
  ap.option("--top-p", "", "top_p of requests that omit it (default: generation_config.json, else 1)", &top_p);
  ap.option("--top-k", "", "top_k of requests that omit it, 0 = off (default: generation_config.json, else 0)",
            &top_k);
  ap.parse(argc, argv);
  if (temperature > 2 || top_p == 0 || top_p > 1) {
    std::cerr << "ingine-serve: error: --temperature must be in [0, 2] and --top-p in (0, 1]\n";
    return 2;
  }

  auto& log = ingine::engine::server_log();
  if (!ingine::engine::set_log_level(log_level)) {
    std::cerr << "ingine-serve: error: unknown level: '" << log_level << "'\n";
    return 2;
  }
  std::signal(SIGPIPE, SIG_IGN);

  try {
    const ingine::text::ChatFormat fmt = ingine::text::ChatFormat::load(model_dir);
    std::unique_ptr<ingine::model::Generator> generator;
    if (fake) {
      generator = std::make_unique<ingine::engine::FakeGenerator>(
          fmt.tokenizer(), "Inkling-2-Mini-fake", tokenizer_max_length(model_dir), ingine::engine::FakeGenerator::Responder{},
          fake_delay);
    } else {
#ifdef INGINE_WITH_MODEL
      ingine::model::RuntimeOptions options;
      options.model_dir = model_dir;
      options.ablation = ablation;
      options.max_ctx = max_ctx;
      options.dense_fp8 = dense_fp8;
      options.kv_fp8 = kv_fp8;
      options.kv_q4 = kv_q4;
      options.full_nvfp4 = full_nvfp4;
      options.spec = spec;
      options.spec_threshold = float(spec_threshold);
      options.park_mb = park_mb;
      generator = ingine::model::make_generator(options);
#else
      log.critical("--engine: this ingine-serve was built without ingine_model (src/model)");
      return 1;
#endif
    }

    ingine::serve::ServerOptions server_options;
    server_options.max_queue = max_queue;
    server_options.max_model_len = max_model_len;
    server_options.model_id = model_id;
    server_options.model_dir = model_dir;
    server_options.sampling = generation_defaults(model_dir);
    if (temperature >= 0) server_options.sampling.temperature = float(temperature);
    if (top_p >= 0) server_options.sampling.top_p = float(top_p);
    if (top_k >= 0) server_options.sampling.top_k = top_k;
    log.info("default sampling: temperature {} top_p {} top_k {}", server_options.sampling.temperature,
             server_options.sampling.top_p, server_options.sampling.top_k);
    ingine::serve::ChatServer server(*generator, fmt, server_options);
    if (!server.bind(host, port)) {
      log.critical("cannot bind {}:{}", host, port);
      return 1;
    }

    if (::pipe(g_stop_pipe) != 0) throw std::runtime_error("pipe() failed");
    std::signal(SIGINT, handle_signal);
    std::signal(SIGTERM, handle_signal);
    log.info("serving {} on http://{}:{}/v1", server.model_id(), host, server.port());
    bool listened = true;
    std::thread listener([&] {
      listened = server.listen();
      handle_signal(0);  // the accept loop ended on its own: wake the main thread
    });
    char c;
    while (::read(g_stop_pipe[0], &c, 1) < 0 && errno == EINTR) {
    }
    server.stop();
    listener.join();
    if (!listened) {
      log.critical("listen failed on {}:{}", host, server.port());
      return 1;
    }
    return 0;
  } catch (const std::exception& e) {
    log.critical("{}", e.what());
    return 1;
  }
}
