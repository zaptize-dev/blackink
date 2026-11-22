#include "engine/log.h"

#include <spdlog/pattern_formatter.h>
#include <spdlog/sinks/stdout_sinks.h>

#include <algorithm>
#include <cctype>
#include <memory>
#include <string>

namespace ingine::engine {
namespace {

// %(levelname)s: Python's level names.
class LevelName final : public spdlog::custom_flag_formatter {
 public:
  void format(const spdlog::details::log_msg& msg, const std::tm&, spdlog::memory_buf_t& dest) override {
    std::string_view name;
    switch (msg.level) {
      case spdlog::level::trace:
      case spdlog::level::debug: name = "DEBUG"; break;
      case spdlog::level::info: name = "INFO"; break;
      case spdlog::level::warn: name = "WARNING"; break;
      case spdlog::level::err: name = "ERROR"; break;
      default: name = "CRITICAL"; break;
    }
    dest.append(name.data(), name.data() + name.size());
  }
  std::unique_ptr<custom_flag_formatter> clone() const override { return std::make_unique<LevelName>(); }
};

std::shared_ptr<spdlog::logger> make_logger() {
  auto logger = std::make_shared<spdlog::logger>("ingine.server", std::make_shared<spdlog::sinks::stderr_sink_mt>());
  auto formatter = std::make_unique<spdlog::pattern_formatter>();
  formatter->add_flag<LevelName>('*').set_pattern("%Y-%m-%d %H:%M:%S,%e %* %n: %v");
  logger->set_formatter(std::move(formatter));
  logger->set_level(spdlog::level::info);
  logger->flush_on(spdlog::level::trace);
  return logger;
}

}  // namespace

spdlog::logger& server_log() {
  static const std::shared_ptr<spdlog::logger> logger = make_logger();
  return *logger;
}

bool set_log_level(std::string_view name) {
  std::string up(name);
  std::transform(up.begin(), up.end(), up.begin(), [](unsigned char c) { return std::toupper(c); });
  spdlog::level::level_enum level;
  if (up == "DEBUG") level = spdlog::level::debug;
  else if (up == "INFO") level = spdlog::level::info;
  else if (up == "WARNING" || up == "WARN") level = spdlog::level::warn;
  else if (up == "ERROR") level = spdlog::level::err;
  else if (up == "CRITICAL" || up == "FATAL") level = spdlog::level::critical;
  else if (up == "NOTSET") level = spdlog::level::trace;
  else return false;
  server_log().set_level(level);
  return true;
}

}  // namespace ingine::engine
