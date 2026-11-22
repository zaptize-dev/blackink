// The server's logger ("ingine.server") in Python logging's format:
// "%(asctime)s %(levelname)s %(name)s: %(message)s".
#pragma once

#include <spdlog/logger.h>

#include <string_view>

namespace ingine::engine {

spdlog::logger& server_log();

// Python logging level names (case-insensitive): DEBUG, INFO, WARNING/WARN, ERROR, CRITICAL/FATAL, NOTSET.
// Returns false (level unchanged) for anything else.
bool set_log_level(std::string_view name);

}  // namespace ingine::engine
