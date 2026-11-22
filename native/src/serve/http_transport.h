// Adapted from fngine (github.com/satellitedown/fngine, Apache-2.0) for ingine.
// SSE transport over a cpp-httplib chunked response (keep-alive comments, client-disconnect detection) and the
// server socket options.
#pragma once

#include <httplib.h>

#include <atomic>
#include <chrono>
#include <exception>
#include <string_view>

namespace ingine::serve {

class ClientDisconnected final : public std::exception {
 public:
  [[nodiscard]] const char* what() const noexcept override { return "client disconnected"; }
};

class SseTransport final {
 public:
  using Clock = std::chrono::steady_clock;

  static constexpr std::chrono::seconds kHeartbeatInterval{5};
  static constexpr std::string_view kHeartbeatComment = ": keep-alive\n\n";

  explicit SseTransport(httplib::DataSink& sink, std::atomic<bool>& cancelled,
                        Clock::duration heartbeat_interval = kHeartbeatInterval, Clock::time_point now = Clock::now());

  // Writes one chunk; throws ClientDisconnected when the peer is gone.
  void write(std::string_view item, Clock::time_point now = Clock::now());

  // Called by the wait loop. Besides observing an already-closed socket, a quiet stream periodically writes an
  // SSE comment so TCP_USER_TIMEOUT has traffic with which to detect an unacknowledged peer. Returns true (and
  // marks the request cancelled) when the client is gone.
  [[nodiscard]] bool poll(Clock::time_point now = Clock::now());

 private:
  bool mark_cancelled() noexcept;

  httplib::DataSink& sink_;
  std::atomic<bool>& cancelled_;
  Clock::duration heartbeat_interval_;
  Clock::time_point last_write_;
};

void configure_http_server_socket(socket_t socket) noexcept;

}  // namespace ingine::serve
