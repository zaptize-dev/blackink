// Adapted from fngine (github.com/satellitedown/fngine, Apache-2.0) for ingine.
#include "serve/http_transport.h"

#if defined(__linux__)
#include <netinet/tcp.h>
#include <sys/socket.h>
#endif

#include <stdexcept>

namespace ingine::serve {
namespace {

#if defined(__linux__)
constexpr int kKeepAliveIdleSeconds = 10;
constexpr int kKeepAliveIntervalSeconds = 3;
constexpr int kKeepAliveProbeCount = 3;
constexpr unsigned int kTcpUserTimeoutMilliseconds = 15000;

template <class T>
void set_socket_option(socket_t socket, int level, int option, const T& value) noexcept {
  (void)::setsockopt(socket, level, option, &value, sizeof(value));
}
#endif

}  // namespace

SseTransport::SseTransport(httplib::DataSink& sink, std::atomic<bool>& cancelled, Clock::duration heartbeat_interval,
                           Clock::time_point now)
    : sink_(sink), cancelled_(cancelled), heartbeat_interval_(heartbeat_interval), last_write_(now) {
  if (heartbeat_interval_ <= Clock::duration::zero()) {
    throw std::invalid_argument("SSE heartbeat interval must be positive");
  }
}

bool SseTransport::mark_cancelled() noexcept {
  cancelled_.store(true, std::memory_order_release);
  return true;
}

void SseTransport::write(std::string_view item, Clock::time_point now) {
  if (cancelled_.load(std::memory_order_acquire) || !sink_.write(item.data(), item.size())) {
    mark_cancelled();
    throw ClientDisconnected();
  }
  last_write_ = now;
}

bool SseTransport::poll(Clock::time_point now) {
  if (cancelled_.load(std::memory_order_acquire)) return true;
  if (sink_.is_writable && !sink_.is_writable()) return mark_cancelled();
  if (now - last_write_ < heartbeat_interval_) return false;
  if (!sink_.write(kHeartbeatComment.data(), kHeartbeatComment.size())) return mark_cancelled();
  last_write_ = now;
  return false;
}

void configure_http_server_socket(socket_t socket) noexcept {
  httplib::default_socket_options(socket);
#if defined(__linux__)
  const int enabled = 1;
  set_socket_option(socket, SOL_SOCKET, SO_KEEPALIVE, enabled);
  set_socket_option(socket, IPPROTO_TCP, TCP_KEEPIDLE, kKeepAliveIdleSeconds);
  set_socket_option(socket, IPPROTO_TCP, TCP_KEEPINTVL, kKeepAliveIntervalSeconds);
  set_socket_option(socket, IPPROTO_TCP, TCP_KEEPCNT, kKeepAliveProbeCount);
  set_socket_option(socket, IPPROTO_TCP, TCP_USER_TIMEOUT, kTcpUserTimeoutMilliseconds);
#endif
}

}  // namespace ingine::serve
