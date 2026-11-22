// Scheduler unit tests with a generator that emits several tokens per callback (like a speculative round):
// the scheduler must cut a batch at a stop id, at max_new_tokens and on cancellation, run jobs in FIFO order,
// bound the waiting list and report generator failures.
#include "engine/scheduler.h"

#include <atomic>
#include <chrono>
#include <cstdio>
#include <functional>
#include <optional>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

namespace {

using ingine::engine::Job;
using ingine::engine::JobBatch;
using ingine::engine::Scheduler;
using ingine::model::GenerateStats;
using ingine::model::Sampling;

int g_failures = 0;
#define CHECK(cond)                                                                   \
  do {                                                                                \
    if (!(cond)) {                                                                    \
      std::fprintf(stderr, "%s:%d: check failed: %s\n", __FILE__, __LINE__, #cond); \
      ++g_failures;                                                                   \
    }                                                                                 \
  } while (0)

// Emits tokens 100, 101, ... in rounds of `round` tokens until on_tokens refuses; throws if prompt[0] < 0.
class RoundGenerator final : public ingine::model::Generator {
 public:
  explicit RoundGenerator(int round, std::chrono::milliseconds pause = {}) : round_(round), pause_(pause) {}
  std::string model_id() const override { return "round"; }
  int max_model_len() const override { return 1 << 20; }
  GenerateStats generate(std::span<const int32_t> prompt, int, const Sampling&, std::span<const int32_t>,
                         const std::function<bool(std::span<const int32_t>)>& on_tokens) override {
    if (!prompt.empty() && prompt[0] < 0) throw std::invalid_argument("prompt does not fit");
    order.push_back(prompt.empty() ? -1 : prompt[0]);
    for (int32_t next = 100;; next += round_) {
      std::vector<int32_t> toks;
      for (int i = 0; i < round_; ++i) toks.push_back(next + i);
      if (pause_.count()) std::this_thread::sleep_for(pause_);
      if (!on_tokens(toks)) break;
    }
    return {};
  }
  std::vector<int32_t> order;

 private:
  int round_;
  std::chrono::milliseconds pause_;
};

std::vector<int32_t> drain(Job& job, std::optional<std::string>* failure = nullptr) {
  std::vector<int32_t> out;
  for (;;) {
    JobBatch b = job.take(std::chrono::seconds(5));
    out.insert(out.end(), b.tokens.begin(), b.tokens.end());
    if (b.failure && failure) *failure = b.failure;
    if (b.end) return out;
    if (b.empty()) throw std::runtime_error("timeout");
  }
}

void test_cuts() {
  RoundGenerator gen(4);
  Scheduler s(gen, {105}, 4);
  auto stop = std::make_shared<Job>(std::vector<int32_t>{1}, 100, Sampling{});
  CHECK(s.submit(stop));
  CHECK((drain(*stop) == std::vector<int32_t>{100, 101, 102, 103, 104, 105}));  // stop id mid-round
  auto len = std::make_shared<Job>(std::vector<int32_t>{2}, 3, Sampling{});
  CHECK(s.submit(len));
  CHECK((drain(*len) == std::vector<int32_t>{100, 101, 102}));  // max_new_tokens mid-round
  CHECK(len->t_start().has_value());
}

void test_cancel_fifo_bound() {
  RoundGenerator gen(2, std::chrono::milliseconds(5));
  Scheduler s(gen, {}, 1);
  auto a = std::make_shared<Job>(std::vector<int32_t>{1}, 1000000, Sampling{});
  auto b = std::make_shared<Job>(std::vector<int32_t>{2}, 2, Sampling{});
  auto c = std::make_shared<Job>(std::vector<int32_t>{3}, 2, Sampling{});
  CHECK(s.submit(a));
  while (!s.stats().running) std::this_thread::sleep_for(std::chrono::milliseconds(1));
  CHECK(s.submit(b));
  CHECK(!s.submit(c));  // running + 1 waiting > max_queue 1
  CHECK(s.stats().waiting == 1 && s.stats().max_queue == 1);
  s.withdraw(b);  // never runs
  CHECK(s.stats().waiting == 0);
  CHECK(s.submit(c));
  a->cancel();
  CHECK(drain(*a).size() < 1000);
  CHECK(drain(*c).size() == 2);
  CHECK((gen.order == std::vector<int32_t>{1, 3}));
}

void test_failure_and_close() {
  RoundGenerator gen(1, std::chrono::milliseconds(2));
  Scheduler s(gen, {}, 4);
  auto bad = std::make_shared<Job>(std::vector<int32_t>{-1}, 5, Sampling{});
  CHECK(s.submit(bad));
  std::optional<std::string> failure;
  CHECK(drain(*bad, &failure).empty());
  CHECK(failure && failure->find("generation failed: std::invalid_argument: prompt does not fit") == 0);
  auto running = std::make_shared<Job>(std::vector<int32_t>{1}, 1000000, Sampling{});
  auto waiting = std::make_shared<Job>(std::vector<int32_t>{2}, 5, Sampling{});
  CHECK(s.submit(running));
  CHECK(s.submit(waiting));
  s.close();
  failure.reset();
  CHECK(drain(*waiting, &failure).empty() && failure == "server shutting down");
  CHECK(drain(*running).size() < 1000000);
  CHECK(!s.submit(std::make_shared<Job>(std::vector<int32_t>{3}, 1, Sampling{})));
}

}  // namespace

int main() {
  test_cuts();
  test_cancel_fifo_bound();
  test_failure_and_close();
  std::printf(g_failures ? "FAILED (%d checks)\n" : "ok\n", g_failures);
  return g_failures ? 1 : 0;
}
