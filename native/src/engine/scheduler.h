// Request scheduler (mirror of engine/ingine/server.py Scheduler/Job): one worker thread runs jobs strictly one at
// a time in FIFO order from a bounded waiting list. The worker only pulls token ids from Generator::generate and
// pushes them to the job's output; detokenisation, parsing and HTTP writes happen on the request thread, so the
// decode loop never waits on the client. Cancellation (client gone, stop string, shutdown) is checked between
// the tokens the generator emits.
#pragma once

#include "ingine/generator.h"

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <deque>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <thread>
#include <unordered_set>
#include <vector>

namespace ingine::engine {

using Clock = std::chrono::steady_clock;

// Output drained from a job: token ids in order, then a failure message and/or the end of generation.
struct JobBatch {
  std::vector<int32_t> tokens;
  std::optional<std::string> failure;
  bool end = false;
  bool empty() const { return tokens.empty() && !failure && !end; }
};

class Job {
 public:
  Job(std::vector<int32_t> prompt_ids, int max_new_tokens, model::Sampling sampling);

  const std::string id;  // "chatcmpl-" + 32 hex digits
  const std::vector<int32_t> prompt_ids;
  const int max_new_tokens;
  const model::Sampling sampling;
  const Clock::time_point t_submit;

  void cancel() { cancel_.store(true, std::memory_order_release); }
  bool cancelled() const { return cancel_.load(std::memory_order_acquire); }
  std::optional<Clock::time_point> t_start() const;

  // Waits up to `timeout` for output, then returns everything available (empty on timeout).
  JobBatch take(Clock::duration timeout);

 private:
  friend class Scheduler;
  void started();
  void push(const std::vector<int32_t>& tokens);
  void fail(std::string message);
  void finish();

  std::atomic<bool> cancel_{false};
  mutable std::mutex mutex_;
  std::condition_variable cv_;
  std::deque<int32_t> tokens_;
  std::optional<std::string> failure_;
  bool end_ = false;
  std::optional<Clock::time_point> t_start_;
};

struct SchedulerStats {
  int waiting = 0;
  bool running = false;
  int max_queue = 0;
};

class Scheduler {
 public:
  Scheduler(model::Generator& generator, std::vector<int32_t> stop_ids, int max_queue);
  ~Scheduler();
  Scheduler(const Scheduler&) = delete;
  Scheduler& operator=(const Scheduler&) = delete;

  // False when closed or when max_queue jobs already wait (the running one counts as in server.py).
  bool submit(std::shared_ptr<Job> job);
  // Cancels the job and drops it from the waiting list if it has not started.
  void withdraw(const std::shared_ptr<Job>& job);
  SchedulerStats stats() const;
  // Fails waiting jobs ("server shutting down"), cancels the running one and joins the worker.
  void close();

 private:
  void loop();
  void run(Job& job);

  model::Generator& generator_;
  const std::vector<int32_t> stop_list_;
  const std::unordered_set<int32_t> stop_ids_;
  const int max_queue_;
  mutable std::mutex mutex_;
  std::condition_variable cv_;
  std::deque<std::shared_ptr<Job>> waiting_;
  std::shared_ptr<Job> active_;
  bool closed_ = false;
  std::thread thread_;
};

}  // namespace ingine::engine
