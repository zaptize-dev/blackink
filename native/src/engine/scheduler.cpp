#include "engine/scheduler.h"

#include "engine/log.h"

#include <cxxabi.h>

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <exception>
#include <random>
#include <span>
#include <typeinfo>
#include <utility>

namespace ingine::engine {
namespace {

// uuid.uuid4().hex: 32 random hex digits.
std::string random_hex32() {
  thread_local std::mt19937_64 rng{std::random_device{}() ^ (uint64_t(std::random_device{}()) << 32)};
  char buf[33];
  std::snprintf(buf, sizeof buf, "%016llx%016llx", static_cast<unsigned long long>(rng()),
                static_cast<unsigned long long>(rng()));
  return buf;
}

std::string type_name(const std::exception& e) {
  int status = 0;
  char* name = abi::__cxa_demangle(typeid(e).name(), nullptr, nullptr, &status);
  std::string out = status == 0 && name ? name : typeid(e).name();
  std::free(name);
  return out;
}

}  // namespace

Job::Job(std::vector<int32_t> prompt_ids_, int max_new_tokens_, model::Sampling sampling_)
    : id("chatcmpl-" + random_hex32()),
      prompt_ids(std::move(prompt_ids_)),
      max_new_tokens(max_new_tokens_),
      sampling(sampling_),
      t_submit(Clock::now()) {}

std::optional<Clock::time_point> Job::t_start() const {
  std::lock_guard lock(mutex_);
  return t_start_;
}

JobBatch Job::take(Clock::duration timeout) {
  std::unique_lock lock(mutex_);
  cv_.wait_for(lock, timeout, [&] { return !tokens_.empty() || failure_ || end_; });
  JobBatch batch;
  batch.tokens.assign(tokens_.begin(), tokens_.end());
  tokens_.clear();
  // The failure and the end are reported once, after every token pushed before them.
  batch.failure = std::exchange(failure_, std::nullopt);
  batch.end = std::exchange(end_, false);
  return batch;
}

void Job::started() {
  std::lock_guard lock(mutex_);
  t_start_ = Clock::now();
}

void Job::push(const std::vector<int32_t>& tokens) {
  if (tokens.empty()) return;
  {
    std::lock_guard lock(mutex_);
    tokens_.insert(tokens_.end(), tokens.begin(), tokens.end());
  }
  cv_.notify_all();
}

void Job::fail(std::string message) {
  {
    std::lock_guard lock(mutex_);
    failure_ = std::move(message);
  }
  cv_.notify_all();
}

void Job::finish() {
  {
    std::lock_guard lock(mutex_);
    end_ = true;
  }
  cv_.notify_all();
}

Scheduler::Scheduler(model::Generator& generator, std::vector<int32_t> stop_ids, int max_queue)
    : generator_(generator),
      stop_list_(std::move(stop_ids)),
      stop_ids_(stop_list_.begin(), stop_list_.end()),
      max_queue_(max_queue),
      thread_([this] { loop(); }) {}

Scheduler::~Scheduler() { close(); }

bool Scheduler::submit(std::shared_ptr<Job> job) {
  {
    std::lock_guard lock(mutex_);
    if (closed_ || int(waiting_.size()) + (active_ != nullptr) > max_queue_) return false;
    waiting_.push_back(std::move(job));
  }
  cv_.notify_all();
  return true;
}

void Scheduler::withdraw(const std::shared_ptr<Job>& job) {
  job->cancel();
  std::lock_guard lock(mutex_);
  auto it = std::find(waiting_.begin(), waiting_.end(), job);
  if (it != waiting_.end()) waiting_.erase(it);
}

SchedulerStats Scheduler::stats() const {
  std::lock_guard lock(mutex_);
  return {int(waiting_.size()), active_ != nullptr, max_queue_};
}

void Scheduler::close() {
  {
    std::lock_guard lock(mutex_);
    closed_ = true;
    for (auto& job : waiting_) {
      job->cancel();
      job->fail("server shutting down");
      job->finish();
    }
    waiting_.clear();
    if (active_) active_->cancel();
  }
  cv_.notify_all();
  if (thread_.joinable()) thread_.join();
}

void Scheduler::loop() {
  for (;;) {
    std::shared_ptr<Job> job;
    {
      std::unique_lock lock(mutex_);
      cv_.wait(lock, [&] { return !waiting_.empty() || closed_; });
      if (closed_) return;
      job = active_ = waiting_.front();
      waiting_.pop_front();
    }
    run(*job);
    job->finish();
    std::lock_guard lock(mutex_);
    active_ = nullptr;
  }
}

void Scheduler::run(Job& job) {
  if (job.cancelled()) return;
  job.started();
  int n = 0;
  std::vector<int32_t> out;
  const auto on_tokens = [&](std::span<const int32_t> tokens) -> bool {
    out.clear();
    bool more = true;
    for (int32_t tok : tokens) {
      if (job.cancelled()) {
        more = false;
        break;
      }
      out.push_back(tok);
      ++n;
      if (stop_ids_.contains(tok) || n >= job.max_new_tokens || job.cancelled()) {
        more = false;
        break;
      }
    }
    job.push(out);
    return more;
  };
  try {
    const model::GenerateStats st =
        generator_.generate(job.prompt_ids, job.max_new_tokens, job.sampling, stop_list_, on_tokens);
    // prefix reuse of the prompt: the path, the tokens reused and prefilled, the prefill time
    server_log().info("{}: reuse={} reused={} prefilled={} prefill={:.3f}s", job.id, model::to_string(st.reuse),
                      st.reused_tokens, st.prompt_tokens - st.reused_tokens, st.prefill_seconds);
  } catch (const std::exception& e) {  // report to the client instead of killing the worker
    server_log().error("generator failed: {}: {}", type_name(e), e.what());
    job.fail("generation failed: " + type_name(e) + ": " + e.what());
  } catch (...) {
    server_log().error("generator failed: unknown exception");
    job.fail("generation failed: unknown exception");
  }
}

}  // namespace ingine::engine
