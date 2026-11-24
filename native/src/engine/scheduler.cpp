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

Job::Job(std::vector<int32_t> prompt_ids_, int max_new_tokens_, model::Sampling sampling_,
         std::vector<model::ImageSpan> images_)
    : id("chatcmpl-" + random_hex32()),
      prompt_ids(std::move(prompt_ids_)),
      max_new_tokens(max_new_tokens_),
      sampling(sampling_),
      images(std::move(images_)),
      t_submit(Clock::now()) {}

std::optional<Clock::time_point> Job::t_start() const {
  std::lock_guard lock(mutex_);
  return t_start_;
}

std::optional<Clock::time_point> Job::t_first_token() const {
  std::lock_guard lock(mutex_);
  return t_first_token_;
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
    if (!t_first_token_) t_first_token_ = Clock::now();
    tokens_.insert(tokens_.end(), tokens.begin(), tokens.end());
  }
  live_tokens_.fetch_add(int(tokens.size()), std::memory_order_acq_rel);
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

SchedulerMetrics Scheduler::metrics() const {
  std::lock_guard lock(mutex_);
  SchedulerMetrics m = metrics_;
  m.requests_processing = active_ != nullptr ? 1 : 0;
  m.requests_deferred = int(waiting_.size());
  m.last_prefill_tps = last_prefill_tps_;
  m.context_used = last_context_used_;
  m.recent_requests.assign(recent_.begin(), recent_.end());
  if (active_) {
    const int live_out = active_->live_tokens();
    const int prompt_n = int(active_->prompt_ids.size());
    const auto t_first = active_->t_first_token();
    const auto t_start = active_->t_start().value_or(active_->t_submit);
    const Clock::time_point now = Clock::now();
    m.active_prompt_tokens = prompt_n;
    m.active_output_tokens = live_out;
    m.context_used = prompt_n + live_out;
    if (t_first && live_out > 0) {
      m.state = "generating";
      const double sec = std::chrono::duration<double>(now - *t_first).count();
      m.active_live_tps = (sec > 1e-6 && live_out > 1) ? double(live_out - 1) / sec : 0.0;
    } else {
      m.state = "prefilling";
    }
    RequestRecord live;
    live.timestamp_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                            std::chrono::system_clock::now().time_since_epoch())
                            .count();
    live.id = active_->id;
    live.status = "running";
    live.prompt_tokens = prompt_n;
    live.reused_tokens = active_->reused_tokens();
    live.output_tokens = live_out;
    live.decode_tps = m.active_live_tps;
    live.duration_s = std::chrono::duration<double>(now - t_start).count();
    live.ttft_ms = t_first ? std::chrono::duration<double>(*t_first - t_start).count() * 1000.0 : 0.0;
    m.recent_requests.insert(m.recent_requests.begin(), std::move(live));
  } else if (!waiting_.empty()) {
    m.state = "queued";
  } else {
    m.state = "idle";
  }
  return m;
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
        generator_.generate_multimodal(job.prompt_ids, job.images, job.max_new_tokens, job.sampling, stop_list_,
                                       on_tokens);
    job.reused_tokens_.store(st.reused_tokens, std::memory_order_release);
    {
      std::lock_guard lock(mutex_);
      const int prefilled = std::max(0, st.prompt_tokens - st.reused_tokens);
      const int cached = std::max(0, st.reused_tokens);
      const int generated = std::max(0, st.generated_tokens);
      metrics_.prompt_tokens_total += uint64_t(prefilled);
      metrics_.prompt_tokens_cached_total += uint64_t(cached);
      if (prefilled > 0) {
        const double psec = std::max(0.0, st.prefill_seconds);
        metrics_.prompt_seconds_total += psec;
        if (psec > 1e-6) last_prefill_tps_ = double(prefilled) / psec;
      }
      metrics_.tokens_predicted_total += uint64_t(generated);
      metrics_.tokens_predicted_seconds_total += std::max(0.0, st.decode_seconds);
      const uint64_t decode_calls =
          uint64_t(prefilled > 0 ? 1 : 0) + uint64_t(st.spec_rounds > 0 ? st.spec_rounds : generated);
      metrics_.n_decode_total += decode_calls;
      const uint64_t seq_len = uint64_t(std::max(0, st.prompt_tokens + generated));
      if (seq_len > metrics_.n_tokens_max) metrics_.n_tokens_max = seq_len;
      if (st.spec_rounds > 0) {
        metrics_.spec_decode_num_drafts_total += uint64_t(st.spec_rounds);
        metrics_.spec_decode_num_draft_tokens_total += uint64_t(st.spec_rounds) * 7ULL;
        metrics_.spec_decode_num_accepted_tokens_total += uint64_t(std::max(0, st.spec_accepted));
        for (std::size_t i = 0; i < st.spec_accepted_per_pos.size(); ++i) {
          metrics_.spec_accepted_per_pos[i] += uint64_t(std::max(0, st.spec_accepted_per_pos[i]));
        }
      }
      last_context_used_ = st.prompt_tokens + generated;
      RequestRecord rec;
      rec.timestamp_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                             std::chrono::system_clock::now().time_since_epoch())
                             .count();
      rec.id = job.id;
      rec.status = generated >= job.max_new_tokens ? "length" : "done";
      rec.prompt_tokens = st.prompt_tokens;
      rec.reused_tokens = st.reused_tokens;
      rec.output_tokens = generated;
      rec.decode_tps = st.decode_seconds > 1e-6 ? double(generated) / st.decode_seconds : 0.0;
      rec.duration_s = st.prefill_seconds + st.decode_seconds;
      rec.ttft_ms = st.prefill_seconds * 1000.0;
      rec.spec_rounds = st.spec_rounds;
      rec.spec_accepted = st.spec_accepted;
      recent_.push_front(std::move(rec));
      if (recent_.size() > 50) recent_.pop_back();
    }
    // prefix reuse of the prompt: the path, the tokens reused and prefilled, the prefill time
    server_log().info("{}: reuse={} reused={} prefilled={} prefill={:.3f}s", job.id, model::to_string(st.reuse),
                      st.reused_tokens, st.prompt_tokens - st.reused_tokens, st.prefill_seconds);
  } catch (const std::exception& e) {  // report to the client instead of killing the worker
    server_log().error("generator failed: {}: {}", type_name(e), e.what());
    job.fail("generation failed: " + type_name(e) + ": " + e.what());
    std::lock_guard lock(mutex_);
    RequestRecord rec;
    rec.timestamp_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                           std::chrono::system_clock::now().time_since_epoch())
                           .count();
    rec.id = job.id;
    rec.status = "error";
    rec.prompt_tokens = int(job.prompt_ids.size());
    recent_.push_front(std::move(rec));
    if (recent_.size() > 50) recent_.pop_back();
  } catch (...) {
    server_log().error("generator failed: unknown exception");
    job.fail("generation failed: unknown exception");
  }
}

}  // namespace ingine::engine
