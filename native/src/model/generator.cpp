// The Generator (engine.Engine): one resident model, one request at a time. Prefill runs in chunks through
// the Prefiller; plain decoding replays the width-1 graph per token, pipelined (each step's input token
// stays on the device, the host launches the next step before it reads the current token); with
// options.spec the speculative loop (SpecLoop) drafts with DSpark and verifies with the width-2/4/8 graphs.
// A prompt that extends the tokens already in the state only prefills the new suffix; any other prompt
// starts from an empty state.
#include "ingine/generator.h"

#include "core/arena.h"
#include "core/device.h"
#include "kernels/head/head.h"
#include "model/drafter.h"
#include "model/prefill.h"
#include "model/runtime.h"
#include "model/spec.h"

#include <algorithm>
#include <array>
#include <chrono>
#include <cstdio>
#include <mutex>
#include <random>
#include <stdexcept>
#include <string>

namespace ingine::model {
namespace {

namespace head = kernels::head;

double seconds_since(std::chrono::steady_clock::time_point t) {
  return std::chrono::duration<double>(std::chrono::steady_clock::now() - t).count();
}

const RuntimeOptions& without_spec(const RuntimeOptions& options) {
  if (options.spec) throw std::invalid_argument("make_generator over given weights: spec needs the model directory");
  return options;
}

class EngineGenerator final : public Generator {
 public:
  explicit EngineGenerator(const RuntimeOptions& options)
      : EngineGenerator(options, std::make_unique<Runtime>(options)) {}
  EngineGenerator(const RuntimeOptions& options, Weights weights, Ablation ablation)
      : EngineGenerator(without_spec(options),
                        std::make_unique<Runtime>(options, std::move(weights), std::move(ablation))) {}

  std::string model_id() const override {
    return options_.ablation.empty() ? "inkling-2-mini" : "inkling-2-mini-abliterated";
  }

  int max_model_len() const override { return options_.max_ctx; }

 private:
  EngineGenerator(const RuntimeOptions& options, std::unique_ptr<Runtime> runtime)
      : options_(options),
        runtime_(std::move(runtime)),
        rt_(*runtime_),
        prefiller_(rt_.weights(), rt_.ablation(), options.prefill_chunk, rt_.stream()),
        chunk_tokens_(std::size_t(options.prefill_chunk) * sizeof(std::int32_t)),
        logits_(std::size_t(UNPADDED_VOCAB) * sizeof(float)),
        token_(sizeof(std::int32_t)),
        host_token_(sizeof(std::int32_t)) {
    if (options.prefill_chunk < 1) throw std::invalid_argument("prefill_chunk must be positive");
    if (options.spec) {
      drafter_ = std::make_unique<Drafter>(options.model_dir, rt_.weights(), options.max_ctx, rt_.stream());
      drafter_->capture();
      taps_ = DeviceBuffer(std::size_t(options.prefill_chunk) * NUM_TAPS * HIDDEN * 2);
      spec_ = std::make_unique<SpecLoop>(rt_, *drafter_);
    }
    // checkpoint layout: target state | drafter context | last prompt logits (pinned host RAM saves 165 MiB VRAM)
    checkpoint_drafter_ = carved(rt_.checkpoint_bytes());
    checkpoint_logits_ = checkpoint_drafter_ + (drafter_ ? carved(drafter_->checkpoint_bytes()) : 0);
    checkpoint_ = DeviceBuffer(checkpoint_logits_ + logits_.bytes, /*host_pinned=*/true);
    CUDA_CHECK(cudaStreamSynchronize(rt_.stream()));
    if (options.park_mb < 0) {  // leave room for the workspaces a long prefill adds later
      std::size_t free = 0, total = 0;
      CUDA_CHECK(cudaMemGetInfo(&free, &total));
      park_budget_ = std::min<std::size_t>(3072, std::max<std::size_t>(free >> 20, 4096) - 4096) << 20;
    } else {
      park_budget_ = std::size_t(options.park_mb) << 20;
    }
    // parked conversations are best-effort: a device allocation that fails releases them and retries
    release_hook_ = add_release_hook([this] { return release_parked(); });
  }

 public:
  GenerateStats generate(std::span<const std::int32_t> prompt, int max_new_tokens, const Sampling& sampling,
                         std::span<const std::int32_t> stop_ids,
                         const std::function<bool(std::span<const std::int32_t>)>& on_tokens) override {
    return generate_multimodal(prompt, {}, max_new_tokens, sampling, stop_ids, on_tokens);
  }

  GenerateStats generate_multimodal(std::span<const std::int32_t> prompt, std::span<const ImageSpan> images,
                                    int max_new_tokens, const Sampling& sampling,
                                    std::span<const std::int32_t> stop_ids,
                                    const std::function<bool(std::span<const std::int32_t>)>& on_tokens) override {
    if (prompt.empty()) throw std::invalid_argument("prompt is empty");
    const int n = int(prompt.size());
    if (!sampling.forced.empty()) max_new_tokens = std::min(max_new_tokens, int(sampling.forced.size()));
    // a verify call writes up to max(verify widths) positions
    const int reserve = drafter_ ? *std::max_element(rt_.verify_widths().begin(), rt_.verify_widths().end()) : 0;
    if (n + std::int64_t(max_new_tokens) + reserve > options_.max_ctx) {
      max_new_tokens = options_.max_ctx - n - reserve;
      if (max_new_tokens <= 0)
        throw std::invalid_argument("prompt of " + std::to_string(n) + " tokens does not fit max_ctx " +
                                    std::to_string(options_.max_ctx));
    }
    GenerateStats stats;
    head::SamplerRng rng;
    rng.seed = sampling.seed ? *sampling.seed : (std::uint64_t(std::random_device{}()) << 32) ^ std::random_device{}();
    std::lock_guard<std::mutex> lock(mutex_);

    auto t = std::chrono::steady_clock::now();
    try {
      prefill(prompt, images, stats);
      stats.prompt_tokens = n;
      if (spec_) {
        std::int32_t token = sample(static_cast<const float*>(logits_.p), sampling, rng);
        if (!sampling.forced.empty()) token = sampling.forced[0];
        stats.prefill_seconds = seconds_since(t);
        t = std::chrono::steady_clock::now();
        if (max_new_tokens > 0) {
          drafter_->set_seed(rng.seed);
          spec_->generate(token, max_new_tokens, sampling, rng, stop_ids, on_tokens, stats, tokens_in_state_);
        }
      } else {
        decode(n, max_new_tokens, sampling, rng, stop_ids, on_tokens, stats, t);
      }
      CUDA_CHECK(cudaStreamSynchronize(rt_.stream()));
    } catch (...) {
      forget_state();
      throw;
    }
    stats.decode_seconds = seconds_since(t);
    return stats;
  }

  ~EngineGenerator() override {
    remove_release_hook(release_hook_);
    // nothing queued may outlive the graphs and buffers of the members destroyed below
    if (rt_.stream()) cudaStreamSynchronize(rt_.stream());
  }

 private:
  // After a failure anywhere in a request the device state may be ahead of or behind tokens_in_state_ (a step
  // in flight, a partial prefill, a round between launch and bookkeeping): wait for the stream and forget
  // everything reusable, so the next request starts from an empty state.
  void forget_state() noexcept {
    cudaStreamSynchronize(rt_.stream());
    tokens_in_state_.clear();
    checkpoint_ids_.clear();
    if (drafter_) drafter_->reset();  // stream-ordered; CUDA failures abort (CUDA_CHECK)
  }

  // Plain decoding after the prefill of n tokens (logits in logits_): the token at position p is sampled on
  // the device, chain_advance makes it the width-1 step's input, and the step at p is launched before the
  // host reads that token, so the GPU never waits for the host. A step launched for a token that ends the
  // generation (stop token, cancel) is undone, leaving the state at tokens_in_state_. Sets prefill_seconds
  // (until the first token is known) and restarts `t` for the decode time.
  void decode(int n, int max_new_tokens, const Sampling& sampling, head::SamplerRng& rng,
              std::span<const std::int32_t> stop_ids,
              const std::function<bool(std::span<const std::int32_t>)>& on_tokens, GenerateStats& stats,
              std::chrono::steady_clock::time_point& t) {
    const cudaStream_t stream = rt_.stream();
    const head::SamplingParams params{sampling.temperature, sampling.top_k, sampling.top_p};
    auto* sampled = static_cast<std::int32_t*>(token_.p);
    // forced replay: the forced tokens on the device; each sampled token is replaced by its forced one
    DeviceBuffer forced;
    if (!sampling.forced.empty()) {
      forced = DeviceBuffer(sampling.forced.size_bytes());
      CUDA_CHECK(cudaMemcpyAsync(forced.p, sampling.forced.data(), sampling.forced.size_bytes(),
                                 cudaMemcpyHostToDevice, stream));
    }
    // engine._sample of the logits into `sampled`, then the inputs of the step at `position`
    auto sample_for = [&](const float* logits, int position) {
      head::sample(logits, 1, UNPADDED_VOCAB, UNPADDED_VOCAB, params, rng, sampled, stream);
      if (forced.p)
        CUDA_CHECK(cudaMemcpyAsync(sampled, static_cast<const std::int32_t*>(forced.p) + (position - n),
                                   sizeof(std::int32_t), cudaMemcpyDeviceToDevice, stream));
      rt_.chain_advance(sampled, position);
    };
    // the step at `position` (its input token is on the device), then the next token
    auto launch = [&](int position) { sample_for(rt_.run_step(1), position + 1); };
    auto is_stop = [&](std::int32_t tok) { return std::find(stop_ids.begin(), stop_ids.end(), tok) != stop_ids.end(); };

    sample_for(static_cast<const float*>(logits_.p), n);
    if (max_new_tokens > 1) launch(n);
    std::int32_t token = rt_.chain_token(n);
    stats.prefill_seconds = seconds_since(t);
    t = std::chrono::steady_clock::now();
    for (int i = 0; i < max_new_tokens; ++i) {
      // `token` is at position n + i; its step is in flight unless it is the budget's last token
      const bool last = i + 1 == max_new_tokens;
      stats.generated_tokens = i + 1;
      const std::int32_t out[1] = {token};
      const bool go_on = on_tokens(out);
      if (last || is_stop(token) || !go_on) {
        if (!last) rt_.chain_undo(n + i);
        break;
      }
      tokens_in_state_.push_back(token);
      if (i + 2 < max_new_tokens) launch(n + i + 1);
      token = rt_.chain_token(n + i + 1);
    }
  }

  // engine._prefill with prompt checkpoints: brings the state to `ids` (last logits in logits_) and sets
  // stats.reused_tokens and stats.reuse.
  // * Extend: `ids` extends tokens_in_state_ (the previous prompt and the generated tokens processed since):
  //   prefill the new suffix.
  // * Checkpoint: the prompt of the request that saved the checkpoint (the state right after its prefill) is a
  //   prefix of `ids`: restore it and prefill the rest. This is the agent-loop case: the next prompt
  //   re-renders the previous answer differently from the tokens the model generated.
  // * Parked: a parked conversation's prompt is a longer prefix of `ids` than the live state offers: park the
  //   live conversation, restore the parked one's checkpoint and prefill the rest.
  // * None: park the live conversation, reset and prefill everything.
  // After a prefill the checkpoint moves to the end of `ids`.
  static std::uint64_t hash_images(std::span<const ImageSpan> images) {
    std::uint64_t h = 14695981039346656037ULL;
    auto mix = [&](const void* p, std::size_t n) {
      const auto* b = static_cast<const std::uint8_t*>(p);
      for (std::size_t i = 0; i < n; ++i) h = (h ^ b[i]) * 1099511628211ULL;
    };
    for (const auto& img : images) {
      mix(&img.offset, sizeof(img.offset));
      mix(&img.n_tokens, sizeof(img.n_tokens));
      if (!img.data.empty()) mix(img.data.data(), std::min<std::size_t>(img.data.size() * 4, 4096));
    }
    return h;
  }

  void prefill(std::span<const std::int32_t> ids, std::span<const ImageSpan> images, GenerateStats& stats) {
    const std::vector<std::int32_t>& prev = tokens_in_state_;
    const std::size_t n = ids.size();
    const std::uint64_t img_hash = images.empty() ? 0 : hash_images(images);
    const bool same_images = (img_hash == images_hash_);
    const bool drafter_in_step = !drafter_ || drafter_->length() == int(prev.size());  // else a row is missing
    State& st = rt_.state();
    const cudaStream_t stream = rt_.stream();
    const bool extend =
        same_images && drafter_in_step && !prev.empty() && prev.size() < n &&
        std::equal(prev.begin(), prev.end(), ids.begin());
    const bool checkpoint = same_images && !extend && !checkpoint_ids_.empty() && checkpoint_ids_.size() <= n &&
                            std::equal(checkpoint_ids_.begin(), checkpoint_ids_.end(), ids.begin());
    Parked* parked = (images.empty() && images_hash_ == 0)
                         ? find_parked(ids, extend ? prev.size() : checkpoint ? checkpoint_ids_.size() : 0)
                         : nullptr;
    std::size_t start = 0;
    if (parked) {
      // parking the live conversation may move the parked list (never evicting the one to restore)
      const std::vector<std::int32_t> keep = parked->ids;
      park_live(&keep);
      unpark(*std::find_if(parked_.begin(), parked_.end(), [&](const Parked& p) { return p.ids == keep; }));
      restore_checkpoint();
      start = checkpoint_ids_.size();
      stats.reuse = Reuse::Parked;
    } else if (extend) {
      start = prev.size();
      stats.reuse = Reuse::Extend;
    } else if (checkpoint) {
      restore_checkpoint();
      start = checkpoint_ids_.size();
      stats.reuse = Reuse::Checkpoint;
    } else {
      park_live();
      reset_state(st, stream);
      if (drafter_) drafter_->reset();
      tokens_in_state_.clear();
      checkpoint_ids_.clear();
      stats.reuse = Reuse::None;
    }
    st.length = int(start);  // decode steps advance the state without its host-side length
    const std::size_t chunk = std::size_t(options_.prefill_chunk);
    for (std::size_t s = start; s < n; s += chunk) {
      const int T = int(std::min(chunk, n - s));
      CUDA_CHECK(cudaMemcpyAsync(chunk_tokens_.p, ids.data() + s, std::size_t(T) * sizeof(std::int32_t),
                                 cudaMemcpyHostToDevice, stream));
      Tensor taps;
      if (drafter_) taps = rm(taps_.p, DType::BF16, {T, NUM_TAPS, HIDDEN});
      prefiller_.run(st, static_cast<const std::int32_t*>(chunk_tokens_.p), T, int(s), static_cast<float*>(logits_.p),
                     drafter_ ? &taps : nullptr, [&] {
                       std::vector<EmbedOverride> ovs;
                       for (const auto& img : images) {
                         const int a = std::max(int(s), img.offset);
                         const int b = std::min(int(s) + T, img.offset + img.n_tokens);
                         if (a < b) {
                           const int local_off = a - int(s);
                           const int img_off = a - img.offset;
                           ovs.push_back({local_off, b - a, img.data.data() + std::size_t(img_off) * HIDDEN});
                         }
                       }
                       return ovs;
                     }());
      if (drafter_) {
        drafter_->note_tokens(ids.subspan(s, std::size_t(T)));  // the prompt's tokens join the drafter's shortlist
        drafter_->commit(taps, int(s));
      }
    }
    images_hash_ = img_hash;
    tokens_in_state_.assign(ids.begin(), ids.end());
    if (start < n) save_checkpoint();
    stats.reused_tokens = int(start);
  }

  // Conversation parking. A parked conversation is a packed copy of a prompt checkpoint (target state |
  // drafter checkpoint regions | last logits) plus the target's full-attention rows [0, n) of its n prompt
  // tokens, in device memory within park_budget_ (least recently used evicted), and the drafter's context
  // tokens of the checkpoint (its shortlist rows).
  struct Parked {
    std::vector<std::int32_t> ids, context;
    DeviceBuffer data;
    std::uint64_t used = 0;
  };
  static constexpr std::size_t kMinParked = 256;  // shorter prompts are cheaper to prefill than to keep

  // the layout of a parked copy of n tokens: {offset, bytes} of each drafter region, logits, full rows
  struct ParkLayout {
    std::array<Drafter::Region, 3> drafter{};
    std::size_t drafter_off[3] = {}, logits = 0, rows = 0, bytes = 0;
  };
  ParkLayout park_layout(int n) const {
    ParkLayout l;
    std::size_t off = checkpoint_drafter_;
    if (drafter_) {
      l.drafter = drafter_->checkpoint_regions(n);
      for (int i = 0; i < 3; ++i) l.drafter_off[i] = off, off += carved(l.drafter[i].bytes);
    }
    l.logits = off;
    l.rows = off + carved(logits_.bytes);
    l.bytes = l.rows + rt_.full_rows_bytes(n);
    return l;
  }

  // the parked conversation whose prompt is the longest prefix of `ids`, if longer than `live` (and kMinParked)
  Parked* find_parked(std::span<const std::int32_t> ids, std::size_t live) {
    Parked* best = nullptr;
    for (Parked& p : parked_)
      if (p.ids.size() >= kMinParked && p.ids.size() > live && p.ids.size() <= ids.size() &&
          (!best || p.ids.size() > best->ids.size()) && std::equal(p.ids.begin(), p.ids.end(), ids.begin()))
        best = &p;
    return best;
  }

  // The release hook (core/arena.h): a device allocation failed; drop every parked conversation so that it can
  // be retried. Runs on the allocating thread, i.e. inside a request (the Generator allocates nothing else).
  bool release_parked() {
    if (parking_ || parked_.empty()) return false;
    cudaStreamSynchronize(rt_.stream());  // copies from or into parked buffers may be queued
    std::fprintf(stderr, "device memory low: released %zu parked conversation(s)\n", parked_.size());
    parked_.clear();
    return true;
  }

  // Parks the live conversation's checkpoint (the state after its last prompt; the full rows [0, n) are still
  // in the caches) unless it is short, already parked or larger than the budget; the parked conversation with
  // prompt `keep` (if given) stays. Never fails the request: a failed allocation disables parking.
  void park_live(const std::vector<std::int32_t>* keep = nullptr) {
    const std::size_t n = checkpoint_ids_.size();
    if (park_budget_ == 0 || n < kMinParked) return;
    for (Parked& p : parked_)
      if (p.ids == checkpoint_ids_) {
        p.used = ++clock_;
        return;
      }
    auto kept = [&](const Parked& p) { return keep && p.ids == *keep; };
    // a parked prefix of this conversation is superseded by it
    std::erase_if(parked_, [&](const Parked& p) {
      return !kept(p) && p.ids.size() < n && std::equal(p.ids.begin(), p.ids.end(), checkpoint_ids_.begin());
    });
    const ParkLayout l = park_layout(int(n));
    std::size_t used = 0, kept_bytes = 0;
    for (const Parked& p : parked_) used += p.data.bytes, kept_bytes += kept(p) ? p.data.bytes : 0;
    if (kept_bytes + l.bytes > park_budget_) return;
    while (used + l.bytes > park_budget_) {  // least recently used first
      auto lru = parked_.end();
      for (auto it = parked_.begin(); it != parked_.end(); ++it)
        if (!kept(*it) && (lru == parked_.end() || it->used < lru->used)) lru = it;
      used -= lru->data.bytes;
      parked_.erase(lru);
    }
    Parked p;
    try {
      parking_ = true;  // a failure here must not release the parked conversations through the hook
      p.data = DeviceBuffer(l.bytes);
      parking_ = false;
    } catch (const std::exception& e) {
      parking_ = false;
      std::fprintf(stderr, "conversation parking disabled: %s\n", e.what());
      cudaGetLastError();  // the failed allocation is not a sticky error
      park_budget_ = 0;
      std::erase_if(parked_, [&](const Parked& q) { return !kept(q); });
      return;
    }
    char* d = static_cast<char*>(p.data.p);
    const char* c = static_cast<const char*>(checkpoint_.p);
    const cudaStream_t s = rt_.stream();
    CUDA_CHECK(cudaMemcpyAsync(d, c, checkpoint_drafter_, cudaMemcpyDeviceToDevice, s));
    for (int i = 0; i < 3 && drafter_; ++i)
      CUDA_CHECK(cudaMemcpyAsync(d + l.drafter_off[i], c + checkpoint_drafter_ + l.drafter[i].offset,
                                 l.drafter[i].bytes, cudaMemcpyDeviceToDevice, s));
    CUDA_CHECK(cudaMemcpyAsync(d + l.logits, c + checkpoint_logits_, logits_.bytes, cudaMemcpyDeviceToDevice, s));
    rt_.save_full_rows(d + l.rows, int(n));
    p.ids = checkpoint_ids_;
    if (drafter_) p.context = drafter_->checkpoint_context();
    p.used = ++clock_;
    parked_.push_back(std::move(p));
  }

  // Puts a parked conversation back as the prompt checkpoint (restore_checkpoint() then restores it) and its
  // full rows into the caches. It stays parked.
  void unpark(Parked& p) {
    const int n = int(p.ids.size());
    const ParkLayout l = park_layout(n);
    const char* d = static_cast<const char*>(p.data.p);
    char* c = static_cast<char*>(checkpoint_.p);
    const cudaStream_t s = rt_.stream();
    CUDA_CHECK(cudaMemcpyAsync(c, d, checkpoint_drafter_, cudaMemcpyDeviceToDevice, s));
    for (int i = 0; i < 3 && drafter_; ++i)
      CUDA_CHECK(cudaMemcpyAsync(c + checkpoint_drafter_ + l.drafter[i].offset, d + l.drafter_off[i],
                                 l.drafter[i].bytes, cudaMemcpyDeviceToDevice, s));
    CUDA_CHECK(cudaMemcpyAsync(c + checkpoint_logits_, d + l.logits, logits_.bytes, cudaMemcpyDeviceToDevice, s));
    rt_.restore_full_rows(d + l.rows, n);
    if (drafter_) drafter_->adopt_checkpoint(c + checkpoint_drafter_, n, p.context);
    checkpoint_ids_ = p.ids;
    p.used = ++clock_;
  }

  // The checkpoint (in pinned host memory): target state (Runtime::save_checkpoint) | drafter context
  // (Drafter::save_checkpoint) | logits of the last prompt token.
  void save_checkpoint() {
    char* p = static_cast<char*>(checkpoint_.p);
    rt_.save_checkpoint(p);
    if (drafter_) drafter_->save_checkpoint(p + checkpoint_drafter_);
    CUDA_CHECK(cudaMemcpyAsync(p + checkpoint_logits_, logits_.p, logits_.bytes, cudaMemcpyDefault,
                               rt_.stream()));
    checkpoint_ids_ = tokens_in_state_;
  }

  void restore_checkpoint() {
    const char* p = static_cast<const char*>(checkpoint_.p);
    rt_.restore_checkpoint(p);
    if (drafter_) drafter_->restore_checkpoint(p + checkpoint_drafter_);
    CUDA_CHECK(cudaMemcpyAsync(logits_.p, p + checkpoint_logits_, logits_.bytes, cudaMemcpyDefault,
                               rt_.stream()));
    tokens_in_state_ = checkpoint_ids_;
  }

  static std::size_t carved(std::size_t bytes) { return (bytes + 255) / 256 * 256; }

  // engine._sample with a host synchronisation (the speculative loop's first anchor): argmax at
  // temperature <= 0, else one draw from engine._probs.
  std::int32_t sample(const float* logits, const Sampling& sampling, head::SamplerRng& rng) {
    head::SamplingParams p;
    p.temperature = sampling.temperature;
    p.top_k = sampling.top_k;
    p.top_p = sampling.top_p;
    head::sample(logits, 1, UNPADDED_VOCAB, UNPADDED_VOCAB, p, rng, static_cast<std::int32_t*>(token_.p),
                 rt_.stream());
    CUDA_CHECK(cudaMemcpyAsync(host_token_.data(), token_.p, sizeof(std::int32_t), cudaMemcpyDeviceToHost,
                               rt_.stream()));
    CUDA_CHECK(cudaStreamSynchronize(rt_.stream()));
    return *static_cast<const std::int32_t*>(host_token_.data());
  }

  RuntimeOptions options_;
  std::unique_ptr<Runtime> runtime_;
  Runtime& rt_;
  Prefiller prefiller_;
  std::unique_ptr<Drafter> drafter_;
  std::unique_ptr<SpecLoop> spec_;
  DeviceBuffer chunk_tokens_, logits_, token_, taps_;
  PinnedHostBuffer host_token_;
  // prompt checkpoint (save_checkpoint) and the prompt it holds (empty: none)
  DeviceBuffer checkpoint_;
  std::size_t checkpoint_drafter_ = 0, checkpoint_logits_ = 0;  // byte offsets in checkpoint_
  std::vector<std::int32_t> checkpoint_ids_;
  std::vector<std::int32_t> tokens_in_state_;
  std::uint64_t images_hash_ = 0;
  std::vector<Parked> parked_;
  std::size_t park_budget_ = 0;  // bytes
  std::uint64_t clock_ = 0;      // last use of a parked conversation
  bool parking_ = false;         // park_live is allocating
  int release_hook_ = -1;
  std::mutex mutex_;
};

}  // namespace

std::unique_ptr<Generator> make_generator(const RuntimeOptions& options) {
  return std::make_unique<EngineGenerator>(options);
}

std::unique_ptr<Generator> make_generator(const RuntimeOptions& options, Weights weights, Ablation ablation) {
  return std::make_unique<EngineGenerator>(options, std::move(weights), std::move(ablation));
}

}  // namespace ingine::model
