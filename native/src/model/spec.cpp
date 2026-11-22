#include "model/spec.h"

#include "core/device.h"
#include "kernels/draft/draft.h"
#include "model/model_kernels.h"

#include <algorithm>
#include <cstring>
#include <stdexcept>

namespace ingine::model {

namespace kd = kernels::draft;
namespace kh = kernels::head;

namespace {

std::uint64_t splitmix64(std::uint64_t x) {
  x += 0x9E3779B97F4A7C15ull;
  x = (x ^ (x >> 30)) * 0xBF58476D1CE4E5B9ull;
  x = (x ^ (x >> 27)) * 0x94D049BB133111EBull;
  return x ^ (x >> 31);
}

static_assert(sizeof(VerifyRecord::drafts) / sizeof(std::int32_t) == kd::GAMMA);

Widths verify_widths(const Runtime& rt) {
  const std::span<const int> w = rt.verify_widths();
  if (w.empty() || w.size() > 3) throw std::invalid_argument("SpecLoop: the runtime has no verify widths (spec off)");
  Widths r;
  r.n = int(w.size());
  std::copy(w.begin(), w.end(), r.w);
  if (r.w[r.n - 1] < kd::BLOCK) throw std::invalid_argument("SpecLoop: the widest verify call must hold a block");
  return r;
}

}  // namespace

SpecLoop::SpecLoop(Runtime& rt, Drafter& drafter)
    : rt_(rt), drafter_(drafter), width_(sizeof(std::int32_t)), record_(sizeof(VerifyRecord)),
      forced_window_(kd::BLOCK * sizeof(std::int32_t)) {
  if (drafter.stream() != rt.stream()) throw std::invalid_argument("SpecLoop: drafter and runtime streams differ");
  const Widths widths = verify_widths(rt);
  // the draft outputs live at fixed addresses: one draft names them (its scratch rows are reset)
  draft_ = drafter.draft(0, 0, 0.f);
  drafter.reset();
  width_.fill(0);
  const float threshold = rt.options().spec_threshold;
  verify_ = rt.capture_switch(
      [&](cudaStream_t s, cudaGraphConditionalHandle h) {
        verify_select(draft_.conf, draft_.tokens, threshold, widths, rt_.tokens(), static_cast<std::int32_t*>(width_.p),
                      h, s);
      },
      [&](cudaStream_t s, int m) {
        rt_.record_save_verify(m, s);
        rt_.record_step(m, s);
      });
  // where a round's outcome lives on the device: after the acceptance the drafter commits the accepted
  // rows and drafts for the next anchor from these (one graph, no host round trip in between)
  {
    const Drafter::AcceptScratch& scratch = drafter.accept_scratch();
    kd::RoundInputs round;
    round.nwidths = widths.n;
    for (int i = 0; i < widths.n; ++i) {
      round.widths[i] = widths.w[i];
      round.taps[i] = ptr<const __nv_bfloat16>(rt.buffers(widths.w[i]).taps);
    }
    round.m = static_cast<const std::int32_t*>(width_.p);
    round.accepted = &scratch.result->accepted;
    round.next_greedy = &scratch.result->next;
    round.next_sampled = &static_cast<VerifyRecord*>(record_.device_ptr())->next;  // sample_from_probs
    drafter.bind_round(round);
  }
  CUDA_CHECK(cudaEventCreateWithFlags(&done_, cudaEventDisableTiming));
  CUDA_CHECK(cudaStreamSynchronize(rt.stream()));
}

SpecLoop::~SpecLoop() {
  if (done_) cudaEventDestroy(done_);
}

void SpecLoop::prepare_accept(const Sampling& sampling) {
  const bool forced = !sampling.forced.empty();
  const bool greedy = sampling.temperature <= 0.f || forced;
  const AcceptKey key = greedy ? AcceptKey{true, forced, 0.f, 1.f, 0}
                               : AcceptKey{false, false, sampling.temperature, sampling.top_p, sampling.top_k};
  if (accept_ && key == accept_key_) return;
  const Widths widths = verify_widths(rt_);
  const int V = drafter_.vocab();
  const Drafter::AcceptScratch& scratch = drafter_.accept_scratch();
  const kh::SamplingParams params{sampling.temperature, sampling.top_k, sampling.top_p};
  const auto* width = static_cast<const std::int32_t*>(width_.p);
  auto* record = static_cast<VerifyRecord*>(record_.device_ptr());
  // forced replay: the drafts are compared with the forced tokens (the argmax still runs, for the timing)
  const auto* target = forced ? static_cast<const std::int32_t*>(forced_window_.device_ptr()) : scratch.target;
  accept_ = GraphExec();
  accept_ = rt_.capture_switch(
      [&](cudaStream_t s, cudaGraphConditionalHandle h) { width_select(width, widths, h, s); },
      [&](cudaStream_t s, int m) {
        // engine._accept over the m verify rows (anchor + m - 1 drafts)
        const float* logits = static_cast<const float*>(rt_.buffers(m).logits.data);
        const int n = m - 1;
        if (greedy) {
          kh::argmax(logits, m, V, V, scratch.target, s);
          kd::greedy_accept(target, draft_.tokens, n, scratch.result, s);
        } else {
          kh::sampling_probs(logits, m, V, V, params, scratch.probs, s);
          kd::sampled_accept(scratch.probs, draft_.logits, V, draft_.tokens, n, sampling.temperature, draft_.qstats,
                             scratch.uniforms, scratch.ws, scratch.resid, scratch.result, s);
        }
        rt_.record_rollback(m, &scratch.result->accepted, s);
        verify_publish(width, &scratch.result->accepted, greedy ? &scratch.result->next : nullptr, rt_.tokens(), record,
                       s);
      });
  accept_key_ = key;
}

void SpecLoop::generate(std::int32_t token, int max_new_tokens, const Sampling& sampling, kh::SamplerRng& rng,
                        std::span<const std::int32_t> stop_ids,
                        const std::function<bool(std::span<const std::int32_t>)>& on_tokens, GenerateStats& stats,
                        std::vector<std::int32_t>& tokens_in_state) {
  const cudaStream_t stream = rt_.stream();
  int P = int(tokens_in_state.size());  // position of the anchor (sampled, not yet processed)
  if (drafter_.length() != P) throw std::logic_error("SpecLoop: drafter context does not match the state");

  const bool forced = !sampling.forced.empty();
  const bool greedy = sampling.temperature <= 0.f || forced;  // forced: acceptance by equality
  const int V = drafter_.vocab();
  const Drafter::AcceptScratch& scratch = drafter_.accept_scratch();
  auto* record_dev = static_cast<VerifyRecord*>(record_.device_ptr());
  const auto* record = static_cast<const VerifyRecord*>(record_.data());
  prepare_accept(sampling);
  if (sampling.temperature > 0.f) drafter_.set_seed(splitmix64(rng.seed ^ splitmix64(rng.counter++)));

  int emitted = 0, rounds = 0, accepted = 0;
  std::vector<std::int32_t> out{token};
  // the last verify call (none before the first round): its position, width and accepted drafts
  int round_P = -1, round_m = 0, round_acc = 0;
  // forced replay: per round, events around the verify + acceptance and around the next draft
  struct RoundEvents {
    cudaEvent_t e[3];
  };
  std::vector<RoundEvents> events;
  auto record_event = [&](int i) {
    CUDA_CHECK(cudaEventCreate(&events.back().e[i]));
    CUDA_CHECK(cudaEventRecord(events.back().e[i], stream));
  };
  // the first draft; later ones are enqueued by the previous round (they leave the target state alone, so
  // the last one is wasted when the generation ends) and run while the host emits
  drafter_.draft(token, P, sampling.temperature);
  for (;;) {
    const std::int32_t anchor = out.back();

    // emit this round's tokens up to a stop token or the budget
    std::size_t k = 0;
    bool done = false;
    while (k < out.size()) {
      const std::int32_t tok = out[k++];
      ++emitted;
      if (std::find(stop_ids.begin(), stop_ids.end(), tok) != stop_ids.end() || emitted == max_new_tokens) {
        done = true;
        break;
      }
    }
    stats.generated_tokens = emitted;
    const bool go_on = on_tokens(std::span<const std::int32_t>(out.data(), k));
    drafter_.note_tokens(std::span<const std::int32_t>(out.data(), k));  // for the drafts after the next one
    if (done || !go_on) {
      // The last emitted token stays unprocessed, as in plain decoding. If the generation ended inside the
      // round's accepted drafts (out[k - 1] with k - 1 < accepted), the state, the drafter and
      // tokens_in_state hold drafts past it: keep the verify call's first k rows (anchor + k - 1 drafts).
      const int keep = int(k) - 1;
      if (round_P >= 0 && keep < round_acc) {
        rt_.rollback(round_P, keep + 1, round_m);
        drafter_.commit(ptr<const __nv_bfloat16>(rt_.buffers(round_m).taps), keep + 1, round_P);
        tokens_in_state.resize(tokens_in_state.size() - std::size_t(round_acc - keep));
      }
      break;
    }

    // verify the confident drafts in the width the device picks, accept, roll back the rejected rows; then
    // commit the accepted rows and draft for the next anchor, all from device values
    const std::int32_t inputs[1] = {anchor};
    rt_.set_inputs(inputs, P);
    if (forced) {  // the forced tokens after the anchor (forced[emitted - 1]); the accept graph reads them
      auto* w = static_cast<std::int32_t*>(forced_window_.data());
      for (int i = 0; i < kd::BLOCK; ++i)
        w[i] = std::size_t(emitted + i) < sampling.forced.size() ? sampling.forced[emitted + i] : 0;
      events.emplace_back();
      record_event(0);
    }
    if (!greedy) kh::uniform(kd::GAMMA, rng, scratch.uniforms, stream);
    verify_.launch(stream);
    accept_.launch(stream);
    if (!greedy) kh::sample_from_probs(scratch.resid, 1, V, V, rng, &record_dev->next, stream);
    CUDA_CHECK(cudaEventRecord(done_, stream));
    if (forced) record_event(1);
    drafter_.advance(P);
    if (forced) record_event(2);
    CUDA_CHECK(cudaEventSynchronize(done_));  // the round's only host synchronisation
    VerifyRecord r;
    std::memcpy(&r, record, sizeof r);
    const int acc = r.accepted;
    const int commit = acc + 1;
    round_P = P;
    round_m = r.m;
    round_acc = acc;
    if (forced) {  // the run all 7 drafts would have reached
      int all = 0;
      while (all < kd::GAMMA && std::size_t(emitted + all) < sampling.forced.size() &&
             r.drafts[all] == sampling.forced[emitted + all])
        ++all;
      stats.spec_trace.push_back({r.m, acc, all, 0.f, 0.f});
    }

    drafter_.advanced(commit);
    tokens_in_state.push_back(anchor);
    tokens_in_state.insert(tokens_in_state.end(), r.drafts, r.drafts + acc);
    P += commit;
    ++rounds;
    accepted += acc;
    out.assign(r.drafts, r.drafts + acc);
    out.push_back(r.next);
  }
  if (forced) {
    CUDA_CHECK(cudaStreamSynchronize(stream));
    for (std::size_t i = 0; i < events.size(); ++i) {
      if (i < stats.spec_trace.size()) {
        CUDA_CHECK(cudaEventElapsedTime(&stats.spec_trace[i].verify_ms, events[i].e[0], events[i].e[1]));
        CUDA_CHECK(cudaEventElapsedTime(&stats.spec_trace[i].draft_ms, events[i].e[1], events[i].e[2]));
      }
      for (cudaEvent_t e : events[i].e) cudaEventDestroy(e);
    }
  }
  stats.spec_rounds = rounds;
  stats.spec_accepted = accepted;
}

}  // namespace ingine::model
