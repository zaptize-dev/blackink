// Teacher-forced replay (Sampling::forced) with the full model: replaying a speculative greedy generation's
// own output must emit exactly that output and repeat its rounds (same drafts, widths and acceptances: the
// forced token after each accepted run is the target's argmax there); a plain generator replays it too.
#include "common/test_util.h"
#include "ingine/generator.h"

#include <algorithm>
#include <cstdio>
#include <filesystem>
#include <numeric>
#include <vector>

using namespace ingine::test;
namespace model = ingine::model;

namespace {

int g_failures = 0;

void expect(bool ok, const char* what) {
  std::printf("%s %s\n", ok ? "PASS" : "FAIL", what);
  if (!ok) ++g_failures;
}

// the start of a captured agent prompt (system message)
const std::vector<std::int32_t> kPrompt = {
    200002, 200004, 1735,   149914, 220,    65281,  6,      82,     194368, 163872, 171159, 13,     68014,  2,
    154675, 198,    12,     167508, 1255,   11,     150126, 151894, 12,     2409,   152725, 69137,  13,     165179,
    154393, 152059, 26,     152293, 220,    67211,  150665, 149882, 150261, 1528,   183763, 13,     198,    12,
    194063, 67,     150406, 25,     199531, 152297, 784,    168047, 11,     152145, 279,    11,     167527, 13,
    198,    12,     150058, 139526, 166183, 151122, 149914, 149876, 150409, 6,      82,     26,     200010, 200000,
    200004, 3,      200010, 200001};

model::GenerateStats run(model::Generator& gen, int tokens, const model::Sampling& s, std::vector<std::int32_t>& out) {
  out.clear();
  return gen.generate(kPrompt, tokens, s, {}, [&](std::span<const std::int32_t> t) {
    out.insert(out.end(), t.begin(), t.end());
    return true;
  });
}

}  // namespace

int main() {
  if (skip_without_gpu()) return kSkip;
  const std::filesystem::path dir = INGINE_MODEL_DIR;
  if (!std::filesystem::exists(dir / "mtp.safetensors")) {
    std::printf("skip: no model in %s\n", dir.c_str());
    return kSkip;
  }
  model::RuntimeOptions o;
  o.model_dir = dir;
  o.max_ctx = 8192;
  o.dense_fp8 = true;
  o.kv_fp8 = true;
  const int tokens = 160;
  std::vector<std::int32_t> greedy_out, out;
  {
    o.spec = true;
    auto gen = model::make_generator(o);
    model::Sampling greedy;
    greedy.temperature = 0.f;
    const model::GenerateStats ref = run(*gen, tokens, greedy, greedy_out);
    model::Sampling forced = greedy;
    forced.forced = greedy_out;
    const model::GenerateStats st = run(*gen, tokens, forced, out);
    std::printf("greedy: %d rounds, %d accepted; forced: %d rounds, %d accepted, %zu traced\n", ref.spec_rounds,
                ref.spec_accepted, st.spec_rounds, st.spec_accepted, st.spec_trace.size());
    expect(out == greedy_out, "speculative forced replay emits the forced output");
    // The greedy run's last round may accept drafts past the token budget (1 + rounds + accepted tokens were
    // produced, the first `tokens` emitted); the replay has no forced tokens there to accept.
    const int past_budget = std::max(0, ref.spec_rounds + ref.spec_accepted - tokens);
    expect(st.spec_rounds == ref.spec_rounds && st.spec_accepted == ref.spec_accepted - past_budget,
           "forced replay of the greedy output repeats its rounds");
    int traced = 0;
    for (const auto& r : st.spec_trace) traced += r.accepted;
    bool sane = st.spec_trace.size() == std::size_t(st.spec_rounds) && traced == st.spec_accepted;
    for (const auto& r : st.spec_trace)
      sane &= (r.width == 2 || r.width == 4 || r.width == 8) && r.accepted < r.width && r.accepted <= r.accepted_all &&
              (r.accepted_all == r.accepted || r.accepted == r.width - 1) && r.verify_ms > 0 && r.draft_ms > 0;
    expect(sane, "trace: one record per round, widths 2/4/8, accepted <= accepted_all, timings");
    // a forced output the model would not produce: every token emitted as given
    std::vector<std::int32_t> other(greedy_out.rbegin(), greedy_out.rend());
    forced.forced = other;
    run(*gen, tokens, forced, out);
    expect(out == other, "speculative forced replay of an arbitrary output");
  }
  {
    o.spec = false;
    auto gen = model::make_generator(o);
    model::Sampling forced;
    forced.temperature = 1.f;
    forced.seed = 7;
    forced.forced = greedy_out;
    const model::GenerateStats st = run(*gen, tokens, forced, out);
    expect(out == greedy_out && st.spec_trace.empty(), "plain forced replay emits the forced output");
  }
  std::printf("%s: %d failure(s)\n", g_failures ? "FAILED" : "OK", g_failures);
  return g_failures ? 1 : 0;
}
