// Teacher-forced replay: the engine runs its real decode loop along a fixed output (Sampling::forced), so
// builds and settings are compared on identical text. Speculative rounds are traced (verify width, accepted
// drafts, the run all 7 drafts would have reached, GPU ms of the verify and of the draft) and summarised per
// output kind; the oracle estimate re-prices every round at the width that verifies exactly its run + 1.
// With a free seed the items' prompts are continued freely instead (run_free): sampled speed and tokens/verify.
#include "forced.h"

#include <nlohmann/json.hpp>

#include <algorithm>
#include <cstdio>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace ingine::bench {
namespace {

enum Kind { REASONING, TEXT, TOOL, OTHER, NKINDS };
constexpr const char* KIND_NAMES[NKINDS] = {"reasoning", "text", "tool", "other"};

// The content kind of every output token: the last content marker at or before it.
std::vector<int> kinds_of(const std::vector<std::int32_t>& out) {
  std::vector<int> kinds(out.size());
  int kind = OTHER;
  for (std::size_t i = 0; i < out.size(); ++i) {
    switch (out[i]) {
      case 200008: kind = REASONING; break;           // <|content_thinking|>
      case 200004: kind = TEXT; break;                // <|content_text|>
      case 200049: case 200057: case 200024: kind = TOOL; break;  // tool json / tool text / xml
      default: break;
    }
    kinds[i] = kind;
  }
  return kinds;
}

constexpr int WIDTHS = 9;  // widths 0..8

struct Totals {
  long items = 0, tokens = 0, rounds = 0, round_tokens = 0, first_rejected = 0, oracle_tokens = 0;
  long width_rounds[WIDTHS] = {}, width_tokens[WIDTHS] = {};
  double seconds = 0, draft_ms = 0, verify_ms = 0, width_verify_ms[WIDTHS] = {};

  void add_round(const model::SpecRoundRecord& r) {
    ++rounds;
    round_tokens += r.accepted + 1;
    oracle_tokens += r.accepted_all + 1;
    first_rejected += r.accepted_all == 0;
    if (r.width >= 0 && r.width < WIDTHS) {
      ++width_rounds[r.width];
      width_tokens[r.width] += r.accepted + 1;
      width_verify_ms[r.width] += r.verify_ms;
    }
    draft_ms += r.draft_ms;
    verify_ms += r.verify_ms;
  }
};

int oracle_width(int run) {  // the smallest verify width holding the anchor and `run` drafts
  return run + 1 <= 2 ? 2 : (run + 1 <= 4 ? 4 : 8);
}

// Oracle and verify-all estimates for rounds `tr` priced with the measured verify ms per width of `all`
// (overall means) and this set's own draft ms and per-round host/other overhead.
void estimates(const Totals& t, const Totals& all, const std::vector<model::SpecRoundRecord>& tr, double& oracle_tps,
               double& all8_tps) {
  oracle_tps = all8_tps = 0;
  if (t.rounds == 0) return;
  const double overhead_ms = std::max(0.0, (t.seconds * 1e3 - t.draft_ms - t.verify_ms) / double(t.rounds));
  auto vms = [&](int w) { return all.width_rounds[w] ? all.width_verify_ms[w] / double(all.width_rounds[w]) : 0.0; };
  double o_ms = 0, a_ms = 0, o_tok = 0, a_tok = 0;
  for (const auto& r : tr) {
    const double base = overhead_ms + r.draft_ms;
    o_ms += base + vms(oracle_width(r.accepted_all));
    o_tok += r.accepted_all + 1;
    a_ms += base + vms(8);
    a_tok += r.accepted_all + 1;
  }
  oracle_tps = o_tok / (o_ms / 1e3);
  all8_tps = a_tok / (a_ms / 1e3);
}

void print_totals(const char* name, const Totals& t, const Totals& all, const std::vector<model::SpecRoundRecord>& tr) {
  const double tps = t.seconds > 0 ? double(t.tokens) / t.seconds : 0;
  std::printf("%-28s %6ld tok  %7.1f tok/s", name, t.tokens, tps);
  if (t.rounds) {
    double oracle = 0, all8 = 0;
    estimates(t, all, tr, oracle, all8);
    std::printf("  %5.2f tok/verify  first-rejected %4.1f%%  draft %.3f ms  verify %.3f ms  widths", double(t.round_tokens) / t.rounds,
                100.0 * t.first_rejected / t.rounds, t.draft_ms / t.rounds, t.verify_ms / t.rounds);
    for (int w = 2; w < WIDTHS; w *= 2)
      std::printf(" %d:%4.1f%%/%.2ft", w, 100.0 * t.width_rounds[w] / t.rounds,
                  t.width_rounds[w] ? double(t.width_tokens[w]) / t.width_rounds[w] : 0.0);
    std::printf("  oracle-width %.1f tok/s (%.2f tok/verify), verify-all %.1f tok/s", oracle,
                double(t.oracle_tokens) / t.rounds, all8);
  }
  std::printf("\n");
}

nlohmann::ordered_json to_json(const Totals& t, const Totals& all, const std::vector<model::SpecRoundRecord>& tr) {
  nlohmann::ordered_json j;
  j["tokens"] = t.tokens;
  j["tok_s"] = t.seconds > 0 ? double(t.tokens) / t.seconds : 0;
  if (t.rounds) {
    double oracle = 0, all8 = 0;
    estimates(t, all, tr, oracle, all8);
    j["rounds"] = t.rounds;
    j["tokens_per_verify"] = double(t.round_tokens) / t.rounds;
    j["first_rejected"] = double(t.first_rejected) / t.rounds;
    j["draft_ms"] = t.draft_ms / t.rounds;
    j["verify_ms"] = t.verify_ms / t.rounds;
    for (int w = 2; w < WIDTHS; w *= 2) {
      j["width_share"][std::to_string(w)] = double(t.width_rounds[w]) / t.rounds;
      j["width_tokens_per_verify"][std::to_string(w)] =
          t.width_rounds[w] ? double(t.width_tokens[w]) / t.width_rounds[w] : 0.0;
      j["width_verify_ms"][std::to_string(w)] = t.width_rounds[w] ? t.width_verify_ms[w] / t.width_rounds[w] : 0.0;
    }
    j["oracle_tokens_per_verify"] = double(t.oracle_tokens) / t.rounds;
    j["oracle_tok_s"] = oracle;
    j["verify_all_tok_s"] = all8;
  }
  return j;
}

struct Item {
  std::string name;
  std::vector<std::int32_t> prompt, output;
};

// Free generation from every item's prompt for len(output) tokens (no stop tokens) at the temperature, with
// the sampler seeded by `seed`: decode tokens/s and tokens/verify of sampled decoding. A forced replay at
// T > 0 under-counts sampled acceptance (a draft counts only if it equals the recorded token, where the loop
// accepts it with min(1, p/q)), so sampled A/B needs these runs; the texts differ between builds, compare
// several seeds.
std::string run_free(model::Generator& gen, const std::string& path, const std::vector<Item>& list, double temperature,
                     int seed, const std::string& trace_path) {
  long items = 0, tokens = 0, rounds = 0, accepted = 0;
  double seconds = 0;
  for (const Item& it : list) {
    if (it.output.size() < 2) continue;
    model::Sampling sampling;
    sampling.temperature = float(temperature);
    sampling.seed = std::uint64_t(seed);
    const model::GenerateStats st = gen.generate(it.prompt, int(it.output.size()), sampling, {},
                                                 [](std::span<const std::int32_t>) { return true; });
    const long tok = st.generated_tokens - 1;  // the first token comes with the prefill
    std::printf("%-28s %6ld tok  %7.1f tok/s", it.name.c_str(), tok, st.decode_seconds > 0 ? tok / st.decode_seconds : 0.0);
    if (st.spec_rounds) std::printf("  %5.2f tok/verify", double(st.spec_accepted + st.spec_rounds) / st.spec_rounds);
    std::printf("\n");
    std::fflush(stdout);
    if (!trace_path.empty()) {
      nlohmann::ordered_json row;
      row["item"] = it.name;
      row["free_seed"] = seed;
      row["tokens"] = tok;
      row["seconds"] = st.decode_seconds;
      row["spec_rounds"] = st.spec_rounds;
      row["spec_accepted"] = st.spec_accepted;
      std::ofstream(trace_path, std::ios::app) << row.dump() << "\n";
    }
    ++items;
    tokens += tok;
    seconds += st.decode_seconds;
    rounds += st.spec_rounds;
    accepted += st.spec_accepted;
  }
  nlohmann::ordered_json j;
  j["file"] = path;
  j["items"] = items;
  j["free_seed"] = seed;
  j["temperature"] = temperature;
  j["total"]["tokens"] = tokens;
  j["total"]["tok_s"] = seconds > 0 ? double(tokens) / seconds : 0;
  if (rounds) {
    j["total"]["rounds"] = rounds;
    j["total"]["tokens_per_verify"] = double(accepted + rounds) / double(rounds);
  }
  std::printf("%-28s %6ld tok  %7.1f tok/s", "TOTAL", tokens, seconds > 0 ? double(tokens) / seconds : 0.0);
  if (rounds) std::printf("  %5.2f tok/verify", double(accepted + rounds) / double(rounds));
  std::printf("\n");
  std::fflush(stdout);
  return j.dump();
}

}  // namespace

std::string run_forced(model::Generator& gen, const std::string& path, double temperature, int skip, int items,
                       const std::string& trace_path, int free_seed) {
  std::ifstream in(path);
  if (!in) throw std::runtime_error("cannot open " + path);
  std::vector<Item> list;
  int index = 0;
  for (std::string line; std::getline(in, line);) {
    if (line.empty() || index++ < skip) continue;
    const auto j = nlohmann::json::parse(line);
    list.push_back({j.at("name").get<std::string>(), j.at("prompt").get<std::vector<std::int32_t>>(),
                    j.at("output").get<std::vector<std::int32_t>>()});
    if (items > 0 && int(list.size()) == items) break;
  }
  if (free_seed >= 0) return run_free(gen, path, list, temperature, free_seed, trace_path);

  Totals total, kinds[NKINDS];
  std::vector<model::SpecRoundRecord> trace_all, trace_kind[NKINDS];
  struct Done {
    std::string name;
    Totals t;
    std::vector<model::SpecRoundRecord> tr;
  };
  std::vector<Done> done;
  for (const Item& it : list) {
    if (it.output.size() < 2) continue;
    model::Sampling sampling;
    sampling.temperature = float(temperature);
    sampling.seed = 0;
    sampling.forced = it.output;
    const model::GenerateStats st = gen.generate(it.prompt, int(it.output.size()), sampling, {},
                                                 [](std::span<const std::int32_t>) { return true; });
    Totals t;
    t.items = 1;
    t.tokens = st.generated_tokens - 1;  // the first token comes with the prefill
    t.seconds = st.decode_seconds;
    for (const auto& r : st.spec_trace) t.add_round(r);
    // per kind: a round belongs to the kind of its first draft's forced token
    const std::vector<int> k = kinds_of(it.output);
    long e = 1;  // tokens emitted before the round's verify (its anchor is output[e - 1])
    Totals per_kind[NKINDS];
    for (const auto& r : st.spec_trace) {
      const int kind = k[std::min<std::size_t>(std::size_t(e), k.size() - 1)];
      per_kind[kind].add_round(r);
      trace_kind[kind].push_back(r);
      e += r.accepted + 1;
    }
    // apportion decode time and tokens to the kinds by their rounds' GPU time share (rounds only)
    const double gpu = t.draft_ms + t.verify_ms;
    for (int kd = 0; kd < NKINDS; ++kd) {
      per_kind[kd].tokens = per_kind[kd].round_tokens;
      per_kind[kd].seconds = gpu > 0 ? t.seconds * (per_kind[kd].draft_ms + per_kind[kd].verify_ms) / gpu : 0;
      Totals& K = kinds[kd];
      K.items += per_kind[kd].rounds > 0;
      K.tokens += per_kind[kd].tokens;
      K.seconds += per_kind[kd].seconds;
      K.rounds += per_kind[kd].rounds;
      K.round_tokens += per_kind[kd].round_tokens;
      K.first_rejected += per_kind[kd].first_rejected;
      K.oracle_tokens += per_kind[kd].oracle_tokens;
      K.draft_ms += per_kind[kd].draft_ms;
      K.verify_ms += per_kind[kd].verify_ms;
      for (int w = 0; w < WIDTHS; ++w) {
        K.width_rounds[w] += per_kind[kd].width_rounds[w];
        K.width_tokens[w] += per_kind[kd].width_tokens[w];
        K.width_verify_ms[w] += per_kind[kd].width_verify_ms[w];
      }
    }
    total.items += 1;
    total.tokens += t.tokens;
    total.seconds += t.seconds;
    total.rounds += t.rounds;
    total.round_tokens += t.round_tokens;
    total.first_rejected += t.first_rejected;
    total.oracle_tokens += t.oracle_tokens;
    total.draft_ms += t.draft_ms;
    total.verify_ms += t.verify_ms;
    for (int w = 0; w < WIDTHS; ++w) {
      total.width_rounds[w] += t.width_rounds[w];
      total.width_tokens[w] += t.width_tokens[w];
      total.width_verify_ms[w] += t.width_verify_ms[w];
    }
    trace_all.insert(trace_all.end(), st.spec_trace.begin(), st.spec_trace.end());
    if (!trace_path.empty()) {
      nlohmann::ordered_json row;
      row["item"] = it.name;
      row["tokens"] = t.tokens;
      row["seconds"] = t.seconds;
      row["rounds"] = nlohmann::json::array();
      long e2 = 1;
      for (const auto& r : st.spec_trace) {
        row["rounds"].push_back({r.width, r.accepted, r.accepted_all, r.verify_ms, r.draft_ms,
                                 KIND_NAMES[k[std::min<std::size_t>(std::size_t(e2), k.size() - 1)]]});
        e2 += r.accepted + 1;
      }
      std::ofstream(trace_path, std::ios::app) << row.dump() << "\n";
    }
    done.push_back({it.name, t, st.spec_trace});
  }

  for (const Done& d : done) print_totals(d.name.c_str(), d.t, total, d.tr);
  for (int kd = 0; kd < NKINDS; ++kd)
    if (kinds[kd].tokens) print_totals((std::string("kind ") + KIND_NAMES[kd]).c_str(), kinds[kd], total, trace_kind[kd]);
  print_totals("TOTAL", total, total, trace_all);
  std::fflush(stdout);

  nlohmann::ordered_json j;
  j["file"] = path;
  j["items"] = total.items;
  j["total"] = to_json(total, total, trace_all);
  for (int kd = 0; kd < NKINDS; ++kd)
    if (kinds[kd].tokens) j["kinds"][KIND_NAMES[kd]] = to_json(kinds[kd], total, trace_kind[kd]);
  for (const Done& d : done) j["per_item"][d.name] = to_json(d.t, total, d.tr);
  return j.dump();
}

}  // namespace ingine::bench
