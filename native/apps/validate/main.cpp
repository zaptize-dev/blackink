// ingine-validate: parity of the C++ model against the Python engine's goldens (tools/goldens_model.py).
//
// Default ("pieces"): for every golden case, runs the C++ Prefiller over the prompt (chunked as the goldens
// were) and the width-1 decode graph for the golden greedy tokens (teacher forced), and reports per stage
// the KL divergence KL(python || c++) over the golden top-256 (+ the remaining mass as one bucket), top-1
// agreement and the largest logit difference among the golden top-256. --generate instead runs the full
// Generator greedily and compares its tokens with the golden greedy continuation; --generate --spec does the
// same with DSpark speculative decoding (greedy speculation must reproduce plain greedy decoding).
// Exit status 0 when the mean KL over all stages is within --max-mean-kl, every stage within --max-kl and top-1
// agreement is at least --min-top1, else 1. The defaults sit at the Python reference's own reproducibility: on
// the long golden prompts, Python against itself (prefill chunk 2048 vs 1536, or flex TF32 vs exact FP32
// attention) already differs by KL 6e-3 .. 8e-2 per stage with top-1 flips at near ties, while a real bug gives
// KL >= 1 on most stages. --generate needs every selected case's greedy tokens to match exactly, so it is meant
// for prompts without near ties (the CTest entries pick the BF16-cache chat and 300-token cases).
// --path N checks long generations, where exact goldens cannot exist: the Generator (with --spec: speculative)
// emits N greedy tokens per case; then a plain runtime prefills the same prompt and is teacher-forced along those
// tokens with the width-1 graph. Every emitted token must be the plain argmax, or lose to it by at most
// --tie-margin logits (speculative verification runs wider kernels whose rounding differs, so near ties may
// resolve differently); without --spec the generator runs the same width-1 kernels and must match exactly.
// With --max-gap-rate R the margin is a rate instead: over all selected cases, at most R tokens per 10,000 may
// lose to the plain argmax by more than --tie-margin (single positions of this model can move by several
// logits between kernel widths, so one long path is not evidence either way). Prints the gap distribution.
// N > 1024 covers the sliding-window ring wrap and the drafter's context buckets.
#include "ingine/generator.h"

#include "core/arena.h"
#include "core/device.h"
#include "model/prefill.h"
#include "model/runtime.h"

#include <nlohmann/json.hpp>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <map>
#include <string>
#include <vector>

namespace fs = std::filesystem;
using ingine::DeviceBuffer;
using namespace ingine::model;

namespace {

// $INGINE_MODEL_DIR, else the installer's download location (as ingine-bench)
fs::path default_model_dir() {
  if (const char* env = std::getenv("INGINE_MODEL_DIR"); env != nullptr && *env != '\0') return env;
  return fs::path(INGINE_REPO_DIR) / "models" / "Inkling-2-Mini";
}

struct Args {
  fs::path model_dir = default_model_dir();
  fs::path ablation = fs::path(INGINE_REPO_DIR) / "ablation" / "inkling-2-mini-abliterated.safetensors";
  fs::path golden = fs::path(INGINE_REPO_DIR) / "native" / "tests" / "golden" / "model";
  std::string filter;
  bool generate = false;
  bool spec = false;  // --generate / --path with DSpark speculative decoding
  int path_tokens = 0;        // --path N
  double tie_margin = 0.5;    // --tie-margin X
  double max_gap_rate = -1;   // --max-gap-rate R: tokens per 10,000 above the margin (off: per-case maximum)
  double max_kl = 0.25;
  double max_mean_kl = 0.01;
  double min_top1 = 0.95;
};

void usage() {
  std::printf(
      "usage: ingine-validate [--model-dir DIR] [--ablation FILE] [--golden DIR] [--cases SUBSTR] [--generate [--spec]]\n"
      "                       [--path N [--spec] [--tie-margin X] [--max-gap-rate R]] [--max-kl X] [--max-mean-kl X]\n"
      "                       [--min-top1 F]\n"
      "--cases takes comma-separated name substrings\n");
}

struct Stage {
  double kl = 0, max_diff = 0;
  bool top1 = false;
};

// KL(p || q) with p from the golden top-256 + logsumexp and q from the full C++ row; the mass outside the
// golden top-256 is compared as one bucket.
Stage compare(const nlohmann::json& g, const std::vector<float>& c) {
  double mx = -INFINITY;
  int arg = 0;
  for (std::size_t i = 0; i < c.size(); ++i)
    if (c[i] > mx) mx = c[i], arg = int(i);
  double s = 0;
  for (float v : c) s += std::exp(double(v) - mx);
  const double lse_c = mx + std::log(s);
  const double lse_g = g.at("logsumexp").get<double>();
  const auto& ids = g.at("top_ids");
  const auto& vals = g.at("top_logits");
  Stage st;
  double pin = 0, qin = 0;
  for (std::size_t k = 0; k < ids.size(); ++k) {
    const int id = ids[k].get<int>();
    const double lp = vals[k].get<double>() - lse_g, lq = double(c[std::size_t(id)]) - lse_c;
    const double p = std::exp(lp);
    st.kl += p * (lp - lq);
    pin += p;
    qin += std::exp(lq);
    st.max_diff = std::max(st.max_diff, std::fabs(vals[k].get<double>() - double(c[std::size_t(id)])));
  }
  const double pr = std::max(0.0, 1.0 - pin), qr = std::max(1e-30, 1.0 - qin);
  if (pr > 0) st.kl += pr * std::log(pr / qr);
  st.top1 = arg == ids[0].get<int>();
  return st;
}

nlohmann::json load_json(const fs::path& p) {
  std::ifstream in(p);
  if (!in) throw std::runtime_error("cannot open " + p.string());
  return nlohmann::json::parse(in);
}

struct Totals {
  int stages = 0, top1 = 0;
  double max_kl = 0, sum_kl = 0;
  void add(const Stage& s) {
    ++stages;
    top1 += s.top1;
    sum_kl += s.kl;
    max_kl = std::max(max_kl, s.kl);
  }
};

void run_pieces(const RuntimeOptions& opt, const std::vector<nlohmann::json>& cases, Totals& tot) {
  Runtime rt(opt);
  Prefiller prefiller(rt.weights(), rt.ablation(), opt.prefill_chunk, rt.stream());
  DeviceBuffer tokens(std::size_t(opt.prefill_chunk) * 4), logits(std::size_t(UNPADDED_VOCAB) * 4);
  std::vector<float> host(UNPADDED_VOCAB);
  auto fetch = [&](const float* d) {
    CUDA_CHECK(cudaMemcpyAsync(host.data(), d, host.size() * 4, cudaMemcpyDeviceToHost, rt.stream()));
    CUDA_CHECK(cudaStreamSynchronize(rt.stream()));
  };
  for (const auto& c : cases) {
    const std::string name = c.at("name");
    const std::vector<std::int32_t> ids = c.at("prompt_ids").get<std::vector<std::int32_t>>();
    reset_state(rt.state(), rt.stream());
    for (int s = 0; s < int(ids.size()); s += opt.prefill_chunk) {
      const int T = std::min(opt.prefill_chunk, int(ids.size()) - s);
      CUDA_CHECK(cudaMemcpyAsync(tokens.p, ids.data() + s, std::size_t(T) * 4, cudaMemcpyHostToDevice, rt.stream()));
      prefiller.run(rt.state(), static_cast<const std::int32_t*>(tokens.p), T, s, static_cast<float*>(logits.p),
                    nullptr);
    }
    fetch(static_cast<const float*>(logits.p));
    const Stage pre = compare(c.at("prefill"), host);
    tot.add(pre);
    std::printf("%-22s prefill (%5zu tokens): KL %.3e  top1 %s  max|dlogit| %.4f\n", name.c_str(), ids.size(), pre.kl,
                pre.top1 ? "ok" : "MISS", pre.max_diff);
    double sum_kl = 0, max_kl = 0, max_diff = 0;
    int agree = 0, steps = 0;
    for (const auto& step : c.at("decode")) {
      const float* d = rt.decode_logits(step.at("input_token").get<std::int32_t>(), step.at("pos").get<int>());
      fetch(d);
      const Stage s = compare(step, host);
      tot.add(s);
      sum_kl += s.kl;
      max_kl = std::max(max_kl, s.kl);
      max_diff = std::max(max_diff, s.max_diff);
      agree += s.top1;
      ++steps;
    }
    std::printf("%-22s decode  (%2d steps):     KL mean %.3e max %.3e  top1 %d/%d  max|dlogit| %.4f\n", name.c_str(),
                steps, steps ? sum_kl / steps : 0.0, max_kl, agree, steps, max_diff);
  }
}

// A fresh Generator per case: the golden prompts share prefixes (t300 starts t2100 starts t5000), and a Generator
// that reuses the previous request's prompt checkpoint prefills the rest with other chunk boundaries than the
// goldens and the plain reference runtime, which is a different (equally valid) rounding of the same state.
void run_generate(const RuntimeOptions& opt, const std::vector<nlohmann::json>& cases, Totals& tot) {
  for (const auto& c : cases) {
    auto gen = make_generator(opt);
    const std::string name = c.at("name");
    const std::vector<std::int32_t> ids = c.at("prompt_ids").get<std::vector<std::int32_t>>();
    const std::vector<std::int32_t> want = c.at("greedy_tokens").get<std::vector<std::int32_t>>();
    std::vector<std::int32_t> got;
    Sampling greedy;
    greedy.temperature = 0.f;
    const GenerateStats st = gen->generate(ids, int(want.size()), greedy, {}, [&](std::span<const std::int32_t> t) {
      got.insert(got.end(), t.begin(), t.end());
      return true;
    });
    std::size_t match = 0;
    while (match < want.size() && match < got.size() && want[match] == got[match]) ++match;
    Stage s;
    s.top1 = match == want.size();
    tot.add(s);
    std::printf("%-22s generate: %zu/%zu greedy tokens match  (prefill %.3f s, decode %.3f s, %d tokens)\n",
                name.c_str(), match, want.size(), st.prefill_seconds, st.decode_seconds, st.generated_tokens);
  }
}

// gap distribution of --path over all cases: emitted tokens that are not the plain argmax, by logit gap
constexpr double kGapEdges[5] = {0.1, 0.25, 0.5, 1.0, 2.0};
struct Gaps {
  std::size_t tokens = 0, other = 0, over[5] = {}, over_margin = 0;
  double max = 0;
};

// --path: see the file header. One stage per case; `top1` is the case's verdict.
void run_path(RuntimeOptions opt, const std::vector<nlohmann::json>& cases, const Args& a, Totals& tot, Gaps& gaps) {
  const int N = a.path_tokens;
  std::size_t longest = 0;
  for (const auto& c : cases) longest = std::max(longest, c.at("prompt_ids").size());
  opt.max_ctx = std::max(opt.max_ctx, int((longest + N + 64 + 4095) / 4096 * 4096));
  std::vector<std::vector<std::int32_t>> emitted(cases.size());
  for (std::size_t i = 0; i < cases.size(); ++i) {
    RuntimeOptions g = opt;
    g.spec = a.spec;
    auto gen = make_generator(g);  // per case (see run_generate); freed before the plain runtime loads
    Sampling greedy;
    greedy.temperature = 0.f;
    const std::vector<std::int32_t> ids = cases[i].at("prompt_ids").get<std::vector<std::int32_t>>();
    const GenerateStats st = gen->generate(ids, N, greedy, {}, [&](std::span<const std::int32_t> t) {
      emitted[i].insert(emitted[i].end(), t.begin(), t.end());
      return true;
    });
    std::printf("%-22s generated %zu tokens at %.1f tok/s%s\n", cases[i].at("name").get<std::string>().c_str(),
                emitted[i].size(), st.decode_seconds > 0 ? st.generated_tokens / st.decode_seconds : 0.0,
                a.spec ? (", " + std::to_string(st.spec_rounds) + " rounds").c_str() : "");
  }
  opt.spec = false;
  Runtime rt(opt);
  Prefiller prefiller(rt.weights(), rt.ablation(), opt.prefill_chunk, rt.stream());
  DeviceBuffer tokens(std::size_t(opt.prefill_chunk) * 4), logits(std::size_t(UNPADDED_VOCAB) * 4);
  std::vector<float> host(UNPADDED_VOCAB);
  auto fetch = [&](const float* d) {
    CUDA_CHECK(cudaMemcpyAsync(host.data(), d, host.size() * 4, cudaMemcpyDeviceToHost, rt.stream()));
    CUDA_CHECK(cudaStreamSynchronize(rt.stream()));
  };
  for (std::size_t i = 0; i < cases.size(); ++i) {
    const std::string name = cases[i].at("name");
    const std::vector<std::int32_t> ids = cases[i].at("prompt_ids").get<std::vector<std::int32_t>>();
    const std::vector<std::int32_t>& out = emitted[i];
    reset_state(rt.state(), rt.stream());
    for (int s = 0; s < int(ids.size()); s += opt.prefill_chunk) {
      const int T = std::min(opt.prefill_chunk, int(ids.size()) - s);
      CUDA_CHECK(cudaMemcpyAsync(tokens.p, ids.data() + s, std::size_t(T) * 4, cudaMemcpyHostToDevice, rt.stream()));
      prefiller.run(rt.state(), static_cast<const std::int32_t*>(tokens.p), T, s, static_cast<float*>(logits.p),
                    nullptr);
    }
    int other = 0, first = -1;
    double worst = 0;
    for (std::size_t k = 0; k < out.size(); ++k) {
      // logits that predict out[k]: the prefill's last row, then the step that consumed out[k - 1]
      fetch(k == 0 ? static_cast<const float*>(logits.p) : rt.decode_logits(out[k - 1], int(ids.size() + k - 1)));
      const int best = int(std::max_element(host.begin(), host.end()) - host.begin());
      if (best == out[k]) continue;
      ++other;
      if (first < 0) first = int(k);
      const double gap = double(host[best]) - double(host[out[k]]);
      worst = std::max(worst, gap);
      for (int e = 0; e < 5; ++e) gaps.over[e] += gap > kGapEdges[e];
      gaps.over_margin += gap > a.tie_margin;
    }
    gaps.tokens += out.size();
    gaps.other += std::size_t(other);
    gaps.max = std::max(gaps.max, worst);
    Stage s;
    const bool margin_ok = a.max_gap_rate >= 0 || worst <= (a.spec ? a.tie_margin : 0.0);
    s.top1 = int(out.size()) == N && margin_ok && other <= (a.spec ? N / 20 : 0);
    s.max_diff = worst;
    tot.add(s);
    std::printf("%-22s path: %zu tokens, %d not the plain argmax (first at %d), largest gap %.4f logits  %s\n",
                name.c_str(), out.size(), other, first, worst, s.top1 ? "ok" : "FAIL");
  }
}

}  // namespace

int main(int argc, char** argv) {
  Args a;
  for (int i = 1; i < argc; ++i) {
    const std::string k = argv[i];
    auto val = [&]() -> std::string {
      if (i + 1 >= argc) {
        usage();
        std::exit(2);
      }
      return argv[++i];
    };
    if (k == "--model-dir") a.model_dir = val();
    else if (k == "--ablation") a.ablation = val();
    else if (k == "--golden") a.golden = val();
    else if (k == "--cases") a.filter = val();
    else if (k == "--generate") a.generate = true;
    else if (k == "--spec") a.spec = true;
    else if (k == "--path") a.path_tokens = std::stoi(val());
    else if (k == "--tie-margin") a.tie_margin = std::stod(val());
    else if (k == "--max-gap-rate") a.max_gap_rate = std::stod(val());
    else if (k == "--max-kl") a.max_kl = std::stod(val());
    else if (k == "--max-mean-kl") a.max_mean_kl = std::stod(val());
    else if (k == "--min-top1") a.min_top1 = std::stod(val());
    else {
      usage();
      return k == "--help" || k == "-h" ? 0 : 2;
    }
  }
  try {
    const nlohmann::json manifest = load_json(a.golden / "manifest.json");
    // one runtime per (ablated, kv_fp8) configuration, loaded one at a time
    std::map<std::pair<bool, bool>, std::vector<nlohmann::json>> groups;
    for (const auto& c : manifest.at("cases")) {
      const std::string name = c.at("name");
      if (!a.filter.empty()) {
        bool hit = false;
        for (std::size_t b = 0; b <= a.filter.size();) {
          const std::size_t e = std::min(a.filter.find(',', b), a.filter.size());
          hit |= e > b && name.find(a.filter.substr(b, e - b)) != std::string::npos;
          b = e + 1;
        }
        if (!hit) continue;
      }
      groups[{c.at("ablated").get<bool>(), c.at("kv_fp8").get<bool>()}].push_back(
          load_json(a.golden / c.at("file").get<std::string>()));
    }
    if (groups.empty()) {
      std::printf("no golden cases selected\n");
      return 1;
    }
    Totals tot;
    Gaps gaps;
    for (const auto& [key, cases] : groups) {
      RuntimeOptions opt;
      opt.model_dir = a.model_dir;
      opt.ablation = key.first ? a.ablation : fs::path();
      opt.kv_fp8 = key.second;
      opt.dense_fp8 = manifest.value("dense_fp8", true);
      opt.max_ctx = manifest.value("max_ctx", 8192);
      opt.prefill_chunk = manifest.value("prefill_chunk", 2048);
      opt.spec = a.generate && a.spec;
      std::printf("== %s model, %s full-attention KV\n", key.first ? "abliterated" : "original",
                  key.second ? "FP8" : "BF16");
      if (a.path_tokens > 0) run_path(opt, cases, a, tot, gaps);
      else if (a.generate) run_generate(opt, cases, tot);
      else run_pieces(opt, cases, tot);
    }
    const double top1 = tot.stages ? double(tot.top1) / tot.stages : 0.0;
    const double mean_kl = tot.stages ? tot.sum_kl / tot.stages : 0.0;
    const bool no_kl = a.generate || a.path_tokens > 0;
    bool ok = (no_kl || (tot.max_kl <= a.max_kl && mean_kl <= a.max_mean_kl)) &&
              top1 >= (a.path_tokens > 0 ? 1.0 : a.min_top1);
    if (a.path_tokens > 0) {
      const double rate = gaps.tokens ? 1e4 * double(gaps.over_margin) / double(gaps.tokens) : 0.0;
      std::printf("gaps: %zu tokens, %zu not the plain argmax; above 0.1 / 0.25 / 0.5 / 1 / 2 logits: %zu / %zu / %zu / "
                  "%zu / %zu; max %.4f; above the %.2f margin: %.2f per 10,000",
                  gaps.tokens, gaps.other, gaps.over[0], gaps.over[1], gaps.over[2], gaps.over[3], gaps.over[4],
                  gaps.max, a.tie_margin, rate);
      if (a.max_gap_rate >= 0) {
        ok &= rate <= a.max_gap_rate;
        std::printf(" (limit %.2f)", a.max_gap_rate);
      }
      std::printf("\n");
    }
    std::printf("%s: %d stages, top1 %.3f, KL mean %.3e max %.3e (limits: KL mean %.3e max %.3e, top1 %.3f)\n",
                ok ? "PASS" : "FAIL", tot.stages, top1, mean_kl, tot.max_kl, a.max_mean_kl, a.max_kl, a.min_top1);
    return ok ? 0 : 1;
  } catch (const std::exception& e) {
    std::fprintf(stderr, "ingine-validate: %s\n", e.what());
    return 1;
  }
}
