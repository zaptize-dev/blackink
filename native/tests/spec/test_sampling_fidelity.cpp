// Sampling fidelity at the full vocabulary (200058 tokens, temperature 1, no top-k / top-p: the served default):
// do the plain sampler and the speculative acceptance path draw the target distribution's tail exactly?
//
// Rows: golden decode steps (the Python reference's top-256 logits and log-sum-exp) completed with a synthetic
// tail that carries exactly the remaining mass, spread log-normally over the other 199802 tokens. Per row:
//   (a) head::sample: 200K draws, multi-block (8 rows per call) and one-CTA (9 rows per call) kernels;
//   (b) the speculative path for one draft: p = head::sampling_probs (multi-block, or one-CTA via a 9-row call),
//       a draft d ~ q with q a realistic draft distribution (softmax over 1024 candidates: the target's top
//       logits plus noise, -inf elsewhere, as DSpark's), draft::sampled_accept, and the residual draw
//       head::sample_from_probs when d is rejected; the emitted token is d if accepted, else the residual draw.
// For k in {10, 50, 256}: the number of draws outside the target's top k against the exact probability
// (binomial z-test, |z| < 4.5), and the ratio observed / expected.
#include "common/test_util.h"

#include "kernels/draft/draft.h"
#include "kernels/head/head.h"

#include <cuda_runtime.h>
#include <nlohmann/json.hpp>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <fstream>
#include <numeric>
#include <string>
#include <vector>

using namespace ingine::test;
namespace head = ingine::kernels::head;
namespace kd = ingine::kernels::draft;

namespace {

constexpr int V = 200058;
constexpr int kDraws = 200000;
constexpr int kCand = 1024;

template <class T>
struct Dev {
  T* p = nullptr;
  explicit Dev(std::size_t n) { INGINE_TEST_CUDA(cudaMalloc(&p, std::max<std::size_t>(n, 1) * sizeof(T))); }
  Dev(const std::vector<T>& h) : Dev(h.size()) {
    INGINE_TEST_CUDA(cudaMemcpy(p, h.data(), h.size() * sizeof(T), cudaMemcpyHostToDevice));
  }
  ~Dev() { cudaFree(p); }
  std::vector<T> get(std::size_t n) const {
    std::vector<T> h(n);
    INGINE_TEST_CUDA(cudaMemcpy(h.data(), p, n * sizeof(T), cudaMemcpyDeviceToHost));
    return h;
  }
};

struct Row {
  std::string name;
  std::vector<float> logits;  // [V]
  std::vector<double> p;      // exact softmax
  std::vector<int> order;     // token ids by descending p
};

Row golden_row(const std::string& file, int step, Rng& rng) {
  const auto j = nlohmann::json::parse(std::ifstream(std::string(INGINE_GOLDEN_DIR) + "/model/" + file));
  const auto& s = j.at("decode").at(step);
  const auto ids = s.at("top_ids").get<std::vector<int>>();
  const auto top = s.at("top_logits").get<std::vector<double>>();
  const double lse = s.at("logsumexp").get<double>();
  double top_mass = 0;
  for (double x : top) top_mass += std::exp(x - lse);
  const double rest = std::max(1.0 - top_mass, 1e-7);
  Row r;
  r.name = file + " step " + std::to_string(step);
  r.logits.assign(V, 0.f);
  std::vector<char> is_top(V, 0);
  for (std::size_t i = 0; i < ids.size(); ++i) r.logits[ids[i]] = float(top[i]), is_top[ids[i]] = 1;
  std::vector<double> w(V, 0.0);
  double sw = 0;
  for (int v = 0; v < V; ++v)
    if (!is_top[v]) sw += w[v] = std::exp(1.5 * rng.normal());
  for (int v = 0; v < V; ++v)
    if (!is_top[v]) r.logits[v] = float(lse + std::log(rest * w[v] / sw));
  double m = -1e300, z = 0;
  for (float x : r.logits) m = std::max(m, double(x));
  for (float x : r.logits) z += std::exp(double(x) - m);
  r.p.resize(V);
  for (int v = 0; v < V; ++v) r.p[v] = std::exp(double(r.logits[v]) - m) / z;
  r.order.resize(V);
  std::iota(r.order.begin(), r.order.end(), 0);
  std::sort(r.order.begin(), r.order.end(), [&](int a, int b) { return r.p[a] > r.p[b]; });
  return r;
}

// draws outside the top k against the exact tail mass, for k = 10, 50, 256
bool check_tail(const Row& r, const std::vector<int>& draws, const char* what) {
  std::vector<int> rank(V);
  for (int i = 0; i < V; ++i) rank[r.order[i]] = i;
  bool ok = true;
  std::string line;
  for (int k : {10, 50, 256}) {
    double tail = 0;
    for (int i = k; i < V; ++i) tail += r.p[r.order[i]];
    long n_out = 0;
    for (int d : draws) n_out += rank[d] >= k;
    const double n = double(draws.size()), e = n * tail;
    const double z = (n_out - e) / std::sqrt(std::max(e * (1 - tail), 1e-12));
    ok &= std::fabs(z) < 4.5;
    char buf[128];
    std::snprintf(buf, sizeof buf, "  top-%d: tail %.2e, %ld drawn vs %.1f expected (x%.3f, z %+.2f)", k, tail, n_out, e,
                  e > 0 ? n_out / e : 0.0, z);
    line += buf;
  }
  std::printf("%s %s, %s:%s\n", ok ? "PASS" : "FAIL", r.name.c_str(), what, line.c_str());
  return ok;
}

std::vector<int> plain_draws(const Row& r, int rows, std::uint64_t seed) {
  std::vector<float> rep(std::size_t(rows) * V);
  for (int i = 0; i < rows; ++i) std::copy(r.logits.begin(), r.logits.end(), rep.begin() + std::size_t(i) * V);
  Dev<float> dl(rep);
  const int calls = (kDraws + rows - 1) / rows;
  Dev<int> out(std::size_t(calls) * rows);
  head::SamplerRng rng{seed, 0};
  const head::SamplingParams plain{1.0f, 0, 1.0f};
  for (int c = 0; c < calls; ++c) head::sample(dl.p, rows, V, V, plain, rng, out.p + std::size_t(c) * rows, nullptr);
  INGINE_TEST_CUDA(cudaDeviceSynchronize());
  auto d = out.get(std::size_t(calls) * rows);
  d.resize(kDraws);
  return d;
}

// one speculative draft per round; one_cta: the probabilities and the residual draw through the one-CTA kernels
std::vector<int> spec_draws(const Row& r, bool one_cta, std::uint64_t seed, Rng& host_rng, double* accept_rate) {
  const int prow = one_cta ? 9 : 2;  // sampled_accept reads rows 0..n (n = 1); 9 rows force the one-CTA kernels
  std::vector<float> rep(std::size_t(prow) * V);
  for (int i = 0; i < prow; ++i) std::copy(r.logits.begin(), r.logits.end(), rep.begin() + std::size_t(i) * V);
  Dev<float> dl(rep), probs(std::size_t(prow) * V);
  const head::SamplingParams plain{1.0f, 0, 1.0f};
  head::sampling_probs(dl.p, prow, V, V, plain, probs.p, nullptr);
  // the draft distribution: the target's top kCand tokens with noisy logits (the drafter disagrees on order and
  // sharpness), -inf elsewhere; q in double for the host draws, qstats (max, sum of exp(l - max)) for the kernel
  std::vector<float> ql(V, -INFINITY);
  for (int i = 0; i < kCand; ++i) {
    const int v = r.order[i];
    ql[v] = float(0.8 * r.logits[v] + 0.7 * host_rng.normal());
  }
  float qm = -INFINITY;
  for (float x : ql) qm = std::max(qm, x);
  double qs = 0;
  for (float x : ql)
    if (x > -INFINITY) qs += std::exp(double(x) - qm);
  std::vector<double> qcum;
  std::vector<int> qid;
  double c = 0;
  for (int v = 0; v < V; ++v)
    if (ql[v] > -INFINITY) qcum.push_back(c += std::exp(double(ql[v]) - qm) / qs), qid.push_back(v);
  std::vector<int> drafts(kDraws);
  for (int& d : drafts) {
    const double u = host_rng.uniform() * c;
    d = qid[std::size_t(std::lower_bound(qcum.begin(), qcum.end(), u) - qcum.begin())];
  }
  const std::vector<float> qstats = {qm, float(qs)};
  Dev<float> dq(ql), dqs(qstats), u(kDraws), ws(kd::sampled_accept_workspace_floats(V)),
      resid(std::size_t(one_cta ? 9 : 1) * V);
  Dev<int> dd(drafts);
  Dev<kd::AcceptResult> res(kDraws);
  head::SamplerRng rng{seed, 0};
  for (int i = 0; i < kDraws; ++i) {
    head::uniform(1, rng, u.p + i, nullptr);
    kd::sampled_accept(probs.p, dq.p, V, dd.p + i, 1, 1.0f, dqs.p, u.p + i, ws.p, resid.p, res.p + i, nullptr);
    if (one_cta) {
      // 9 rows (the first is the residual, the others zero-padded copies of it) select the one-CTA kernel
      for (int k = 1; k < 9; ++k)
        INGINE_TEST_CUDA(cudaMemcpyAsync(resid.p + std::size_t(k) * V, resid.p, std::size_t(V) * 4,
                                         cudaMemcpyDeviceToDevice, nullptr));
      static Dev<int> nine(9);
      head::sample_from_probs(resid.p, 9, V, V, rng, nine.p, nullptr);
      INGINE_TEST_CUDA(cudaMemcpyAsync(&res.p[i].next, nine.p, 4, cudaMemcpyDeviceToDevice, nullptr));
    } else {
      head::sample_from_probs(resid.p, 1, V, V, rng, &res.p[i].next, nullptr);
    }
  }
  INGINE_TEST_CUDA(cudaDeviceSynchronize());
  const auto out = res.get(kDraws);
  std::vector<int> emitted(kDraws);
  long acc = 0;
  for (int i = 0; i < kDraws; ++i) {
    emitted[i] = out[i].accepted ? drafts[i] : out[i].next;
    acc += out[i].accepted;
  }
  *accept_rate = double(acc) / kDraws;
  return emitted;
}

}  // namespace

int main() {
  if (skip_without_gpu()) return kSkip;
  Rng rng(2026);
  bool ok = true;
  // a typical row (0.23% outside the top 256), and the two goldens with the most tail mass (5.5% and 12%)
  for (const auto& [file, step] : {std::pair{"abl_kv8_t2100.json", 1}, std::pair{"abl_kv8_t5000.json", 13},
                                    std::pair{"abl_kv8_t300.json", 24}}) {
    const Row r = golden_row(file, step, rng);
    ok &= check_tail(r, plain_draws(r, 8, 11), "sample (multi-block)");
    ok &= check_tail(r, plain_draws(r, 9, 12), "sample (one-CTA)");
    for (const bool one_cta : {false, true}) {
      double acc = 0;
      const auto d = spec_draws(r, one_cta, 13 + one_cta, rng, &acc);
      char what[96];
      std::snprintf(what, sizeof what, "speculative accept + residual (%s, acceptance %.2f)",
                    one_cta ? "one-CTA" : "multi-block", acc);
      ok &= check_tail(r, d, what);
    }
  }
  std::printf("%s\n", ok ? "ALL PASS" : "SOME FAILED");
  return ok ? 0 : 1;
}
