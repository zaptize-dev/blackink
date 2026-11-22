// Token selection kernels (src/kernels/head) against host oracles of engine.Engine._probs / _sample:
// argmax (first maximum on ties), the sampling distribution over the full 200058-token vocabulary for
// temperature / top-k / top-p combinations, top-k = 1 and temperature 0 draws, seeded determinism, the
// empirical distribution of many draws, sample_from_probs and uniform.
#include "kernels/oracle.h"

#include "kernels/head/head.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <numeric>

using namespace ingine::test;
namespace head = ingine::kernels::head;

namespace {

constexpr int V = 200058;
cudaStream_t g_stream = nullptr;

void sync() { INGINE_TEST_CUDA(cudaStreamSynchronize(g_stream)); }

// engine._probs in double (tokens tied with the last top-p token kept, as the kernel does)
std::vector<double> probs_oracle(const std::vector<float>& logits, int n, const head::SamplingParams& p) {
  const float inv_t = 1.0f / p.temperature;
  std::vector<float> z(n);
  for (int i = 0; i < n; ++i) z[i] = logits[i] * inv_t;
  float kth = -INFINITY;
  if (p.top_k > 0 && p.top_k < n) {
    std::vector<float> s(z);
    std::nth_element(s.begin(), s.begin() + (p.top_k - 1), s.end(), std::greater<float>());
    kth = s[p.top_k - 1];
  }
  double mx = -INFINITY;
  for (float v : z) mx = std::max(mx, double(v));
  std::vector<double> pr(n, 0.0);
  double sum = 0;
  for (int i = 0; i < n; ++i)
    if (z[i] >= kth) sum += (pr[i] = std::exp(double(z[i]) - mx));
  for (auto& v : pr) v /= sum;
  if (p.top_p < 1.f) {
    std::vector<int> order(n);
    std::iota(order.begin(), order.end(), 0);
    std::stable_sort(order.begin(), order.end(), [&](int a, int b) { return z[a] > z[b]; });
    // drop a token when the mass before it exceeds top_p, unless it ties with the last kept token
    double cum = 0;
    float last = NAN;
    for (int i : order) {
      if (cum > p.top_p && z[i] != last) {
        pr[i] = 0;
        continue;
      }
      cum += pr[i];
      last = z[i];
    }
    double kept = 0;
    for (double v : pr) kept += v;
    for (auto& v : pr) v /= kept;
  }
  return pr;
}

bool test_argmax(Rng& rng) {
  const int rows = 3;
  std::vector<float> l = randn(rng, size_t(rows) * V, 2.0);
  l[size_t(0) * V + 777] = 50.f, l[size_t(0) * V + 123] = 50.f;  // tie: the lower id wins
  l[size_t(1) * V + V - 1] = 60.f;
  DeviceArray<float> dl(l);
  DeviceArray<int> out(rows);
  head::argmax(dl.get(), rows, V, V, out.get(), g_stream);
  sync();
  const auto got = out.download();
  int want2 = int(std::max_element(l.begin() + 2 * V, l.end()) - (l.begin() + 2 * V));
  const bool ok = got[0] == 123 && got[1] == V - 1 && got[2] == want2;
  std::printf("%s argmax: %d %d %d\n", ok ? "PASS" : "FAIL", got[0], got[1], got[2]);
  return ok;
}

bool test_probs(Rng& rng, const head::SamplingParams& p) {
  std::vector<float> l = randn(rng, V, 3.0);
  for (int i = 0; i < 40; ++i) l[rng.next() % V] = float(12.0 + rng.normal());  // a peaked head
  DeviceArray<float> dl(l), out(V);
  head::sampling_probs(dl.get(), 1, V, V, p, out.get(), g_stream);
  sync();
  const auto got = out.download();
  const auto ref = probs_oracle(l, V, p);
  double max_err = 0, sum = 0;
  size_t support_diff = 0;
  for (int i = 0; i < V; ++i) {
    max_err = std::max(max_err, std::fabs(got[i] - ref[i]));
    sum += got[i];
    support_diff += (got[i] > 0) != (ref[i] > 0);
  }
  const bool ok = max_err < 1e-5 && std::fabs(sum - 1.0) < 1e-4 && support_diff == 0;
  std::printf("%s sampling_probs T=%.2f top_k=%d top_p=%.2f: max err %.2e sum %.6f support diffs %zu\n",
              ok ? "PASS" : "FAIL", p.temperature, p.top_k, p.top_p, max_err, sum, support_diff);
  return ok;
}

bool test_sample_special(Rng& rng) {
  std::vector<float> l = randn(rng, size_t(2) * V, 2.0);
  DeviceArray<float> dl(l);
  DeviceArray<int> a(2), b(2), c(2), d(2);
  head::SamplerRng r1{7, 0}, r2{7, 0};
  head::argmax(dl.get(), 2, V, V, a.get(), g_stream);
  head::sample(dl.get(), 2, V, V, {0.f, 0, 1.f}, r1, b.get(), g_stream);   // greedy
  head::sample(dl.get(), 2, V, V, {1.f, 1, 1.f}, r1, c.get(), g_stream);   // top-k 1
  head::sample(dl.get(), 2, V, V, {0.8f, 40, 0.9f}, r2, d.get(), g_stream);
  DeviceArray<int> e(2);
  head::SamplerRng r3{7, 0};
  head::sample(dl.get(), 2, V, V, {0.8f, 40, 0.9f}, r3, e.get(), g_stream);  // same seed and counter
  sync();
  const auto A = a.download(), B = b.download(), Cc = c.download(), Dd = d.download(), Ee = e.download();
  const bool ok = A == B && A == Cc && Dd == Ee && r1.counter == 1 && r2.counter == 1;
  std::printf("%s sample: greedy/top-1 = argmax, seeded draws reproducible\n", ok ? "PASS" : "FAIL");
  return ok;
}

bool test_sample_distribution(Rng& rng) {
  const int n = 64, draws = 20000;
  std::vector<float> l = randn(rng, n, 1.0);
  const head::SamplingParams p{0.9f, 20, 0.8f};
  const auto ref = probs_oracle(l, n, p);
  DeviceArray<float> dl(l), dp(n);
  DeviceArray<int> out(draws);
  head::SamplerRng r{12345, 0};
  for (int i = 0; i < draws; ++i) head::sample(dl.get(), 1, n, n, p, r, out.get() + i, g_stream);
  head::sampling_probs(dl.get(), 1, n, n, p, dp.get(), g_stream);
  DeviceArray<int> out2(draws);
  for (int i = 0; i < draws; ++i) head::sample_from_probs(dp.get(), 1, n, n, r, out2.get() + i, g_stream);
  sync();
  bool ok = true;
  for (const auto* o : {&out, &out2}) {
    const auto got = o->download();
    std::vector<double> freq(n, 0.0);
    for (int t : got) freq[t] += 1.0 / draws;
    double worst = 0;
    bool outside = false;
    for (int i = 0; i < n; ++i) {
      worst = std::max(worst, std::fabs(freq[i] - ref[i]) / std::sqrt(ref[i] * (1 - ref[i]) / draws + 1e-12));
      outside |= ref[i] == 0 && freq[i] > 0;
    }
    const bool pass = worst < 5.0 && !outside;
    std::printf("%s %s distribution over %d draws: worst |freq - p| = %.2f sigma%s\n", pass ? "PASS" : "FAIL",
                o == &out ? "sample" : "sample_from_probs", draws, worst, outside ? ", token outside support" : "");
    ok &= pass;
  }
  return ok;
}

bool test_uniform() {
  const int n = 100000;
  DeviceArray<float> out(n);
  head::SamplerRng r{99, 5};
  head::uniform(n, r, out.get(), g_stream);
  sync();
  const auto u = out.download();
  double mean = 0;
  bool range = true;
  for (float v : u) mean += v / n, range &= v >= 0.f && v < 1.f;
  const bool ok = range && std::fabs(mean - 0.5) < 0.005 && r.counter == 6;
  std::printf("%s uniform: mean %.4f in-range %d\n", ok ? "PASS" : "FAIL", mean, int(range));
  return ok;
}

// The multi-block paths (rows <= 8; sampling without top-k / top-p) against the one-CTA kernels, which a
// 9-row call still uses: argmax, Gumbel draws from logits and from probabilities are identical token for
// token (same seed and counter: the noise is keyed by row and index), sampling_probs agrees up to the softmax
// summation order. Plus the time per call of each multi-block path at the real vocabulary.
bool test_fast_paths(Rng& rng) {
  const int rows = 9;
  std::vector<float> l = randn(rng, size_t(rows) * V, 3.0);
  for (int r = 0; r < rows; ++r)
    for (int i = 0; i < 30; ++i) l[size_t(r) * V + rng.next() % V] = float(10.0 + rng.normal());
  l[size_t(4) * V + 999] = 40.f, l[size_t(4) * V + 77] = 40.f;  // argmax tie in a fast row: lower id
  DeviceArray<float> dl(l), p9(size_t(rows) * V), p8(size_t(8) * V);
  DeviceArray<int> a9(rows), a8(8), s9(rows), s8(8), m9(rows), m8(8);
  const head::SamplingParams plain{1.0f, 0, 1.0f};
  head::argmax(dl.get(), rows, V, V, a9.get(), g_stream);
  head::argmax(dl.get(), 8, V, V, a8.get(), g_stream);
  head::SamplerRng r1{42, 7}, r2{42, 7};
  head::sample(dl.get(), rows, V, V, plain, r1, s9.get(), g_stream);
  head::sample(dl.get(), 8, V, V, plain, r2, s8.get(), g_stream);
  head::sampling_probs(dl.get(), rows, V, V, plain, p9.get(), g_stream);
  head::sampling_probs(dl.get(), 8, V, V, plain, p8.get(), g_stream);
  head::SamplerRng r3{43, 1}, r4{43, 1};
  head::sample_from_probs(p9.get(), rows, V, V, r3, m9.get(), g_stream);
  head::sample_from_probs(p9.get(), 8, V, V, r4, m8.get(), g_stream);
  sync();
  const auto A9 = a9.download(), A8 = a8.download(), S9 = s9.download(), S8 = s8.download();
  const auto M9 = m9.download(), M8 = m8.download();
  const auto P9 = p9.download(), P8 = p8.download();
  bool same_a = A8[4] == 77, same_s = true, same_m = true;
  for (int r = 0; r < 8; ++r) same_a &= A8[r] == A9[r], same_s &= S8[r] == S9[r], same_m &= M8[r] == M9[r];
  double rel = 0;
  for (size_t i = 0; i < P8.size(); ++i) rel = std::max(rel, std::fabs(double(P8[i]) - P9[i]) / std::max(double(P9[i]), 1e-30));
  bool ok = same_a && same_s && same_m && rel < 1e-5;
  std::printf("%s multi-block argmax/sample/sample_from_probs equal the one-CTA kernels (%d %d %d), probs rel diff %.1e\n",
              ok ? "PASS" : "FAIL", int(same_a), int(same_s), int(same_m), rel);

  // time per call, one row (plain decoding) and 8 rows (a verify call's probabilities)
  cudaEvent_t e0, e1;
  INGINE_TEST_CUDA(cudaEventCreate(&e0));
  INGINE_TEST_CUDA(cudaEventCreate(&e1));
  auto time_us = [&](auto&& call) {
    for (int i = 0; i < 10; ++i) call();
    INGINE_TEST_CUDA(cudaEventRecord(e0, g_stream));
    for (int i = 0; i < 200; ++i) call();
    INGINE_TEST_CUDA(cudaEventRecord(e1, g_stream));
    INGINE_TEST_CUDA(cudaEventSynchronize(e1));
    float ms = 0;
    INGINE_TEST_CUDA(cudaEventElapsedTime(&ms, e0, e1));
    return 1000.0 * ms / 200;
  };
  head::SamplerRng rt{1, 0};
  const double t_arg = time_us([&] { head::argmax(dl.get(), 1, V, V, a8.get(), g_stream); });
  const double t_s1 = time_us([&] { head::sample(dl.get(), 1, V, V, plain, rt, s8.get(), g_stream); });
  const double t_p8 = time_us([&] { head::sampling_probs(dl.get(), 8, V, V, plain, p8.get(), g_stream); });
  const double t_m1 = time_us([&] { head::sample_from_probs(p8.get(), 1, V, V, rt, m8.get(), g_stream); });
  const double t_s1k = time_us([&] { head::sample(dl.get(), 1, V, V, {1.0f, 50, 0.95f}, rt, s8.get(), g_stream); });
  INGINE_TEST_CUDA(cudaEventDestroy(e0));
  INGINE_TEST_CUDA(cudaEventDestroy(e1));
  std::printf("INFO us per call: argmax 1 row %.1f, sample T=1 1 row %.1f, sampling_probs 8 rows %.1f, "
              "sample_from_probs 1 row %.1f, sample top-k/top-p (one-CTA path) %.1f\n",
              t_arg, t_s1, t_p8, t_m1, t_s1k);
  return ok;
}

// Draws of the multi-block paths over the full vocabulary follow engine._probs (T = 1, no top-k / top-p).
// top-k / top-p through the multi-block passes (rows <= 8) against the one-CTA kernel (9 rows): the same
// draws token for token (same seed and counter) and the same probabilities up to the softmax sum's rounding;
// plus the time per call.
bool test_fast_truncation(Rng& rng) {
  const int rows = 9;
  std::vector<float> l = randn(rng, size_t(rows) * V, 3.0);
  for (int r = 0; r < rows; ++r)
    for (int i = 0; i < 300; ++i) l[size_t(r) * V + rng.next() % V] = float(8.0 + 2.0 * rng.normal());
  DeviceArray<float> dl(l), p9(size_t(rows) * V), p8(size_t(8) * V);
  DeviceArray<int> s9(rows), s8(8);
  bool ok = true;
  for (const head::SamplingParams p : {head::SamplingParams{0.7f, 50, 1.0f}, head::SamplingParams{1.0f, 0, 0.9f},
                                       head::SamplingParams{1.3f, 1000, 0.95f}, head::SamplingParams{1.0f, 40, 0.8f},
                                       head::SamplingParams{1.0f, 1, 1.0f}}) {
    int same = 0;
    for (int c = 0; c < 20; ++c) {
      head::SamplerRng r1{5, std::uint64_t(c)}, r2{5, std::uint64_t(c)};
      head::sample(dl.get(), rows, V, V, p, r1, s9.get(), g_stream);
      head::sample(dl.get(), 8, V, V, p, r2, s8.get(), g_stream);
      sync();
      const auto A = s9.download(), B = s8.download();
      for (int r = 0; r < 8; ++r) same += A[r] == B[r];
    }
    head::sampling_probs(dl.get(), rows, V, V, p, p9.get(), g_stream);
    head::sampling_probs(dl.get(), 8, V, V, p, p8.get(), g_stream);
    sync();
    const auto P9 = p9.download(), P8 = p8.download();
    double rel = 0;
    size_t support = 0;
    for (size_t i = 0; i < P8.size(); ++i) {
      rel = std::max(rel, std::fabs(double(P8[i]) - P9[i]) / std::max(double(P9[i]), 1e-30));
      support += (P8[i] > 0) != (P9[i] > 0);
    }
    const bool pass = same == 160 && rel < 1e-5 && support == 0;
    std::printf("%s multi-block top-k/top-p T=%.1f k=%d p=%.2f: %d/160 draws equal the one-CTA kernel, probs rel diff "
                "%.1e, support diffs %zu\n",
                pass ? "PASS" : "FAIL", p.temperature, p.top_k, p.top_p, same, rel, support);
    ok &= pass;
  }
  cudaEvent_t e0, e1;
  INGINE_TEST_CUDA(cudaEventCreate(&e0));
  INGINE_TEST_CUDA(cudaEventCreate(&e1));
  auto time_us = [&](auto&& call) {
    for (int i = 0; i < 10; ++i) call();
    INGINE_TEST_CUDA(cudaEventRecord(e0, g_stream));
    for (int i = 0; i < 100; ++i) call();
    INGINE_TEST_CUDA(cudaEventRecord(e1, g_stream));
    INGINE_TEST_CUDA(cudaEventSynchronize(e1));
    float ms = 0;
    INGINE_TEST_CUDA(cudaEventElapsedTime(&ms, e0, e1));
    return 1000.0 * ms / 100;
  };
  head::SamplerRng rt{1, 0};
  const head::SamplingParams topp{1.0f, 0, 0.95f}, both{1.0f, 50, 0.95f};
  const double t1 = time_us([&] { head::sample(dl.get(), 1, V, V, topp, rt, s8.get(), g_stream); });
  const double t2 = time_us([&] { head::sample(dl.get(), 1, V, V, both, rt, s8.get(), g_stream); });
  const double t3 = time_us([&] { head::sampling_probs(dl.get(), 8, V, V, topp, p8.get(), g_stream); });
  INGINE_TEST_CUDA(cudaEventDestroy(e0));
  INGINE_TEST_CUDA(cudaEventDestroy(e1));
  std::printf("INFO us per call (multi-block truncation): sample top-p 1 row %.1f, top-k + top-p 1 row %.1f, "
              "sampling_probs top-p 8 rows %.1f\n",
              t1, t2, t3);
  return ok;
}

bool test_fast_distribution(Rng& rng) {
  std::vector<float> l = randn(rng, V, 1.0);
  const int heads[6] = {11, 50000, 100001, 150000, 199000, 200057};
  for (int i = 0; i < 6; ++i) l[heads[i]] = float(10.0 + 0.4 * i);  // most of the mass on six tokens
  const head::SamplingParams plain{1.0f, 0, 1.0f};
  const auto ref = probs_oracle(l, V, plain);
  const int draws = 20000;
  DeviceArray<float> dl(l), dp(V);
  DeviceArray<int> out(draws), out2(draws);
  head::SamplerRng r{777, 0};
  for (int i = 0; i < draws; ++i) head::sample(dl.get(), 1, V, V, plain, r, out.get() + i, g_stream);
  head::sampling_probs(dl.get(), 1, V, V, plain, dp.get(), g_stream);
  for (int i = 0; i < draws; ++i) head::sample_from_probs(dp.get(), 1, V, V, r, out2.get() + i, g_stream);
  sync();
  bool ok = true;
  for (const auto* o : {&out, &out2}) {
    const auto got = o->download();
    double worst = 0, rest_ref = 1, rest = 0;
    for (int h : heads) {
      const double f = double(std::count(got.begin(), got.end(), h)) / draws;
      worst = std::max(worst, std::fabs(f - ref[h]) / std::sqrt(ref[h] * (1 - ref[h]) / draws));
      rest_ref -= ref[h];
      rest += f;
    }
    rest = 1 - rest;
    worst = std::max(worst, std::fabs(rest - rest_ref) / std::sqrt(rest_ref * (1 - rest_ref) / draws + 1e-12));
    const bool pass = worst < 5.0;
    std::printf("%s %s (multi-block) over the vocabulary, %d draws: worst |freq - p| = %.2f sigma\n",
                pass ? "PASS" : "FAIL", o == &out ? "sample" : "sample_from_probs", draws, worst);
    ok &= pass;
  }
  return ok;
}

}  // namespace

int main() {
  if (skip_without_gpu()) return kSkip;
  INGINE_TEST_CUDA(cudaStreamCreateWithFlags(&g_stream, cudaStreamNonBlocking));
  Rng rng(31337);
  bool ok = test_argmax(rng);
  ok &= test_probs(rng, {1.0f, 0, 1.0f});
  ok &= test_probs(rng, {0.7f, 50, 1.0f});
  ok &= test_probs(rng, {0.7f, 0, 0.9f});
  ok &= test_probs(rng, {1.3f, 1000, 0.95f});
  ok &= test_sample_special(rng);
  ok &= test_sample_distribution(rng);
  ok &= test_uniform();
  ok &= test_fast_paths(rng);
  ok &= test_fast_distribution(rng);
  ok &= test_fast_truncation(rng);
  INGINE_TEST_CUDA(cudaStreamDestroy(g_stream));
  std::printf("%s\n", ok ? "ALL PASS" : "SOME FAILED");
  return ok ? 0 : 1;
}
