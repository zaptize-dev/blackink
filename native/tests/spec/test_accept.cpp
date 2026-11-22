// Speculative rejection sampling (kernels::draft::sampled_accept, engine.Engine._accept) on toy
// distributions: every emitted token must be distributed as the target's p at its position, whatever the
// draft distribution q; plus the deterministic edge cases (all accepted -> bonus from p_n, zero residual ->
// p_a).
#include "common/test_util.h"
#include "kernels/draft/draft.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <vector>

using namespace ingine::test;
namespace kd = ingine::kernels::draft;

namespace {

constexpr int V = 16;
int g_failures = 0;

void expect(bool ok, const char* what) {
  std::printf("%s %s\n", ok ? "PASS" : "FAIL", what);
  if (!ok) ++g_failures;
}

std::vector<double> softmax_t(const std::vector<float>& l, float T) {
  double m = -INFINITY;
  for (float x : l) m = std::max(m, double(x) / T);
  std::vector<double> q(l.size());
  double s = 0;
  for (size_t i = 0; i < l.size(); ++i) s += (q[i] = std::exp(double(l[i]) / T - m));
  for (double& x : q) x /= s;
  return q;
}

// markov_finish's q statistics of draft-logit rows dl [n, V]: (max l / T, sum exp(l / T - max)) per row
std::vector<float> qstats_of(const std::vector<float>& dl, int n, float T) {
  std::vector<float> st(2 * n);
  for (int r = 0; r < n; ++r) {
    float m = -INFINITY, s = 0.f;
    for (int x = 0; x < V; ++x) m = std::max(m, dl[size_t(r) * V + x] / T);
    for (int x = 0; x < V; ++x) s += std::exp(dl[size_t(r) * V + x] / T - m);
    st[2 * r] = m;
    st[2 * r + 1] = s;
  }
  return st;
}

int draw(Rng& r, const std::vector<double>& w) {
  double s = 0;
  for (double x : w) s += x;
  double u = r.uniform() * s;
  for (size_t i = 0; i < w.size(); ++i) {
    if (u < w[i]) return int(i);
    u -= w[i];
  }
  for (size_t i = w.size(); i-- > 0;)
    if (w[i] > 0) return int(i);
  return 0;
}

// chi-square goodness of fit (categories with expected count 0 must be empty); df <= 15 -> 0.1% critical
// values
bool fits(const std::vector<long>& counts, const std::vector<double>& p, const char* what) {
  static const double crit[] = {0, 10.83, 13.82, 16.27, 18.47, 20.52, 22.46, 24.32, 26.12,
                                27.88, 29.59, 31.26, 32.91, 34.53, 36.12, 37.70};
  long n = 0;
  for (long c : counts) n += c;
  double chi = 0;
  int df = -1;
  bool impossible = false;
  for (int i = 0; i < V; ++i) {
    const double e = p[i] * double(n);
    if (p[i] <= 0) {
      impossible |= counts[i] != 0;
      continue;
    }
    ++df;
    chi += (counts[i] - e) * (counts[i] - e) / e;
  }
  const bool ok = !impossible && df >= 1 && chi <= crit[df];
  std::printf("%s %s: n=%ld chi2=%.2f df=%d crit=%.2f%s\n", ok ? "PASS" : "FAIL", what, n, chi, df, crit[df],
              impossible ? " (impossible token drawn)" : "");
  if (!ok) ++g_failures;
  return ok;
}

struct Toy {
  std::vector<std::vector<double>> p;   // [n + 1][V]
  std::vector<std::vector<float>> dl;   // [n][V] draft logits (-inf outside the candidates)
  float T;
};

void run_statistics(const Toy& toy, int trials, const char* name) {
  const int n = int(toy.dl.size());
  std::vector<float> p_flat, dl_flat;
  for (const auto& row : toy.p)
    for (double x : row) p_flat.push_back(float(x));
  for (const auto& row : toy.dl) dl_flat.insert(dl_flat.end(), row.begin(), row.end());
  std::vector<std::vector<double>> q;
  for (const auto& row : toy.dl) q.push_back(softmax_t(row, toy.T));
  // all trials' drafts and uniforms up front; per trial the accept result and residual are copied out on
  // the device, so the loop never synchronises
  Rng r(7 + n);
  std::vector<int32_t> drafts(size_t(trials) * n);
  std::vector<float> u(size_t(trials) * n);
  for (int t = 0; t < trials; ++t)
    for (int i = 0; i < n; ++i) {
      drafts[size_t(t) * n + i] = draw(r, q[i]);
      u[size_t(t) * n + i] = float(r.uniform());
    }
  DeviceArray<float> d_p(p_flat), d_dl(dl_flat), d_u(u), d_ws(kd::sampled_accept_workspace_floats(V)),
      d_resid(size_t(trials) * V), d_q(qstats_of(dl_flat, n, toy.T));
  DeviceArray<int32_t> d_drafts(drafts);
  DeviceArray<kd::AcceptResult> d_out(trials);
  for (int t = 0; t < trials; ++t)
    kd::sampled_accept(d_p.get(), d_dl.get(), V, d_drafts.get() + size_t(t) * n, n, toy.T, d_q.get(),
                       d_u.get() + size_t(t) * n, d_ws.get(), d_resid.get() + size_t(t) * V, d_out.get() + t, nullptr);
  INGINE_TEST_CUDA(cudaDeviceSynchronize());
  const auto out = d_out.download();
  const auto resid = d_resid.download();
  std::vector<std::vector<long>> counts(n + 1, std::vector<long>(V, 0));
  long accept0 = 0;
  for (int t = 0; t < trials; ++t) {
    const int acc = out[t].accepted;
    const int nxt = draw(r, std::vector<double>(resid.begin() + size_t(t) * V, resid.begin() + size_t(t + 1) * V));
    // emitted tokens: drafts[:acc] + [nxt]; position k's token is counted when emitted
    for (int k = 0; k <= acc; ++k) counts[k][k < acc ? drafts[size_t(t) * n + k] : nxt]++;
    accept0 += acc >= 1;
  }
  char what[96];
  for (int k = 0; k <= n; ++k) {
    std::snprintf(what, sizeof(what), "%s: token %d ~ p_%d", name, k, k);
    fits(counts[k], toy.p[k], what);
  }
  // acceptance rate of the first draft = sum_x min(p0, q0)
  double a = 0;
  for (int x = 0; x < V; ++x) a += std::min(toy.p[0][x], q[0][x]);
  const double rate = double(accept0) / trials, sd = std::sqrt(a * (1 - a) / trials);
  std::snprintf(what, sizeof(what), "%s: first-draft acceptance %.4f vs sum min(p, q) = %.4f", name, rate, a);
  expect(std::fabs(rate - a) <= 4 * sd + 1e-9, what);
}

std::vector<double> normalised(std::vector<double> w) {
  double s = 0;
  for (double x : w) s += x;
  for (double& x : w) x /= s;
  return w;
}

void edge_cases() {
  // p = q = one-hot at token 3 for row 0; u = 1 (u q < p fails) -> rejected with an all-zero residual,
  // which falls back to p_0
  {
    std::vector<float> p(2 * V, 0.f), dl(V, -INFINITY);
    p[3] = 1.f;
    p[V + 5] = 1.f;
    dl[3] = 2.f;
    DeviceArray<float> d_p(p), d_dl(dl), d_u(std::vector<float>{1.f}), d_ws(kd::sampled_accept_workspace_floats(V)),
        d_resid(V), d_q(qstats_of(dl, 1, 1.f));
    DeviceArray<int32_t> d_drafts(std::vector<int32_t>{3});
    DeviceArray<kd::AcceptResult> d_out(1);
    kd::sampled_accept(d_p.get(), d_dl.get(), V, d_drafts.get(), 1, 1.f, d_q.get(), d_u.get(), d_ws.get(), d_resid.get(),
                       d_out.get(), nullptr);
    INGINE_TEST_CUDA(cudaDeviceSynchronize());
    const auto resid = d_resid.download();
    expect(d_out.download()[0].accepted == 0 && std::equal(resid.begin(), resid.end(), p.begin()),
           "zero residual falls back to p_a");
  }
  // every draft accepted (u = 0 with p > 0) -> residual = p_n
  {
    const int n = 3;
    std::vector<float> p((n + 1) * V), dl(n * V);
    Rng r(3);
    for (auto& x : p) x = float(r.uniform(0.01, 1.0) / V);
    for (auto& x : dl) x = float(r.normal());
    DeviceArray<float> d_p(p), d_dl(dl), d_u(std::vector<float>(n, 0.f)), d_ws(kd::sampled_accept_workspace_floats(V)),
        d_resid(V), d_q(qstats_of(dl, n, 0.6f));
    DeviceArray<int32_t> d_drafts(std::vector<int32_t>{1, 2, 3});
    DeviceArray<kd::AcceptResult> d_out(1);
    kd::sampled_accept(d_p.get(), d_dl.get(), V, d_drafts.get(), n, 0.6f, d_q.get(), d_u.get(), d_ws.get(), d_resid.get(),
                       d_out.get(), nullptr);
    INGINE_TEST_CUDA(cudaDeviceSynchronize());
    const auto resid = d_resid.download();
    expect(d_out.download()[0].accepted == n && std::equal(resid.begin(), resid.end(), p.begin() + n * V),
           "all accepted: bonus from p_n");
  }
  // deterministic accept count and residual for given uniforms
  {
    const int n = 2;
    const float T = 0.9f;
    Rng r(11);
    std::vector<float> p((n + 1) * V), dl(n * V, -INFINITY);
    for (int row = 0; row <= n; ++row) {
      std::vector<double> w(V);
      for (auto& x : w) x = r.uniform();
      w = normalised(w);
      for (int x = 0; x < V; ++x) p[row * V + x] = float(w[x]);
    }
    for (int row = 0; row < n; ++row)
      for (int x = 0; x < V; x += 2) dl[row * V + x] = float(r.normal() * 2);
    const std::vector<int32_t> drafts{4, 6};
    bool ok = true;
    for (float u0 : {0.f, 0.3f, 0.99f}) {
      const std::vector<float> u{u0, 0.5f};
      DeviceArray<float> d_p(p), d_dl(dl), d_u(u), d_ws(kd::sampled_accept_workspace_floats(V)), d_resid(V),
          d_q(qstats_of(dl, n, T));
      DeviceArray<int32_t> d_drafts(drafts);
      DeviceArray<kd::AcceptResult> d_out(1);
      kd::sampled_accept(d_p.get(), d_dl.get(), V, d_drafts.get(), n, T, d_q.get(), d_u.get(), d_ws.get(), d_resid.get(),
                         d_out.get(), nullptr);
      INGINE_TEST_CUDA(cudaDeviceSynchronize());
      int acc = 0;
      std::vector<std::vector<double>> q;
      for (int row = 0; row < n; ++row)
        q.push_back(softmax_t(std::vector<float>(dl.begin() + row * V, dl.begin() + (row + 1) * V), T));
      while (acc < n && u[acc] * q[acc][drafts[acc]] < p[acc * V + drafts[acc]]) ++acc;
      ok &= d_out.download()[0].accepted == acc;
      const auto resid = d_resid.download();
      for (int x = 0; x < V; ++x) {
        const double ref = acc < n ? std::max(double(p[acc * V + x]) - q[acc][x], 0.0) : p[n * V + x];
        ok &= std::fabs(resid[x] - ref) <= 1e-6;
      }
    }
    expect(ok, "accept count and residual match the host computation");
  }
}

}  // namespace

int main() {
  if (skip_without_gpu()) return kSkip;
  Rng r(99);
  // p and q overlapping partly; q has support outside p and p outside q
  Toy a;
  a.T = 0.8f;
  for (int row = 0; row < 3; ++row) {
    std::vector<double> w(V);
    for (int x = 0; x < V; ++x) w[x] = (x % 5 == row) ? 0.0 : r.uniform(0.0, 1.0);
    a.p.push_back(normalised(w));
  }
  for (int row = 0; row < 2; ++row) {
    std::vector<float> l(V, -INFINITY);
    for (int x = 0; x < V; ++x)
      if (x % 3 != row) l[x] = float(r.normal() * 1.5);
    a.dl.push_back(l);
  }
  run_statistics(a, 30000, "toy 2 drafts");
  // a confident draft close to p
  Toy b;
  b.T = 0.5f;
  {
    std::vector<double> w(V, 0.01);
    w[2] = 1.0;
    b.p.push_back(normalised(w));
    std::vector<double> w2(V, 0.05);
    w2[9] = 0.5;
    b.p.push_back(normalised(w2));
    std::vector<float> l(V, -INFINITY);
    l[2] = 3.f;
    l[7] = 1.f;
    l[11] = 0.f;
    b.dl.push_back(l);
  }
  run_statistics(b, 30000, "toy 1 draft");
  edge_cases();
  std::printf("%s: %d failure(s)\n", g_failures ? "FAILED" : "OK", g_failures);
  return g_failures ? 1 : 0;
}
