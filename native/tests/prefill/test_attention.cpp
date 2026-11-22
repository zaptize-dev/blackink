// Prefill attention against an FP64 oracle of model.Model._attend (+ flex's sink factor and the output gate):
// full-attention caches (BF16 and FP8 E4M3, keys past the relative extent), sliding rings (partially filled,
// wrapped, a stale slot, chunks longer than the window), ring_gather / key_tile_bounds / ring_update.
#include "kernels/prefill/prefill.h"
#include "prefill/prefill_test.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <vector>

using namespace ingine::test;
using namespace ingine::test::prefill;
namespace pk = ingine::kernels::prefill;

namespace {

constexpr int HQ = 24, HK = 8, D = 128, REL = 1024, W = 1024;
constexpr int PROJ_LD = 8664, GATE = 3072, SINK = 6528;

struct Inputs {
  int T = 0, pos0 = 0;
  std::vector<uint16_t> q, rel, proj;  // [T,24,128], [T,24,1024], [T,PROJ_LD]
};

Inputs make_inputs(Rng& rng, int T, int pos0) {
  Inputs in;
  in.T = T, in.pos0 = pos0;
  in.q.resize(size_t(T) * HQ * D);
  for (auto& x : in.q) x = bf16_bits(float(rng.normal() * 1.2));
  in.rel.resize(size_t(T) * HQ * REL);
  for (auto& x : in.rel) x = bf16_bits(float(rng.normal() * 1.5));
  in.proj.assign(size_t(T) * PROJ_LD, 0);
  for (int t = 0; t < T; ++t) {
    for (int c = 0; c < HQ * D; ++c) in.proj[size_t(t) * PROJ_LD + GATE + c] = bf16_bits(float(rng.normal()));
    for (int h = 0; h < HQ; ++h) in.proj[size_t(t) * PROJ_LD + SINK + h] = bf16_bits(float(rng.normal() * 2.0));
  }
  return in;
}

// One key/value row [8, 128]: keys of norm <= ~22, values RMS-normalised per head.
void make_kv_row(Rng& rng, float* k, float* v) {
  for (int h = 0; h < HK; ++h) {
    double ss = 0;
    for (int d = 0; d < D; ++d) {
      k[h * D + d] = rbf(rng.normal() * 1.4);
      const double x = rng.normal();
      v[h * D + d] = float(x);
      ss += x * x;
    }
    const double r = 1.0 / std::sqrt(ss / D + 1e-6);
    for (int d = 0; d < D; ++d) v[h * D + d] = rbf(v[h * D + d] * r);
  }
}

// Reference output (BF16 roundings of model._visit emulated) for rows (t, h) with t % stride == 0.
std::vector<double> oracle(const Inputs& in, const std::vector<float>& K, const std::vector<float>& V,
                           const std::vector<int>& kpos, int window, int stride, std::vector<size_t>& index) {
  std::vector<double> out;
  const int L = int(kpos.size());
  std::vector<double> s(L);
  for (int t = 0; t < in.T; t += stride) {
    const int qp = in.pos0 + t;
    for (int h = 0; h < HQ; ++h) {
      const int g = h / 3;
      const uint16_t* q = &in.q[(size_t(t) * HQ + h) * D];
      double mx = -INFINITY;
      for (int j = 0; j < L; ++j) {
        const int d = qp - kpos[j];
        const bool ok = kpos[j] >= 0 && d >= 0 && (window == 0 || d < window);
        if (!ok) {
          s[j] = -INFINITY;
          continue;
        }
        double dot = 0;
        const float* k = &K[(size_t(j) * HK + g) * D];
        for (int e = 0; e < D; ++e) dot += double(bf16_val(q[e])) * k[e];
        s[j] = dot / std::sqrt(128.0) + (d < REL ? double(bf16_val(in.rel[(size_t(t) * HQ + h) * REL + d])) : 0.0);
        mx = std::max(mx, s[j]);
      }
      const double sink = bf16_val(in.proj[size_t(t) * PROJ_LD + SINK + h]);
      double z = 0;
      std::vector<double> o(D, 0.0);
      for (int j = 0; j < L; ++j) {
        if (s[j] == -INFINITY) continue;
        const double p = std::exp(s[j] - mx);
        z += p;
        const float* v = &V[(size_t(j) * HK + g) * D];
        for (int e = 0; e < D; ++e) o[e] += p * v[e];
      }
      const double lse = mx + std::log(z);
      const double factor = 1.0 / (1.0 + std::exp(sink - lse));
      for (int e = 0; e < D; ++e) {
        const double a = rbf(o[e] / z * factor);
        const double gate = bf16_val(in.proj[size_t(t) * PROJ_LD + GATE + h * D + e]);
        out.push_back(a / (1.0 + std::exp(-gate)));
        index.push_back((size_t(t) * HQ + h) * D + e);
      }
    }
  }
  return out;
}

bool run_attention(const char* name, const Inputs& in, const void* k_dev, const void* v_dev, bool fp8,
                   const int* kpos_dev, const int4* bounds_dev, int L, int window, const std::vector<float>& K,
                   const std::vector<float>& V, const std::vector<int>& kpos, int stride) {
  DeviceArray<uint16_t> q(in.q), rel(in.rel), proj(in.proj), out(size_t(in.T) * HQ * D);
  pk::AttnParams p;
  p.q = reinterpret_cast<const __nv_bfloat16*>(q.get());
  p.k = k_dev, p.v = v_dev, p.kv_fp8 = fp8;
  p.kpos = kpos_dev, p.bounds = bounds_dev, p.L = L;
  p.T = in.T, p.pos0 = in.pos0, p.window = window;
  p.rel = reinterpret_cast<const __nv_bfloat16*>(rel.get());
  p.proj = reinterpret_cast<const __nv_bfloat16*>(proj.get()), p.ld = PROJ_LD;
  p.out = reinterpret_cast<__nv_bfloat16*>(out.get());
  pk::attention(p, nullptr);
  INGINE_TEST_CUDA(cudaDeviceSynchronize());
  const auto got_bits = out.download();
  std::vector<size_t> index;
  const auto ref = oracle(in, K, V, kpos, window, stride, index);
  std::vector<float> got(index.size());
  for (size_t i = 0; i < index.size(); ++i) got[i] = bf16_val(got_bits[index[i]]);
  // one BF16 rounding of the output (plus a rare flip of the rounded pre-gate value) and FP16 probabilities
  return check_close(name, got, ref, 1.2e-2, 2e-4);
}

bool full_case(Rng& rng, bool fp8, int pos0, int T) {
  const int L = pos0 + T, rows = L + 37;  // extra rows past L: never read
  std::vector<float> K(size_t(rows) * HK * D), V(K.size());
  for (int j = 0; j < rows; ++j) make_kv_row(rng, &K[size_t(j) * HK * D], &V[size_t(j) * HK * D]);
  DeviceArray<uint8_t> kd(K.size() * (fp8 ? 1 : 2)), vd(K.size() * (fp8 ? 1 : 2));
  if (fp8) {
    std::vector<uint8_t> kb(K.size()), vb(K.size());
    for (size_t i = 0; i < K.size(); ++i) {
      kb[i] = e4m3_bits(K[i]), vb[i] = e4m3_bits(V[i]);
      K[i] = e4m3_to_float(kb[i]), V[i] = e4m3_to_float(vb[i]);
    }
    kd.upload(kb), vd.upload(vb);
  } else {
    std::vector<uint16_t> kb(K.size()), vb(K.size());
    for (size_t i = 0; i < K.size(); ++i) kb[i] = bf16_bits(K[i]), vb[i] = bf16_bits(V[i]);
    std::vector<uint8_t> kbytes(K.size() * 2), vbytes(K.size() * 2);
    std::memcpy(kbytes.data(), kb.data(), kbytes.size());
    std::memcpy(vbytes.data(), vb.data(), vbytes.size());
    kd.upload(kbytes), vd.upload(vbytes);
  }
  std::vector<int> kpos(L);
  for (int j = 0; j < L; ++j) kpos[j] = j;
  const Inputs in = make_inputs(rng, T, pos0);
  char name[96];
  std::snprintf(name, sizeof name, "full %s pos0=%d T=%d", fp8 ? "fp8" : "bf16", pos0, T);
  return run_attention(name, in, kd.get(), vd.get(), fp8, nullptr, nullptr, L, 0, K, V, kpos, 1);
}

// Sliding visit: ring state before the chunk (ring_pos per slot), chunk keys, gathered rows.
bool sliding_case(Rng& rng, int pos0, int T, bool stale, int stride) {
  std::vector<float> ringK(size_t(W) * HK * D), ringV(ringK.size());
  std::vector<int64_t> ring_pos(W, -1);
  for (int s = 0; s < W; ++s) make_kv_row(rng, &ringK[size_t(s) * HK * D], &ringV[size_t(s) * HK * D]);
  for (int p = std::max(0, pos0 - W); p < pos0; ++p) ring_pos[p % W] = p;
  if (stale) ring_pos[(pos0 + 5) % W] = pos0 + 5;  // a slot holding a position at/after the chunk start
  std::vector<uint16_t> rk(ringK.size()), rv(ringK.size());
  for (size_t i = 0; i < rk.size(); ++i) rk[i] = bf16_bits(ringK[i]), rv[i] = bf16_bits(ringV[i]);

  const int L = W + T, kpos_rows = (L + 63) / 64 * 64;
  std::vector<float> K(size_t(L) * HK * D), V(K.size());
  std::vector<int> kpos(L);
  for (int i = 0; i < W; ++i) {
    const int slot = (pos0 + i) % W;
    std::copy_n(&ringK[size_t(slot) * HK * D], HK * D, &K[size_t(i) * HK * D]);
    std::copy_n(&ringV[size_t(slot) * HK * D], HK * D, &V[size_t(i) * HK * D]);
    kpos[i] = int(ring_pos[slot]);
  }
  std::vector<uint16_t> rows_k(size_t(L) * HK * D, 0), rows_v(rows_k.size(), 0);
  for (int t = 0; t < T; ++t) {
    float* k = &K[size_t(W + t) * HK * D];
    float* v = &V[size_t(W + t) * HK * D];
    make_kv_row(rng, k, v);
    for (int c = 0; c < HK * D; ++c)
      rows_k[size_t(W + t) * HK * D + c] = bf16_bits(k[c]), rows_v[size_t(W + t) * HK * D + c] = bf16_bits(v[c]);
    kpos[W + t] = pos0 + t;
  }
  DeviceArray<uint16_t> ring_k(rk), ring_v(rv), kr(rows_k), vr(rows_v);
  DeviceArray<int64_t> rpos(ring_pos);
  DeviceArray<int> kp(kpos_rows);
  DeviceArray<int4> bounds(kpos_rows / 64);
  pk::ring_gather(reinterpret_cast<const __nv_bfloat16*>(ring_k.get()),
                  reinterpret_cast<const __nv_bfloat16*>(ring_v.get()), rpos.get(), pos0, T,
                  reinterpret_cast<__nv_bfloat16*>(kr.get()), reinterpret_cast<__nv_bfloat16*>(vr.get()), kp.get(),
                  kpos_rows, nullptr);
  pk::key_tile_bounds(kp.get(), L, bounds.get(), nullptr);
  INGINE_TEST_CUDA(cudaDeviceSynchronize());
  bool ok = true;
  {
    const auto gk = kp.download();
    const auto gr = kr.download();
    size_t bad = 0;
    for (int i = 0; i < kpos_rows; ++i) bad += gk[i] != (i < L ? kpos[i] : -1);
    for (size_t i = 0; i < size_t(W) * HK * D; ++i) bad += gr[i] != bf16_bits(K[i]);
    std::printf("%s ring_gather rows/positions: %zu mismatches\n", bad ? "FAIL" : "PASS", bad);
    ok &= bad == 0;
  }
  const Inputs in = make_inputs(rng, T, pos0);
  char name[96];
  std::snprintf(name, sizeof name, "sliding pos0=%d T=%d%s", pos0, T, stale ? " stale" : "");
  ok &= run_attention(name, in, kr.get(), vr.get(), false, kp.get(), bounds.get(), L, W, K, V, kpos, stride);

  pk::ring_update(reinterpret_cast<const __nv_bfloat16*>(kr.get()), reinterpret_cast<const __nv_bfloat16*>(vr.get()),
                  pos0, T, reinterpret_cast<__nv_bfloat16*>(ring_k.get()),
                  reinterpret_cast<__nv_bfloat16*>(ring_v.get()), rpos.get(), nullptr);
  INGINE_TEST_CUDA(cudaDeviceSynchronize());
  {
    auto exp_pos = ring_pos;
    auto exp_k = rk;
    const int n = std::min(T, W);
    for (int c = T - n; c < T; ++c) {
      const int slot = (pos0 + c) % W;
      exp_pos[slot] = pos0 + c;
      std::copy_n(&rows_k[size_t(W + c) * HK * D], HK * D, &exp_k[size_t(slot) * HK * D]);
    }
    const auto gp = rpos.download();
    const auto gk = ring_k.download();
    size_t bad = 0;
    for (int s = 0; s < W; ++s) bad += gp[s] != exp_pos[s];
    for (size_t i = 0; i < exp_k.size(); ++i) bad += gk[i] != exp_k[i];
    std::printf("%s ring_update: %zu mismatches\n", bad ? "FAIL" : "PASS", bad);
    ok &= bad == 0;
  }
  return ok;
}

}  // namespace

int main() {
  if (skip_without_gpu()) return kSkip;
  Rng rng(1234);
  bool ok = true;
  ok &= full_case(rng, false, 0, 45);
  ok &= full_case(rng, false, 1200, 77);
  ok &= full_case(rng, true, 1200, 77);
  ok &= full_case(rng, true, 3, 130);
  ok &= sliding_case(rng, 0, 50, false, 1);
  ok &= sliding_case(rng, 700, 300, false, 3);
  ok &= sliding_case(rng, 5000, 1100, true, 7);
  std::printf("%s\n", ok ? "ALL PASS" : "SOME FAILED");
  return ok ? 0 : 1;
}
