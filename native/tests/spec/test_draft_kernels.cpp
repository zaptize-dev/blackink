// CPU-oracle tests of the DSpark draft kernels (src/kernels/draft) at small sizes. The oracles repeat
// dspark.py's rounding points in FP32 (reductions in FP64), so results agree to about one BF16 ulp; discrete
// outputs (candidate sets, picks, tokens, slots) must match exactly or be consistent with the kernel's own
// logits.
#include "common/test_util.h"
#include "kernels/draft/draft.h"
#include "kernels/draft/rng.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <functional>
#include <numeric>
#include <vector>

using namespace ingine::test;
namespace kd = ingine::kernels::draft;
using bf16 = __nv_bfloat16;

namespace {

constexpr int H = kd::H, D = kd::HEAD_DIM;
int g_failures = 0;

void expect(bool ok, const char* what) {
  if (!ok) {
    std::printf("FAIL %s\n", what);
    ++g_failures;
  }
}

std::vector<bf16> rand_bf16(Rng& r, size_t n, double scale = 1.0) {
  std::vector<bf16> v(n);
  for (auto& x : v) x = float_to_bf16(float(r.normal() * scale));
  return v;
}
std::vector<float> to_float(const std::vector<bf16>& v) {
  std::vector<float> f(v.size());
  for (size_t i = 0; i < v.size(); ++i) f[i] = bf16_to_float(v[i]);
  return f;
}
float bfr(float x) { return round_bf16(x); }

// float8_e4m3fn: the nearest code (ties to the even code, saturating at +-448) and its value
uint8_t e4m3_encode(float x) {
  static const std::vector<float> values = [] {
    std::vector<float> v(127);  // codes 0..126 ascend from 0 to 448 (127 is NaN)
    for (int c = 0; c < 127; ++c) v[c] = e4m3_to_float(uint8_t(c));
    return v;
  }();
  const float a = std::fabs(x);
  int hi = int(std::lower_bound(values.begin(), values.end(), a) - values.begin());  // first value >= a
  int best = hi;
  if (hi == 127) {
    best = 126;
  } else if (hi > 0) {
    const float up = values[hi] - a, down = a - values[hi - 1];
    best = up < down || (up == down && (hi & 1) == 0) ? hi : hi - 1;
  }
  return uint8_t(best | (std::signbit(x) ? 0x80 : 0));
}
float e4m3r(float x) { return e4m3_to_float(e4m3_encode(x)); }

// a key/value cache as floats: BF16 or E4M3 rows
std::vector<float> cache_floats(const DeviceArray<uint8_t>& a, bool fp8) {
  const auto raw = a.download();
  std::vector<float> f(fp8 ? raw.size() : raw.size() / 2);
  for (size_t i = 0; i < f.size(); ++i) {
    if (fp8) {
      f[i] = e4m3_to_float(raw[i]);
    } else {
      bf16 b;
      std::memcpy(&b, &raw[2 * i], 2);
      f[i] = bf16_to_float(b);
    }
  }
  return f;
}
std::vector<uint8_t> cache_bytes(const std::vector<float>& f, bool fp8) {
  std::vector<uint8_t> raw(fp8 ? f.size() : 2 * f.size());
  for (size_t i = 0; i < f.size(); ++i) {
    if (fp8) {
      raw[i] = e4m3_encode(f[i]);
    } else {
      const bf16 b = float_to_bf16(f[i]);
      std::memcpy(&raw[2 * i], &b, 2);
    }
  }
  return raw;
}

// cache contents against BF16 reference values: within `ulps` BF16 ulps (check_close), or for E4M3 rows the
// E4M3 rounding of a value that close to the reference
bool check_cache(const std::vector<float>& got, const std::vector<float>& ref, bool fp8, const char* what,
                 double ulps = 2.0, double floor = 1e-3);

// every |got - ref| within `ulps` BF16 ulps of max(|ref|, floor)
bool check_close(const std::vector<float>& got, const std::vector<float>& ref, const char* what, double ulps = 2.0,
           double floor = 1e-3) {
  double worst = 0;
  size_t wi = 0;
  for (size_t i = 0; i < ref.size(); ++i) {
    const double scale = std::max(std::fabs(double(ref[i])), floor);
    const double e = std::fabs(double(got[i]) - double(ref[i])) / (scale * std::ldexp(1.0, -7));
    if (!(e <= worst)) {
      worst = e;
      wi = i;
    }
  }
  const bool ok = worst <= ulps;
  std::printf("%s %s: worst %.2f ulp at %zu (got %.6g ref %.6g)\n", ok ? "PASS" : "FAIL", what, worst, wi,
              ref.empty() ? 0.0 : double(got[wi]), ref.empty() ? 0.0 : double(ref[wi]));
  if (!ok) ++g_failures;
  return ok;
}

bool check_cache(const std::vector<float>& got, const std::vector<float>& ref, bool fp8, const char* what, double ulps,
                 double floor) {
  if (!fp8) return check_close(got, ref, what, ulps, floor);
  size_t bad = 0, first = 0;
  for (size_t i = 0; i < ref.size(); ++i) {
    const float tol = float(ulps * std::max(std::fabs(double(ref[i])), floor) * std::ldexp(1.0, -7));
    if (!(got[i] >= e4m3r(ref[i] - tol) && got[i] <= e4m3r(ref[i] + tol)) && bad++ == 0) first = i;
  }
  std::printf("%s %s: %zu of %zu values are not the E4M3 rounding (first at %zu: got %.6g ref %.6g)\n",
              bad ? "FAIL" : "PASS", what, bad, ref.size(), first, ref.empty() ? 0.0 : double(got[first]),
              ref.empty() ? 0.0 : double(ref[first]));
  if (bad) ++g_failures;
  return bad == 0;
}

std::vector<float> download_bf16(const DeviceArray<bf16>& a) { return to_float(a.download()); }

// dspark.rms over a row (FP64 sum of squares), times w, rounded to BF16
std::vector<float> rms_row(const float* x, int n, const float* w) {
  double ss = 0;
  for (int i = 0; i < n; ++i) ss += double(x[i]) * x[i];
  const float rs = 1.f / std::sqrt(float(ss / n) + kd::EPS);
  std::vector<float> y(n);
  for (int i = 0; i < n; ++i) y[i] = bfr((x[i] * rs) * w[i]);
  return y;
}

// neox rope of one BF16 head row at `pos`, rounded to BF16
std::vector<float> rope(const std::vector<float>& y, int pos) {
  std::vector<float> o(D);
  for (int i = 0; i < D / 2; ++i) {
    const float inv = std::pow(kd::ROPE_THETA, (-float(i) * 2.f) / float(D));
    const float ang = float(pos) * inv;
    const float c = std::cos(ang), s = std::sin(ang);
    o[i] = bfr(y[i] * c - y[i + D / 2] * s);
    o[i + D / 2] = bfr(y[i + D / 2] * c + y[i] * s);
  }
  return o;
}

std::vector<float> norm_rope(const float* x, const float* w, int pos) { return rope(rms_row(x, D, w), pos); }

void set_params(kd::DraftParams* p, int anchor, int P, float T, int start, uint64_t seed) {
  kd::set_seed(p, seed, nullptr);
  kd::set_draft_inputs(p, anchor, P, T, nullptr);
  kd::set_commit_start(p, start, nullptr);
  INGINE_TEST_CUDA(cudaDeviceSynchronize());
}

// ---- tests -------------------------------------------------------------------------------------------

void test_mix_rmsnorm(Rng& r) {
  const int n = 5;
  auto feats = rand_bf16(r, size_t(n) * 4 * H);
  auto alpha = rand_bf16(r, 4, 0.5);
  auto w = rand_bf16(r, H, 0.3);
  DeviceArray<bf16> d_feats(feats), d_alpha(alpha), d_w(w), d_out(size_t(n) * H);
  kd::mix_rmsnorm(d_feats.get(), n, d_alpha.get(), d_w.get(), d_out.get(), nullptr);
  INGINE_TEST_CUDA(cudaDeviceSynchronize());
  const auto f = to_float(feats), a = to_float(alpha), wf = to_float(w);
  std::vector<float> ref;
  for (int i = 0; i < n; ++i) {
    std::vector<float> m(H);
    for (int d = 0; d < H; ++d) {
      const float* fr = f.data() + size_t(i) * 4 * H;
      m[d] = ((fr[d] * a[0] + fr[H + d] * a[1]) + fr[2 * H + d] * a[2]) + fr[3 * H + d] * a[3];
    }
    auto y = rms_row(m.data(), H, wf.data());
    ref.insert(ref.end(), y.begin(), y.end());
  }
  check_close(download_bf16(d_out), ref, "mix_rmsnorm");
}

void test_ctx_kv_store(Rng& r) {
  DeviceArray<kd::DraftParams> p(1);
  for (const int n : {6, 20}) {
    const int size = 16, start = 1000, row0 = 3;
    const int keep_from = n - std::min(n, size);
    set_params(p.get(), 0, 0, 0.f, start, 1);
    auto kv = rand_bf16(r, size_t(n) * 2 * kd::KV_DIM);
    auto kn = rand_bf16(r, D, 0.5);
    DeviceArray<bf16> d_kv(kv), d_kn(kn), d_k(size_t(size) * kd::KV_DIM), d_v(size_t(size) * kd::KV_DIM);
    DeviceArray<int32_t> d_pos(std::vector<int32_t>(size, -1));
    kd::ctx_kv_store(d_kv.get(), n, p.get(), row0, keep_from, d_kn.get(), d_k.get(), d_v.get(), false, size,
                     d_pos.get(), nullptr);
    INGINE_TEST_CUDA(cudaDeviceSynchronize());
    const auto kvf = to_float(kv), knf = to_float(kn);
    std::vector<float> ref_k(size_t(size) * kd::KV_DIM, 0.f), ref_v(size_t(size) * kd::KV_DIM, 0.f);
    std::vector<int32_t> ref_pos(size, -1);
    for (int i = keep_from; i < n; ++i) {
      const int pos = start + row0 + i, slot = pos % size;
      ref_pos[slot] = pos;
      for (int g = 0; g < kd::KV_HEADS; ++g) {
        auto k = norm_rope(kvf.data() + size_t(i) * 2 * kd::KV_DIM + g * D, knf.data(), pos);
        for (int d = 0; d < D; ++d) {
          ref_k[(size_t(slot) * kd::KV_HEADS + g) * D + d] = k[d];
          ref_v[(size_t(slot) * kd::KV_HEADS + g) * D + d] = kvf[size_t(i) * 2 * kd::KV_DIM + kd::KV_DIM + g * D + d];
        }
      }
    }
    check_close(download_bf16(d_k), ref_k, n > size ? "ctx_kv_store k (wrapped)" : "ctx_kv_store k");
    check_close(download_bf16(d_v), ref_v, "ctx_kv_store v", 0.0, 1e-30);
    expect(d_pos.download() == ref_pos, "ctx_kv_store ctx_pos");
    // the global ring: E4M3 rows, no position tags
    DeviceArray<uint8_t> d_k8(size_t(size) * kd::KV_DIM), d_v8(size_t(size) * kd::KV_DIM);
    kd::ctx_kv_store(d_kv.get(), n, p.get(), row0, keep_from, d_kn.get(), d_k8.get(), d_v8.get(), true, size, nullptr,
                     nullptr);
    INGINE_TEST_CUDA(cudaDeviceSynchronize());
    check_cache(cache_floats(d_k8, true), ref_k, true, "ctx_kv_store k (E4M3)");
    check_cache(cache_floats(d_v8, true), ref_v, true, "ctx_kv_store v (E4M3)", 0.0, 1e-30);
  }
}

// round_begin (the device-side round transition: parameters, commit_mix and embed_block in one launch) and
// commit_store (the small commit)
void test_small_commit(Rng& r) {
  const int widths[3] = {2, 4, 8};
  std::vector<std::vector<bf16>> taps;
  std::vector<DeviceArray<bf16>> d_taps;
  d_taps.reserve(3);
  for (int w : widths) {
    taps.push_back(rand_bf16(r, size_t(w) * 4 * H));
    d_taps.emplace_back(taps.back());
  }
  auto alpha = rand_bf16(r, 16, 0.5);
  auto f_norm = rand_bf16(r, H, 0.3);
  std::vector<std::vector<bf16>> ctx_norm, k_norm;
  std::vector<DeviceArray<bf16>> d_ctx_norm, d_k_norm;
  std::vector<DeviceArray<uint8_t>> d_k, d_v;  // layer 1 (the global ring): E4M3 rows
  std::vector<DeviceArray<int32_t>> d_pos;
  const int sizes[3] = {16, 64, 16};
  kd::CtxLayers L;
  for (int l = 0; l < 3; ++l) {
    ctx_norm.push_back(rand_bf16(r, H, 0.3));
    k_norm.push_back(rand_bf16(r, D, 0.5));
  }
  d_ctx_norm.reserve(3);
  d_k_norm.reserve(3);
  d_k.reserve(3);
  d_v.reserve(3);
  d_pos.reserve(3);
  for (int l = 0; l < 3; ++l) {
    d_ctx_norm.emplace_back(ctx_norm[l]);
    d_k_norm.emplace_back(k_norm[l]);
    d_k.emplace_back(size_t(sizes[l]) * kd::KV_DIM * (l == 1 ? 1 : 2));
    d_v.emplace_back(size_t(sizes[l]) * kd::KV_DIM * (l == 1 ? 1 : 2));
    d_pos.emplace_back(std::vector<int32_t>(sizes[l], -1));
    L.ctx_norm[l] = d_ctx_norm[l].get();
    L.k_norm_ctx[l] = d_k_norm[l].get();
    L.k_cache[l] = d_k[l].get();
    L.v_cache[l] = d_v[l].get();
    L.fp8[l] = l == 1;
    L.ctx_pos[l] = l == 1 ? nullptr : d_pos[l].get();
    L.size[l] = sizes[l];
  }
  DeviceArray<bf16> d_alpha(alpha), d_f_norm(f_norm), d_xin(size_t(3) * 8 * H), d_f_rnn(H);
  DeviceArray<kd::DraftParams> p(1);
  DeviceArray<int32_t> d_m(1), d_acc(1), d_next(std::vector<int32_t>{4321}), d_next_s(std::vector<int32_t>{777});
  const auto af = to_float(alpha), fnf = to_float(f_norm);
  // the block's input rows: an embedding holding the two next-anchor tokens
  const int EV = 4400;
  auto embed = rand_bf16(r, size_t(EV) * H), mask = rand_bf16(r, H), in_norm = rand_bf16(r, H, 0.5);
  std::vector<float> en(H);
  for (auto& x : en) x = float(r.uniform(0.5, 1.5));
  DeviceArray<bf16> d_embed(embed), d_mask(mask), d_in(in_norm), d_res(size_t(8) * H), d_xn(size_t(8) * H),
      d_res2(size_t(8) * H), d_xn2(size_t(8) * H);
  DeviceArray<float> d_en(en);
  DeviceArray<unsigned long long> d_picks(std::vector<unsigned long long>(7, 123)), d_picks2(7);
  struct Case {
    int m, acc, start;
    float T;
  };
  for (const Case c : {Case{2, 0, 500, 0.f}, Case{4, 2, 13, 0.6f}, Case{8, 7, 1000, 0.f}, Case{8, 3, 61, 1.f}}) {
    set_params(p.get(), 5, c.start, c.T, 0, 1);  // the last draft was at P = start
    d_m.upload({c.m});
    d_acc.upload({c.acc});
    kd::RoundInputs in;
    in.nwidths = 3;
    for (int i = 0; i < 3; ++i) {
      in.widths[i] = widths[i];
      in.taps[i] = d_taps[i].get();
    }
    in.m = d_m.get();
    in.accepted = d_acc.get();
    in.next_greedy = d_next.get();
    in.next_sampled = d_next_s.get();
    kd::round_begin(p.get(), in, d_alpha.get(), L, d_f_norm.get(), d_embed.get(), d_en.get(), d_mask.get(), d_in.get(),
                    d_xin.get(), d_f_rnn.get(), d_res.get(), d_xn.get(), d_picks.get(), nullptr);
    INGINE_TEST_CUDA(cudaDeviceSynchronize());
    const int wi = c.m == 2 ? 0 : (c.m == 4 ? 1 : 2), n = c.acc + 1;
    const kd::DraftParams got = p.download()[0];
    char what[96];
    std::snprintf(what, sizeof(what), "round_begin parameters m=%d accepted=%d T=%.1f", c.m, c.acc, c.T);
    expect(got.ncommit == n && got.commit_feats == d_taps[wi].get() && got.start == c.start &&
               got.P == c.start + n && got.anchor == (c.T > 0.f ? 777 : 4321) && got.temperature == c.T &&
               got.seed == 1,
           what);
    // its block rows are embed_block's for the published anchor
    kd::embed_block(d_embed.get(), d_en.get(), d_mask.get(), d_in.get(), p.get(), d_res2.get(), d_xn2.get(),
                    d_picks2.get(), nullptr);
    INGINE_TEST_CUDA(cudaDeviceSynchronize());
    const auto picks = d_picks.download();
    expect(download_bf16(d_res) == download_bf16(d_res2) && download_bf16(d_xn) == download_bf16(d_xn2) &&
               std::all_of(picks.begin(), picks.end(), [](auto v) { return v == 0; }),
           "round_begin block rows == embed_block");

    std::vector<float> kv(size_t(8) * 3 * 2 * kd::KV_DIM);
    for (auto& x : kv) x = bfr(float(r.normal()));
    DeviceArray<float> d_kv(kv);
    kd::commit_store(d_kv.get(), p.get(), L, nullptr);
    INGINE_TEST_CUDA(cudaDeviceSynchronize());

    const auto tf = to_float(taps[wi]);
    auto mix = [&](int row, int a, const std::vector<float>& w) {
      std::vector<float> m(H);
      const float* fr = tf.data() + size_t(row) * 4 * H;
      const float* al = af.data() + 4 * a;
      for (int d = 0; d < H; ++d)
        m[d] = ((fr[d] * al[0] + fr[H + d] * al[1]) + fr[2 * H + d] * al[2]) + fr[3 * H + d] * al[3];
      return rms_row(m.data(), H, w.data());
    };
    std::vector<float> ref_xin(size_t(3) * 8 * H, 0.f);
    for (int l = 0; l < 3; ++l)
      for (int row = 0; row < n; ++row) {
        auto y = mix(row, l, to_float(ctx_norm[l]));
        std::copy(y.begin(), y.end(), ref_xin.begin() + (size_t(l) * 8 + row) * H);
      }
    std::snprintf(what, sizeof(what), "commit_mix xin (n=%d)", n);
    check_close(download_bf16(d_xin), ref_xin, what);
    std::snprintf(what, sizeof(what), "commit_mix f_rnn (row %d)", n - 1);
    check_close(download_bf16(d_f_rnn), mix(n - 1, 3, fnf), what);

    for (int l = 0; l < 3; ++l) {
      const auto k = cache_floats(d_k[l], l == 1), v = cache_floats(d_v[l], l == 1);
      const auto pos = d_pos[l].download();
      const auto knf = to_float(k_norm[l]);
      std::vector<float> got_k, got_v, ref_k, ref_v;
      bool pos_ok = true;
      for (int row = 0; row < n; ++row) {
        const int ps = c.start + row, slot = ps % sizes[l];
        if (l != 1) pos_ok &= pos[slot] == ps;
        const float* kr = kv.data() + size_t(row) * 6 * kd::KV_DIM + size_t(l) * 2 * kd::KV_DIM;
        for (int g = 0; g < kd::KV_HEADS; ++g) {
          auto kk = norm_rope(kr + g * D, knf.data(), ps);
          ref_k.insert(ref_k.end(), kk.begin(), kk.end());
          ref_v.insert(ref_v.end(), kr + kd::KV_DIM + g * D, kr + kd::KV_DIM + (g + 1) * D);
          const size_t at = (size_t(slot) * kd::KV_HEADS + g) * D;
          got_k.insert(got_k.end(), k.begin() + at, k.begin() + at + D);
          got_v.insert(got_v.end(), v.begin() + at, v.begin() + at + D);
        }
      }
      // RoPE outputs near zero come from cancellation: absolute tolerance at the head's scale
      std::snprintf(what, sizeof(what), "commit_store layer %d k (n=%d)", l, n);
      check_cache(got_k, ref_k, l == 1, what, 2.0, 2e-2);
      std::snprintf(what, sizeof(what), "commit_store layer %d v", l);
      check_cache(got_v, ref_v, l == 1, what, 0.0, 1e-30);
      std::snprintf(what, sizeof(what), "commit_store layer %d ctx_pos", l);
      expect(pos_ok, what);
    }
  }
}

void test_embed_block(Rng& r) {
  const int V = 50, anchor = 7;
  auto embed = rand_bf16(r, size_t(V) * H);
  std::vector<float> en(H);
  for (auto& x : en) x = float(r.uniform(0.5, 1.5));
  auto mask = rand_bf16(r, H);
  auto in_norm = rand_bf16(r, H, 0.5);
  DeviceArray<bf16> d_embed(embed), d_mask(mask), d_in(in_norm), d_res(size_t(8) * H), d_xn(size_t(8) * H);
  DeviceArray<float> d_en(en);
  DeviceArray<unsigned long long> d_picks(std::vector<unsigned long long>(7, 123));
  DeviceArray<kd::DraftParams> p(1);
  set_params(p.get(), anchor, 0, 0.f, 0, 1);
  kd::embed_block(d_embed.get(), d_en.get(), d_mask.get(), d_in.get(), p.get(), d_res.get(), d_xn.get(),
                  d_picks.get(), nullptr);
  INGINE_TEST_CUDA(cudaDeviceSynchronize());
  const auto ef = to_float(embed), mf = to_float(mask), inf = to_float(in_norm);
  std::vector<float> ref_res, ref_xn;
  auto x0 = rms_row(ef.data() + size_t(anchor) * H, H, en.data());
  for (int i = 0; i < 8; ++i) {
    const std::vector<float>& x = i == 0 ? x0 : mf;
    ref_res.insert(ref_res.end(), x.begin(), x.end());
    auto y = rms_row(x.data(), H, inf.data());
    ref_xn.insert(ref_xn.end(), y.begin(), y.end());
  }
  check_close(download_bf16(d_res), ref_res, "embed_block res");
  check_close(download_bf16(d_xn), ref_xn, "embed_block xn");
  const auto picks = d_picks.download();
  expect(std::all_of(picks.begin(), picks.end(), [](auto v) { return v == 0; }), "embed_block clears picks");
}

void test_qkv_rope(Rng& r) {
  const int P = 37, ctx = 64;
  std::vector<float> qkv(size_t(8) * kd::QKV_ROWS);
  for (auto& x : qkv) x = bfr(float(r.normal()));
  auto qn = rand_bf16(r, D, 0.5), kn = rand_bf16(r, D, 0.5);
  DeviceArray<float> d_qkv(qkv);
  DeviceArray<bf16> d_qn(qn), d_kn(kn), d_q(size_t(8) * kd::Q_DIM);
  DeviceArray<kd::DraftParams> p(1);
  set_params(p.get(), 0, P, 0.f, 0, 1);
  const auto qnf = to_float(qn), knf = to_float(kn);
  struct Kind {
    bool local, fp8;
    const char* q;
    const char* k;
    const char* v;
  };
  for (const Kind kind : {Kind{true, false, "qkv_rope q (local)", "qkv_rope k (local)", "qkv_rope v (local)"},
                          Kind{false, false, "qkv_rope q (global)", "qkv_rope k (global)", "qkv_rope v (global)"},
                          Kind{false, true, "qkv_rope q (global E4M3)", "qkv_rope k (global E4M3)",
                               "qkv_rope v (global E4M3)"}}) {
    const bool local = kind.local;
    const int slots = local ? kd::WINDOW + 8 : ctx;
    DeviceArray<uint8_t> d_k(size_t(slots) * kd::KV_DIM * (kind.fp8 ? 1 : 2)),
        d_v(size_t(slots) * kd::KV_DIM * (kind.fp8 ? 1 : 2));
    kd::qkv_rope(d_qkv.get(), d_qn.get(), d_kn.get(), p.get(), d_q.get(), d_k.get(), d_v.get(), kind.fp8, local, ctx,
                 nullptr);
    INGINE_TEST_CUDA(cudaDeviceSynchronize());
    std::vector<float> ref_q(size_t(8) * kd::Q_DIM), ref_k(size_t(slots) * kd::KV_DIM, 0.f),
        ref_v(size_t(slots) * kd::KV_DIM, 0.f);
    for (int i = 0; i < 8; ++i) {
      const float* row = qkv.data() + size_t(i) * kd::QKV_ROWS;
      for (int h = 0; h < kd::HEADS; ++h) {
        auto q = norm_rope(row + h * D, qnf.data(), P + i);
        std::copy(q.begin(), q.end(), ref_q.begin() + (size_t(i) * kd::HEADS + h) * D);
      }
      const int slot = local ? kd::WINDOW + i : (P + i) % ctx;
      for (int g = 0; g < kd::KV_HEADS; ++g) {
        auto k = norm_rope(row + kd::Q_DIM + g * D, knf.data(), P + i);
        std::copy(k.begin(), k.end(), ref_k.begin() + (size_t(slot) * kd::KV_HEADS + g) * D);
        std::copy(row + kd::Q_DIM + kd::KV_DIM + g * D, row + kd::Q_DIM + kd::KV_DIM + (g + 1) * D,
                  ref_v.begin() + (size_t(slot) * kd::KV_HEADS + g) * D);
      }
    }
    check_close(download_bf16(d_q), ref_q, kind.q);
    check_cache(cache_floats(d_k, kind.fp8), ref_k, kind.fp8, kind.k);
    check_cache(cache_floats(d_v, kind.fp8), ref_v, kind.fp8, kind.v, 0.0, 1e-30);
  }
}

void test_block_attention(Rng& r) {
  struct Case {
    bool local;
    int nkeys, P;
    bool holes;  // local: whole runs of slots empty
    bool fp8;    // global: E4M3 cache rows
    const char* name;
  };
  // every chunking class: local (2 tiles of 16 keys per CTA), global with 2 tiles per CTA (<= 4096 keys) and
  // 4 / 8 / 32 tiles per CTA (8K, 16K, 64K keys)
  const Case cases[] = {{true, kd::WINDOW + 8, 1500, false, false, "attention local (wrapped ring)"},
                        {true, kd::WINDOW + 8, 1500, true, false, "attention local (wrapped ring, empty runs)"},
                        {true, kd::WINDOW + 8, 300, false, false, "attention local (partial ring)"},
                        {true, kd::WINDOW + 8, 0, false, false, "attention local (empty ring)"},
                        {false, 1536, 1300, false, false, "attention global (masked tail)"},
                        {false, 1024, 2000, false, false, "attention global (full ring)"},
                        {false, 2048, 5, false, false, "attention global (short context)"},
                        {false, 1000, 990, false, false, "attention global (ring not a multiple of a tile)"},
                        {false, 8192, 7000, false, false, "attention global 8K (masked tail)"},
                        {false, 16384, 20000, false, false, "attention global 16K"},
                        {false, 65536, 40000, false, false, "attention global 64K (masked tail)"},
                        {false, 1536, 1300, false, true, "attention global E4M3 (masked tail)"},
                        {false, 2048, 5, false, true, "attention global E4M3 (short context)"},
                        {false, 1000, 2000, false, true, "attention global E4M3 (ring not a multiple of a tile)"},
                        {false, 65536, 40003, false, true, "attention global E4M3 64K (masked tail)"}};
  DeviceArray<kd::DraftParams> p(1);
  for (const Case& c : cases) {
    set_params(p.get(), 0, c.P, 0.f, 0, 1);
    auto q = rand_bf16(r, size_t(8) * kd::Q_DIM);
    auto kf = to_float(rand_bf16(r, size_t(c.nkeys) * kd::KV_DIM));
    auto vf = to_float(rand_bf16(r, size_t(c.nkeys) * kd::KV_DIM));
    const std::vector<uint8_t> kb = cache_bytes(kf, c.fp8), vb = cache_bytes(vf, c.fp8);
    if (c.fp8) {  // the stored values
      for (size_t i = 0; i < kb.size(); ++i) kf[i] = e4m3_to_float(kb[i]);
      for (size_t i = 0; i < vb.size(); ++i) vf[i] = e4m3_to_float(vb[i]);
    }
    std::vector<int32_t> cpos(kd::WINDOW);
    for (int j = 0; j < kd::WINDOW; ++j) {
      // the ring as commits leave it: slot j holds the latest position = j mod 1024 below P, some empty
      int pos = (c.P - 1) - ((c.P - 1 - j) % kd::WINDOW + kd::WINDOW) % kd::WINDOW;
      cpos[j] = pos >= 0 && r.uniform() > 0.05 ? pos : -1;
      if (c.holes && ((j >= 128 && j < 320) || j >= 960)) cpos[j] = -1;
    }
    DeviceArray<bf16> d_q(q), d_o(size_t(8) * kd::Q_DIM);
    DeviceArray<uint8_t> d_k(kb), d_v(vb);
    DeviceArray<int32_t> d_cpos(cpos);
    DeviceArray<float> ws(kd::attention_workspace_floats(c.nkeys));
    kd::block_attention(d_q.get(), d_k.get(), d_v.get(), c.nkeys, c.local ? d_cpos.get() : nullptr, c.fp8, p.get(),
                        ws.get(), d_o.get(), nullptr);
    INGINE_TEST_CUDA(cudaDeviceSynchronize());
    const auto qf = to_float(q);
    // The kernel rounds the probabilities to BF16 (the value mma's operand): the error of an output is
    // bounded by 2^-9 of sum_j p_j |v_j| (plus the FP32 score and softmax errors), not by an ulp of the
    // output itself (values cancel).
    std::vector<float> ref(size_t(8) * kd::Q_DIM), bound(size_t(8) * kd::Q_DIM);
    for (int h = 0; h < kd::HEADS; ++h) {
      const int g = h / 3;
      for (int i = 0; i < 8; ++i) {
        std::vector<double> s(c.nkeys, -INFINITY);
        double m = -INFINITY;
        for (int j = 0; j < c.nkeys; ++j) {
          bool valid;
          if (c.local) valid = j >= kd::WINDOW || (cpos[j] >= 0 && cpos[j] < c.P && cpos[j] >= c.P - kd::WINDOW);
          else valid = j < c.P + 8;
          if (!valid) continue;
          double dot = 0;
          for (int d = 0; d < D; ++d)
            dot += double(qf[(size_t(i) * kd::HEADS + h) * D + d]) * kf[(size_t(j) * kd::KV_HEADS + g) * D + d];
          s[j] = dot / std::sqrt(double(D));
          m = std::max(m, s[j]);
        }
        double l = 0;
        std::vector<double> o(D, 0.0), oa(D, 0.0);
        for (int j = 0; j < c.nkeys; ++j) {
          if (s[j] == -INFINITY) continue;
          const double e = std::exp(s[j] - m);
          l += e;
          for (int d = 0; d < D; ++d) {
            const double v = vf[(size_t(j) * kd::KV_HEADS + g) * D + d];
            o[d] += e * v;
            oa[d] += e * std::fabs(v);
          }
        }
        for (int d = 0; d < D; ++d) {
          ref[size_t(i) * kd::Q_DIM + h * D + d] = bfr(float(o[d] / l));
          bound[size_t(i) * kd::Q_DIM + h * D + d] = float(oa[d] / l);
        }
      }
    }
    const auto got = download_bf16(d_o);
    double worst = 0;
    for (size_t i = 0; i < ref.size(); ++i) {
      const double tol = std::ldexp(double(bound[i]), -8) + std::ldexp(std::fabs(double(ref[i])), -7);
      worst = std::max(worst, std::fabs(double(got[i]) - double(ref[i])) / tol);
    }
    std::printf("%s %s: worst error %.2f of the BF16-probability bound\n", worst <= 1.0 ? "PASS" : "FAIL", c.name, worst);
    expect(worst <= 1.0, c.name);
  }
}

void test_add_rmsnorm_silu(Rng& r) {
  const int rows = 3, inter = 640;
  auto res = rand_bf16(r, size_t(rows) * H), delta = rand_bf16(r, size_t(rows) * H), w = rand_bf16(r, H, 0.5);
  DeviceArray<bf16> d_res(res), d_w(w), d_xn(size_t(rows) * H);
  DeviceArray<float> d_delta(to_float(delta));
  kd::add_rmsnorm(d_res.get(), d_delta.get(), d_w.get(), d_xn.get(), rows, nullptr);
  std::vector<float> gu(size_t(rows) * 2 * inter);
  for (auto& x : gu) x = bfr(float(r.normal() * 3));
  DeviceArray<float> d_gu(gu);
  DeviceArray<bf16> d_act(size_t(rows) * inter);
  kd::silu_mul(d_gu.get(), rows, inter, d_act.get(), nullptr);
  INGINE_TEST_CUDA(cudaDeviceSynchronize());
  const auto rf = to_float(res), df = to_float(delta), wf = to_float(w);
  std::vector<float> ref_res(rf.size()), ref_xn;
  for (size_t i = 0; i < rf.size(); ++i) ref_res[i] = bfr(rf[i] + df[i]);
  for (int i = 0; i < rows; ++i) {
    auto y = rms_row(ref_res.data() + size_t(i) * H, H, wf.data());
    ref_xn.insert(ref_xn.end(), y.begin(), y.end());
  }
  check_close(download_bf16(d_res), ref_res, "add_rmsnorm res", 0.0, 1e-30);
  check_close(download_bf16(d_xn), ref_xn, "add_rmsnorm xn");
  std::vector<float> ref_act(size_t(rows) * inter);
  for (int i = 0; i < rows; ++i)
    for (int j = 0; j < inter; ++j) {
      const float g = gu[size_t(i) * 2 * inter + j], u = gu[size_t(i) * 2 * inter + inter + j];
      ref_act[size_t(i) * inter + j] = bfr((g / (1.f + std::exp(-g))) * u);
    }
  check_close(download_bf16(d_act), ref_act, "silu_mul");
}

// reference top-k: the elements above the 1024th value by ascending index, then the lowest-index elements
// equal to it
std::vector<int> topk_ref(const float* x, int V) {
  std::vector<float> sorted(x, x + V);
  std::nth_element(sorted.begin(), sorted.begin() + (kd::CANDIDATES - 1), sorted.end(), std::greater<float>());
  const float T = sorted[kd::CANDIDATES - 1];
  std::vector<int> idx;
  for (int i = 0; i < V; ++i)
    if (x[i] > T) idx.push_back(i);
  for (int i = 0; i < V && int(idx.size()) < kd::CANDIDATES; ++i)
    if (x[i] == T) idx.push_back(i);
  return idx;
}

void test_topk(Rng& r) {
  const char* names[] = {"normal", "heavy ties", "negative", "3000 equal maxima across slices", "all equal",
                         "1024 equal at the threshold"};
  const int rows = 6;
  for (const int V : {5003, 200058}) {
    std::vector<float> base(size_t(rows) * V);
    for (int i = 0; i < rows; ++i)
      for (int x = 0; x < V; ++x) {
        double v = r.normal() * 4;
        if (i == 1) v = std::round(v * 2) / 2;  // heavy ties
        if (i == 2) v = -std::fabs(v);          // negative values (key order)
        if (i == 3) v = (x % (V / 3000 + 1) == 0) ? 50.0 : -std::fabs(v);  // ties spread over the row
        if (i == 4) v = 1.5;
        if (i == 5) v = x % 7 == 3 ? 20.0 : (x % 7 == 4 ? 30.0 : v);  // > 1024 at 30 / 20 for V large
        base[size_t(i) * V + x] = bfr(float(v));
      }
    DeviceArray<float> d_base(base), d_val(size_t(rows) * kd::CANDIDATES);
    DeviceArray<int32_t> d_idx(size_t(rows) * kd::CANDIDATES), d_mapped(size_t(rows) * kd::CANDIDATES);
    DeviceArray<uint8_t> ws(kd::topk_workspace_bytes(rows, V));
    std::vector<int32_t> map(V);  // element -> token id, as a shortlist head's rows
    for (int x = 0; x < V; ++x) map[x] = 7 * x + 3;
    DeviceArray<int32_t> d_map(map);
    kd::topk_candidates(d_base.get(), rows, V, nullptr, d_val.get(), d_idx.get(), ws.get(), nullptr);
    kd::topk_candidates(d_base.get(), rows, V, d_map.get(), d_val.get(), d_mapped.get(), ws.get(), nullptr);
    INGINE_TEST_CUDA(cudaDeviceSynchronize());
    const auto val = d_val.download();
    const auto idx = d_idx.download();
    const auto mapped = d_mapped.download();
    for (int i = 0; i < rows; ++i) {
      const auto ref = topk_ref(base.data() + size_t(i) * V, V);
      bool ok = int(ref.size()) == kd::CANDIDATES;
      for (int k = 0; ok && k < kd::CANDIDATES; ++k) {
        ok &= idx[size_t(i) * kd::CANDIDATES + k] == ref[k];
        ok &= mapped[size_t(i) * kd::CANDIDATES + k] == map[ref[k]];
        ok &= val[size_t(i) * kd::CANDIDATES + k] == base[size_t(i) * V + ref[k]];
      }
      char what[96];
      std::snprintf(what, sizeof(what), "topk_candidates V=%d %s (index order, ties by index, mapped)", V, names[i]);
      std::printf("%s %s\n", ok ? "PASS" : "FAIL", what);
      expect(ok, what);
    }
  }
}

// gather_head_rows copies the chosen rows (FP8 + scales, or BF16) and records their ids.
void test_shortlist_head(Rng& r) {
  const int V = 3000, n = 777;
  for (const bool fp8 : {true, false}) {
    const std::size_t row_bytes = fp8 ? std::size_t(H) : std::size_t(H) * 2;
    std::vector<uint8_t> src(std::size_t(V) * row_bytes);
    for (auto& b : src) b = uint8_t(r.uniform(0, 256));
    std::vector<float> scale(V);
    for (auto& s : scale) s = float(r.uniform(0.1, 2.0));
    std::vector<int32_t> ids(n);
    for (auto& t : ids) t = int32_t(r.uniform(0, V));
    DeviceArray<uint8_t> d_src(src), d_dst(std::size_t(n + 5) * row_bytes);
    DeviceArray<float> d_scale(scale), d_dscale(n + 5);
    DeviceArray<int32_t> d_ids(ids), d_map(n + 5);
    // rows from 5 on, as for context rows appended to a shortlist
    kd::gather_head_rows(d_src.get(), fp8 ? d_scale.get() : nullptr, row_bytes, d_ids.get(), n,
                         d_dst.get() + 5 * row_bytes, fp8 ? d_dscale.get() + 5 : nullptr, d_map.get() + 5, nullptr);
    INGINE_TEST_CUDA(cudaDeviceSynchronize());
    const auto dst = d_dst.download();
    const auto dscale = d_dscale.download();
    const auto map = d_map.download();
    bool ok = true;
    for (int i = 0; i < n; ++i) {
      ok &= std::equal(dst.begin() + std::ptrdiff_t((5 + i) * row_bytes), dst.begin() + std::ptrdiff_t((6 + i) * row_bytes),
                       src.begin() + std::ptrdiff_t(ids[i] * row_bytes));
      ok &= map[5 + i] == ids[i] && (!fp8 || dscale[5 + i] == scale[ids[i]]);
    }
    for (int i = 0; i < 5; ++i) ok &= map[i] == 0 && dscale[i] == 0.f;  // rows before untouched
    expect(ok, fp8 ? "gather_head_rows FP8 rows + scales" : "gather_head_rows BF16 rows");
  }
}

void test_markov(Rng& r) {
  const int V = 3000;
  auto rnn_w = rand_bf16(r, size_t(kd::RNN_OUT) * kd::RNN_IN, 0.02);
  auto rnn_b = rand_bf16(r, kd::RNN_OUT, 0.5);
  auto w1 = rand_bf16(r, size_t(V) * kd::RANK);
  auto fold_w = rand_bf16(r, size_t(V) * kd::RANK, 0.2);
  auto fold_b = rand_bf16(r, V);
  auto h = rand_bf16(r, size_t(kd::GAMMA) * H);
  auto f_rnn = rand_bf16(r, H);
  std::vector<float> base(size_t(kd::GAMMA) * V);
  for (auto& x : base) x = bfr(float(r.normal() * 5));
  const int anchor = 1234;

  DeviceArray<bf16> d_rnn_w(rnn_w), d_rnn_b(rnn_b), d_w1(w1), d_fw(fold_w), d_fb(fold_b), d_h(h), d_f(f_rnn);
  DeviceArray<bf16> d_state(size_t(kd::GAMMA + 1) * kd::RANK), d_latent(kd::RANK);
  DeviceArray<float> d_base(base), d_val(size_t(kd::GAMMA) * kd::CANDIDATES), d_part(size_t(kd::GAMMA) * kd::RNN_OUT);
  DeviceArray<float> d_logits(size_t(kd::GAMMA) * kd::CANDIDATES), d_conf(kd::GAMMA), d_qstats(2 * kd::GAMMA);
  DeviceArray<float> d_draft(std::vector<float>(size_t(kd::GAMMA) * V, -INFINITY));
  DeviceArray<int32_t> d_idx(size_t(kd::GAMMA) * kd::CANDIDATES), d_prev(size_t(kd::GAMMA) * kd::CANDIDATES),
      d_tokens(kd::GAMMA);
  DeviceArray<unsigned long long> d_picks(kd::GAMMA);
  DeviceArray<kd::DraftParams> p(1);
  DeviceArray<uint8_t> topk_ws(kd::topk_workspace_bytes(kd::GAMMA, V));
  kd::topk_candidates(d_base.get(), kd::GAMMA, V, nullptr, d_val.get(), d_idx.get(), topk_ws.get(), nullptr);
  kd::markov_pre(d_rnn_w.get(), d_h.get(), d_f.get(), d_part.get(), nullptr);
  INGINE_TEST_CUDA(cudaDeviceSynchronize());
  const auto cand_val = d_val.download();
  const auto cand_idx = d_idx.download();

  const auto wf = to_float(rnn_w), bfv = to_float(rnn_b), w1f = to_float(w1), fwf = to_float(fold_w),
             fbf = to_float(fold_b), hf = to_float(h), ff = to_float(f_rnn);
  // markov_pre against FP64
  {
    const auto part = d_part.download();
    std::vector<double> ref(part.size());
    for (int i = 0; i < kd::GAMMA; ++i)
      for (int row = 0; row < kd::RNN_OUT; ++row) {
        double s = 0;
        for (int c = 0; c < H; ++c) s += double(wf[size_t(row) * kd::RNN_IN + 512 + c]) * hf[size_t(i) * H + c];
        for (int c = 0; c < H; ++c) s += double(wf[size_t(row) * kd::RNN_IN + 512 + H + c]) * ff[c];
        ref[size_t(i) * kd::RNN_OUT + row] = s;
      }
    expect(report("markov_pre", compare(part.data(), ref.data(), ref.size()), 1e-5), "markov_pre");
  }

  for (const float T : {0.f, 0.7f}) {
    set_params(p.get(), anchor, 100, T, 0, 42);
    INGINE_TEST_CUDA(cudaMemset(d_state.get(), 0, sizeof(bf16) * (kd::GAMMA + 1) * kd::RANK));
    INGINE_TEST_CUDA(cudaMemset(d_picks.get(), 0, sizeof(unsigned long long) * kd::GAMMA));
    std::vector<float> state(kd::RANK, 0.f);
    int prev = anchor;
    std::vector<int> picks_host(kd::GAMMA);
    bool picks_ok = true;
    for (int i = 0; i < kd::GAMMA; ++i) {
      kd::markov_step(i, d_rnn_w.get(), d_rnn_b.get(), d_w1.get(), d_part.get(), d_picks.get(), d_idx.get(), p.get(),
                      d_state.get(), d_latent.get(), nullptr);
      INGINE_TEST_CUDA(cudaDeviceSynchronize());
      const auto part = d_part.download();
      std::vector<float> raw(kd::RNN_OUT);
      for (int row = 0; row < kd::RNN_OUT; ++row) {
        double s = part[size_t(i) * kd::RNN_OUT + row];
        for (int c = 0; c < kd::RANK; ++c) s += double(wf[size_t(row) * kd::RNN_IN + c]) * state[c];
        for (int c = 0; c < kd::RANK; ++c)
          s += double(wf[size_t(row) * kd::RNN_IN + kd::RANK + c]) * w1f[size_t(prev) * kd::RANK + c];
        raw[row] = bfr(bfr(float(s)) + bfv[row]);
      }
      std::vector<float> ref_state(kd::RANK), ref_latent(kd::RANK);
      for (int j = 0; j < kd::RANK; ++j) {
        const float gate = bfr(1.f / (1.f + std::exp(-raw[j])));
        const float cell = bfr(std::tanh(raw[kd::RANK + j]));
        ref_state[j] = bfr(bfr(gate * state[j]) + bfr(bfr(1.f - gate) * cell));
        ref_latent[j] = bfr(std::tanh(raw[2 * kd::RANK + j]));
      }
      const auto all_state = download_bf16(d_state);
      const std::vector<float> got_state(all_state.begin() + size_t(i + 1) * kd::RANK,
                                         all_state.begin() + size_t(i + 2) * kd::RANK);
      const auto got_latent = download_bf16(d_latent);
      char name[64];
      std::snprintf(name, sizeof(name), "markov_step %d state (T=%.1f)", i, T);
      check_close(got_state, ref_state, name, 2.0, 1e-2);
      std::snprintf(name, sizeof(name), "markov_step %d latent (T=%.1f)", i, T);
      check_close(got_latent, ref_latent, name, 2.0, 1e-2);

      kd::markov_logits(i, d_fw.get(), d_fb.get(), d_val.get(), d_idx.get(), d_latent.get(), p.get(), d_logits.get(),
                        d_picks.get(), nullptr);
      INGINE_TEST_CUDA(cudaDeviceSynchronize());
      const auto logits = d_logits.download();
      std::vector<float> ref_logits(kd::CANDIDATES), got_logits(kd::CANDIDATES);
      for (int k = 0; k < kd::CANDIDATES; ++k) {
        const int c = cand_idx[size_t(i) * kd::CANDIDATES + k];
        double dot = 0;
        for (int e = 0; e < kd::RANK; ++e) dot += double(fwf[size_t(c) * kd::RANK + e]) * got_latent[e];
        const float bias = bfr(fbf[c] + bfr(float(dot)));
        const float t = bfr(cand_val[size_t(i) * kd::CANDIDATES + k] + bias);
        ref_logits[k] = bfr(30.f * std::tanh(t / 30.f));
        got_logits[k] = logits[size_t(i) * kd::CANDIDATES + k];
      }
      std::snprintf(name, sizeof(name), "markov_logits %d (T=%.1f)", i, T);
      check_close(got_logits, ref_logits, name, 2.0, 1e-2);
      // the pick must be the argmax (first maximum) of the kernel's own scores
      const auto picks = d_picks.download();
      const int pick = int(0xffffffffu - uint32_t(picks[i] & 0xffffffffu));
      int best = 0;
      float best_s = -INFINITY;
      for (int k = 0; k < kd::CANDIDATES; ++k) {
        float s = got_logits[k];
        if (T > 0.f) s = got_logits[k] / std::max(T, 1e-6f) + kd::gumbel(42, 0, uint32_t(i), uint32_t(k));
        if (s > best_s) {
          best_s = s;
          best = k;
        }
      }
      picks_ok &= pick == best;
      picks_host[i] = pick;
      prev = cand_idx[size_t(i) * kd::CANDIDATES + pick];
      state = got_state;
    }
    std::printf("%s markov picks (T=%.1f)\n", picks_ok ? "PASS" : "FAIL", T);
    expect(picks_ok, "markov picks");

    // finish twice: the second call must clear the first call's candidates
    for (int round = 0; round < 2; ++round) {
      if (round == 1) {
        std::vector<int32_t> shifted(cand_idx.size());
        for (size_t k = 0; k < shifted.size(); ++k) shifted[k] = (cand_idx[k] + 7) % V;
        d_idx.upload(shifted);
      }
      kd::markov_finish(d_idx.get(), d_logits.get(), d_picks.get(), d_prev.get(), d_draft.get(), V, d_tokens.get(),
                        d_conf.get(), d_qstats.get(), p.get(), nullptr);
      INGINE_TEST_CUDA(cudaDeviceSynchronize());
    }
    const auto idx_now = d_idx.download();
    d_idx.upload(cand_idx);
    const auto logits = d_logits.download();
    const auto draft = d_draft.download();
    const auto tokens = d_tokens.download();
    const auto conf = d_conf.download();
    const auto qstats = d_qstats.download();
    bool ok = true;
    for (int i = 0; i < kd::GAMMA; ++i) {
      ok &= tokens[i] == idx_now[size_t(i) * kd::CANDIDATES + picks_host[i]];
      std::vector<float> expect_row(V, -INFINITY);
      double m = -INFINITY, s = 0;
      for (int k = 0; k < kd::CANDIDATES; ++k) {
        expect_row[idx_now[size_t(i) * kd::CANDIDATES + k]] = logits[size_t(i) * kd::CANDIDATES + k];
        m = std::max(m, double(logits[size_t(i) * kd::CANDIDATES + k]));
      }
      for (int k = 0; k < kd::CANDIDATES; ++k) s += std::exp(double(logits[size_t(i) * kd::CANDIDATES + k]) - m);
      ok &= std::equal(expect_row.begin(), expect_row.end(), draft.begin() + size_t(i) * V);
      ok &= std::fabs(conf[i] - 1.0 / s) <= 1e-5 * (1.0 / s);
      if (T > 0.f) {  // q = softmax(logits / T): row max and sum
        double qm = -INFINITY, qs = 0;
        for (int k = 0; k < kd::CANDIDATES; ++k) qm = std::max(qm, double(logits[size_t(i) * kd::CANDIDATES + k] / T));
        for (int k = 0; k < kd::CANDIDATES; ++k) qs += std::exp(double(logits[size_t(i) * kd::CANDIDATES + k] / T) - qm);
        ok &= qstats[2 * i] == float(qm) && std::fabs(qstats[2 * i + 1] - qs) <= 1e-5 * qs;
      }
    }
    const auto params = p.download();
    ok &= params[0].draws == 2;
    std::printf("%s markov_finish (T=%.1f)\n", ok ? "PASS" : "FAIL", T);
    expect(ok, "markov_finish");
  }
}

// The draft Gumbel noise: -log(-log(1 - v)) for v = k / 2^64 to float precision over the whole range (no cap
// on the noise from a coarse uniform), and its tail frequency.
void test_gumbel() {
  bool ok = true;
  float worst = 0.f;
  for (int sh = 0; sh < 64; ++sh)
    for (const std::uint64_t f : {1ull, 3ull, 0x9e3779b97f4a7c15ull}) {
      const std::uint64_t k = sh == 0 ? f : (f << sh) | (f >> (64 - sh));
      if (k == 0) continue;
      // below 2^24 the kernel takes the bin's midpoint k + 0.5
      const double v = std::ldexp(double(k) + (k < (1ull << 24) ? 0.5 : 0.0), -64);
      const double ref = -std::log(-std::log1p(-v));
      const float got = kd::gumbel_from_bits(k);
      // v keeps 24 significant bits: |dv / v| <= 2^-24, d noise = dv / ((1 - v) E) with E = -log(1 - v)
      const double tol = 1e-5 * std::max(1.0, std::fabs(ref)) + std::ldexp(v / ((1 - v) * -std::log1p(-v)), -23);
      worst = std::max(worst, float(std::fabs(got - ref)));
      ok &= std::isfinite(got) && std::fabs(got - ref) <= tol;
    }
  ok &= kd::gumbel_from_bits(1) > 43.f && kd::gumbel_from_bits(~0ull) < -2.7f;  // ends of the range
  // tail frequency over 2^22 draws: P(G > t) = 1 - exp(-e^-t); counts within 5 sigma
  const int n = 1 << 22;
  const double ts[] = {0.0, 3.0, 8.0, 12.0};
  long cnt[4] = {0, 0, 0, 0};
  for (int j = 0; j < n; ++j) {
    const float g = kd::gumbel(7, 3, uint32_t(j), 11u);
    for (int t = 0; t < 4; ++t) cnt[t] += g > ts[t];
  }
  for (int t = 0; t < 4; ++t) {
    const double p = -std::expm1(-std::exp(-ts[t])), mean = n * p, sd = std::sqrt(n * p * (1 - p));
    ok &= std::fabs(cnt[t] - mean) <= 5 * sd + 1;
  }
  std::printf("%s draft gumbel noise exact to the tail (worst |diff| %.3g, %ld above 12 of %.1f)\n", ok ? "PASS" : "FAIL",
              double(worst), cnt[3], n * -std::expm1(-std::exp(-12.0)));
  expect(ok, "draft gumbel noise");
}

void test_fold(Rng& r) {
  const int rows = 100;
  auto hw = rand_bf16(r, size_t(H) * kd::RANK, 0.05);
  auto hb = rand_bf16(r, H, 0.05);
  const auto hwf = to_float(hw), hbf = to_float(hb);
  for (const bool fp8 : {false, true}) {
    std::vector<float> u(size_t(rows) * H);
    std::vector<uint8_t> codes(size_t(rows) * H);
    std::vector<float> scale(rows);
    std::vector<bf16> ubf;
    if (fp8) {
      for (int i = 0; i < rows; ++i) scale[i] = float(r.uniform(0.001, 0.01));
      for (size_t i = 0; i < codes.size(); ++i) {
        uint8_t c;
        do c = uint8_t(r.next() & 0xff);
        while ((c & 0x7f) == 0x7f);
        codes[i] = c;
        u[i] = bfr(e4m3_to_float(c) * scale[i / H]);
      }
    } else {
      ubf = rand_bf16(r, size_t(rows) * H);
      u = to_float(ubf);
    }
    DeviceArray<uint8_t> d_codes(codes);
    DeviceArray<bf16> d_ubf(fp8 ? std::vector<bf16>(1) : ubf), d_hw(hw), d_hb(hb), d_fw(size_t(rows) * kd::RANK),
        d_fb(rows);
    DeviceArray<float> d_scale(scale);
    kd::fold_head(fp8 ? static_cast<const void*>(d_codes.get()) : d_ubf.get(), fp8 ? d_scale.get() : nullptr, rows,
                  d_hw.get(), d_hb.get(), d_fw.get(), d_fb.get(), nullptr);
    INGINE_TEST_CUDA(cudaDeviceSynchronize());
    std::vector<float> ref_w(size_t(rows) * kd::RANK), ref_b(rows);
    for (int i = 0; i < rows; ++i) {
      double sb = 0;
      for (int k = 0; k < H; ++k) sb += double(u[size_t(i) * H + k]) * hbf[k];
      ref_b[i] = bfr(float(sb));
      for (int j = 0; j < kd::RANK; ++j) {
        double s = 0;
        for (int k = 0; k < H; ++k) s += double(u[size_t(i) * H + k]) * hwf[size_t(k) * kd::RANK + j];
        ref_w[size_t(i) * kd::RANK + j] = bfr(float(s));
      }
    }
    check_close(download_bf16(d_fw), ref_w, fp8 ? "fold_w (fp8 rows)" : "fold_w (bf16 rows)", 1.0, 1e-30);
    check_close(download_bf16(d_fb), ref_b, fp8 ? "fold_b (fp8 rows)" : "fold_b (bf16 rows)", 1.0, 1e-30);
  }
}

void test_greedy_accept() {
  struct Case {
    std::vector<int32_t> drafts, target;
    int acc, next;
  };
  const Case cases[] = {{{5, 6, 7}, {5, 6, 7, 8}, 3, 8},
                        {{5, 6, 7}, {5, 9, 7, 8}, 1, 9},
                        {{5, 6, 7}, {4, 6, 7, 8}, 0, 4},
                        {{1}, {1, 2}, 1, 2},
                        {{1, 2, 3, 4, 5, 6, 7}, {1, 2, 3, 4, 5, 6, 0, 9}, 6, 0}};
  DeviceArray<kd::AcceptResult> out(1);
  bool ok = true;
  for (const Case& c : cases) {
    DeviceArray<int32_t> d(c.drafts), t(c.target);
    kd::greedy_accept(t.get(), d.get(), int(c.drafts.size()), out.get(), nullptr);
    INGINE_TEST_CUDA(cudaDeviceSynchronize());
    const auto res = out.download()[0];
    ok &= res.accepted == c.acc && res.next == c.next;
  }
  std::printf("%s greedy_accept (longest prefix equal to the target argmax)\n", ok ? "PASS" : "FAIL");
  expect(ok, "greedy_accept");
}

}  // namespace

int main() {
  if (skip_without_gpu()) return kSkip;
  Rng r(20261120);
  test_mix_rmsnorm(r);
  test_ctx_kv_store(r);
  test_small_commit(r);
  test_embed_block(r);
  test_qkv_rope(r);
  test_block_attention(r);
  test_add_rmsnorm_silu(r);
  test_topk(r);
  test_shortlist_head(r);
  test_markov(r);
  test_gumbel();
  test_fold(r);
  test_greedy_accept();
  std::printf("%s: %d failure(s)\n", g_failures ? "FAILED" : "OK", g_failures);
  return g_failures ? 1 : 0;
}
