// Shared host helpers of the kernel tests: FP8/FP4 encodings, BF16 rounding of doubles, random data and a
// tolerance check against FP64 oracles that emulate the reference's BF16 rounding points.
#pragma once

#include "common/test_util.h"

#include <cuda_bf16.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <optional>
#include <vector>

namespace ingine::test {

inline constexpr float kE2M1[16] = {0.f, .5f, 1.f, 1.5f, 2.f, 3.f, 4.f, 6.f, -0.f, -.5f, -1.f, -1.5f, -2.f, -3.f, -4.f, -6.f};

inline double rbf(double x) { return round_bf16(float(x)); }  // to FP32, then BF16 (nearest even)

inline std::vector<__nv_bfloat16> to_bf16(const std::vector<float>& v) {
  std::vector<__nv_bfloat16> o(v.size());
  for (size_t i = 0; i < v.size(); ++i) o[i] = float_to_bf16(v[i]);
  return o;
}

// float -> E4M3FN, nearest even, saturated to +-448 (torch x.clamp(-448, 448).to(float8_e4m3fn)).
// Codes 0..126 are monotonic in value (127 is NaN).
inline uint8_t float_to_e4m3(float x) {
  static const std::vector<float> table = [] {
    std::vector<float> t(127);
    for (int c = 0; c < 127; ++c) t[c] = e4m3_to_float(uint8_t(c));
    return t;
  }();
  const uint8_t sign = std::signbit(x) ? 0x80 : 0;
  const float a = std::min(std::fabs(x), 448.f);
  const int hi = int(std::lower_bound(table.begin(), table.end(), a) - table.begin());  // first >= a
  int best = std::min(hi, 126);
  if (hi > 0 && hi <= 126) {
    const float dl = a - table[hi - 1], dh = table[hi] - a;
    if (dl < dh || (dl == dh && ((hi - 1) & 1) == 0)) best = hi - 1;
  }
  return uint8_t(sign | best);
}

inline std::vector<float> randn(Rng& r, size_t n, double scale) {
  std::vector<float> v(n);
  for (auto& x : v) x = float(r.normal() * scale);
  return v;
}

// BF16-valued random floats.
inline std::vector<float> randn_bf16(Rng& r, size_t n, double scale) {
  std::vector<float> v = randn(r, n, scale);
  for (auto& x : v) x = round_bf16(x);
  return v;
}

// Values close to an oracle: |got - ref| <= rel * |ref| + abs for all, and at most `frac` of the values off
// the oracle by more than its last-bit rounding (got != rounded ref). Prints a line; returns pass.
struct Check {
  const char* what;
  double rel, abs, frac;
  bool operator()(const std::vector<float>& got, const std::vector<double>& ref,
                  const std::vector<double>* rounded = nullptr) const {
    double worst = 0;
    size_t bad = 0, off = 0, wi = 0;
    for (size_t i = 0; i < got.size(); ++i) {
      const double d = std::fabs(double(got[i]) - ref[i]);
      const double tol = rel * std::fabs(ref[i]) + abs;
      if (!(d <= tol)) ++bad;
      if (d / (tol > 0 ? tol : 1) > worst) worst = d / (tol > 0 ? tol : 1), wi = i;
      if (rounded && double(got[i]) != (*rounded)[i]) ++off;
    }
    const double f = got.empty() ? 0 : double(off) / double(got.size());
    const bool ok = bad == 0 && f <= frac;
    std::printf("%s %s: n=%zu worst err/tol=%.3f at %zu (got %.6g ref %.6g) out-of-tol=%zu off-rounding=%.4f\n",
                ok ? "PASS" : "FAIL", what, got.size(), worst, wi, got.empty() ? 0.0 : double(got[wi]),
                got.empty() ? 0.0 : ref[wi], bad, f);
    return ok;
  }
};

inline std::vector<float> bf16_download(const DeviceArray<__nv_bfloat16>& d) {
  const auto h = d.download();
  std::vector<float> o(h.size());
  for (size_t i = 0; i < h.size(); ++i) o[i] = bf16_to_float(h[i]);
  return o;
}

}  // namespace ingine::test
