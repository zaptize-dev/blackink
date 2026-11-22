// Shared host helpers of the prefill kernel tests: BF16 / E4M3 bit conversion and tolerance checks.
#pragma once

#include "common/test_util.h"

#include <cmath>
#include <cstdint>
#include <cstring>
#include <vector>

namespace ingine::test::prefill {

inline uint16_t bf16_bits(float f) {
  const __nv_bfloat16 b = float_to_bf16(f);
  uint16_t u;
  std::memcpy(&u, &b, 2);
  return u;
}

inline float bf16_val(uint16_t u) {
  const uint32_t w = uint32_t(u) << 16;
  float f;
  std::memcpy(&f, &w, 4);
  return f;
}

inline float rbf(double x) { return bf16_val(bf16_bits(float(x))); }

// float -> E4M3 code: saturate to +-448, nearest value, ties to the even code.
inline uint8_t e4m3_bits(float x) {
  static const std::vector<float> table = [] {
    std::vector<float> t(128);
    for (int c = 0; c < 128; ++c) t[c] = c == 0x7f ? INFINITY : e4m3_to_float(uint8_t(c));
    return t;
  }();
  const float a = std::fmin(std::fabs(x), 448.f);
  int best = 0;
  for (int c = 1; c < 127; ++c) {
    const float d = std::fabs(table[c] - a), db = std::fabs(table[best] - a);
    if (d < db || (d == db && (c & 1) == 0)) best = c;
  }
  return uint8_t(best | (std::signbit(x) && a > 0 ? 0x80 : 0));
}

// Elementwise check: |got - ref| <= rel * |ref| + abs[i] (abs: e.g. the FP32 accumulation error allowance of a
// cancelling sum).
inline bool check_close(const char* what, const std::vector<float>& got, const std::vector<double>& ref, double rel,
                        const std::vector<double>& abs, size_t max_report = 5) {
  size_t bad = 0, worst = 0;
  double worst_ratio = 0;
  for (size_t i = 0; i < got.size(); ++i) {
    const double err = std::fabs(double(got[i]) - ref[i]);
    const double lim = rel * std::fabs(ref[i]) + abs[i];
    if (!(err <= lim)) {
      if (bad < max_report)
        std::printf("  %s[%zu]: got %.6g ref %.6g (allowed %.3g)\n", what, i, double(got[i]), ref[i], lim);
      ++bad;
    }
    if (lim > 0 && err / lim > worst_ratio) worst_ratio = err / lim, worst = i;
  }
  std::printf("%s %s: %zu / %zu outside tolerance, worst err/limit %.3f at %zu\n", bad ? "FAIL" : "PASS", what, bad,
              got.size(), worst_ratio, worst);
  return bad == 0;
}

inline bool check_close(const char* what, const std::vector<float>& got, const std::vector<double>& ref, double rel,
                        double abs, size_t max_report = 5) {
  return check_close(what, got, ref, rel, std::vector<double>(got.size(), abs), max_report);
}

}  // namespace ingine::test::prefill
