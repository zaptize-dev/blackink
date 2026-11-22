// Prefill row kernels against FP64 oracles of the model.py operations: attn_prep (sliding static conv,
// full dynamic conv with log scaling, BF16 and FP8 key/value outputs, conv state before the chunk),
// rel_logits, static_conv + conv_state_update (T < 3 and T > 3), message gates, head logits (BF16 and FP8
// rows), the cuBLASLt GEMM wrapper, taps.
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

constexpr int H = 2304, HQ = 24, HK = 8, D = 128, KV = 1024;
constexpr int OFF_Q = 0, OFF_REL = 6144, OFF_K = 6552, OFF_V = 7576, OFF_KDYN = 8600, OFF_VDYN = 8632;

std::vector<uint16_t> rand_bf16(Rng& rng, size_t n, double scale) {
  std::vector<uint16_t> v(n);
  for (auto& x : v) x = bf16_bits(float(rng.normal() * scale));
  return v;
}
std::vector<float> rand_f32(Rng& rng, size_t n, double scale, double offset = 0) {
  std::vector<float> v(n);
  for (auto& x : v) x = float(offset + rng.normal() * scale);
  return v;
}

double clipf(double sumsq, double thr) { return std::min(thr / (std::sqrt(sumsq) + 1e-6), 1.0); }

bool test_attn_prep(Rng& rng, bool full, bool fp8, int T, int pos0) {
  const int ld = full ? 8664 : 8600;
  const auto proj = rand_bf16(rng, size_t(T) * ld, 1.0);
  const auto q_scale = rand_f32(rng, HQ * D, 0.3, 1.0), r_scale = rand_f32(rng, HQ * 16, 0.3, 1.0);
  const auto kw = rand_f32(rng, 4 * KV, 0.4), vw = rand_f32(rng, 4 * KV, 0.4);
  const auto conv_k = rand_f32(rng, 3 * KV, 1.0), conv_v = rand_f32(rng, 3 * KV, 1.0);
  DeviceArray<uint16_t> d_proj(proj), q(size_t(T) * HQ * D), rel(size_t(T) * HQ * 16);
  DeviceArray<float> d_qs(q_scale), d_rs(r_scale), d_kw(kw), d_vw(vw), d_ck(conv_k), d_cv(conv_v);
  DeviceArray<uint8_t> kout(size_t(T) * KV * 2), vout(size_t(T) * KV * 2);
  pk::AttnPrepParams p;
  p.proj = reinterpret_cast<const __nv_bfloat16*>(d_proj.get()), p.ld = ld, p.T = T, p.pos0 = pos0, p.full = full;
  p.q_scale = d_qs.get(), p.r_scale = d_rs.get(), p.k_w = d_kw.get(), p.v_w = d_vw.get();
  p.conv_k = d_ck.get(), p.conv_v = d_cv.get();
  p.q = reinterpret_cast<__nv_bfloat16*>(q.get()), p.rel = reinterpret_cast<__nv_bfloat16*>(rel.get());
  p.k_out = kout.get(), p.v_out = vout.get(), p.out_fp8 = fp8;
  pk::attn_prep(p, nullptr);
  INGINE_TEST_CUDA(cudaDeviceSynchronize());
  const auto gq = q.download(), grel = rel.download();
  const auto gk = kout.download(), gv = vout.download();
  auto P = [&](int t, int c) { return double(bf16_val(proj[size_t(t) * ld + c])); };

  std::vector<float> got_q, got_r, got_k, got_v;
  std::vector<double> ref_q, ref_r, ref_k, ref_v;
  for (int t = 0; t < T; ++t) {
    const int pos = pos0 + t;
    const double tau = 1 + 0.1 * std::log(std::max((pos + 1) / 8192.0, 1.0));
    for (int h = 0; h < HQ; ++h) {
      double x[D], ss = 0;
      for (int d = 0; d < D; ++d) x[d] = P(t, OFF_Q + h * D + d) * q_scale[h * D + d], ss += x[d] * x[d];
      const double f = clipf(ss, 22.627416997969522);
      for (int d = 0; d < D; ++d) {
        double y = rbf(x[d] * f);
        if (full) y = rbf(y * tau);
        ref_q.push_back(y), got_q.push_back(bf16_val(gq[(size_t(t) * HQ + h) * D + d]));
      }
      double r[16], rs = 0;
      for (int j = 0; j < 16; ++j) r[j] = P(t, OFF_REL + h * 16 + j) * r_scale[h * 16 + j], rs += r[j] * r[j];
      const double fr = clipf(rs, 8.0);
      for (int j = 0; j < 16; ++j) {
        double y = rbf(r[j] * fr);
        if (full) y = rbf(y * tau);
        ref_r.push_back(y), got_r.push_back(bf16_val(grel[(size_t(t) * HQ + h) * 16 + j]));
      }
    }
    for (int kv = 0; kv < 2; ++kv) {
      const int off = kv ? OFF_V : OFF_K;
      const auto& st = kv ? conv_v : conv_k;
      const auto& w = kv ? vw : kw;
      auto in = [&](int s, int c) { return s >= 0 ? P(s, off + c) : double(st[(3 + s) * KV + c]); };
      for (int g = 0; g < HK; ++g) {
        double y[D], ss = 0;
        for (int d = 0; d < D; ++d) {
          const int c = g * D + d;
          double acc = 0;
          if (full) {
            for (int lag = 0; lag < 4; ++lag)
              acc += (double(w[lag * KV + c]) + P(t, (kv ? OFF_VDYN : OFF_KDYN) + g * 4 + lag)) * in(t - lag, c);
            y[d] = rbf((acc + in(t, c)) / 3.0);
          } else {
            for (int j = 0; j < 4; ++j) acc += double(w[c * 4 + j]) * in(t - 3 + j, c);
            y[d] = rbf((acc + in(t, c)) / std::sqrt(5.0));
          }
          ss += y[d] * y[d];
        }
        const double f = kv ? 1.0 / std::sqrt(ss / D + 1e-6) : clipf(ss, 22.627416997969522);
        for (int d = 0; d < D; ++d) {
          const size_t i = size_t(t) * KV + g * D + d;
          const double r = rbf(y[d] * f);
          const auto& gout = kv ? gv : gk;
          float got;
          double ref = r;
          if (fp8) {
            got = e4m3_to_float(gout[i]);
            ref = e4m3_to_float(e4m3_bits(float(r)));
          } else {
            got = bf16_val(uint16_t(gout[2 * i] | (gout[2 * i + 1] << 8)));
          }
          (kv ? got_v : got_k).push_back(got), (kv ? ref_v : ref_k).push_back(ref);
        }
      }
    }
  }
  char name[64];
  std::snprintf(name, sizeof name, "attn_prep %s%s T=%d", full ? "full" : "sliding", fp8 ? " fp8" : "", T);
  bool ok = true;
  const double tol = fp8 ? 0.07 : 8e-3;  // one ulp of the output format (a rounding flip)
  std::string n = name;
  ok &= check_close((n + " q").c_str(), got_q, ref_q, 8e-3, 1e-6);
  ok &= check_close((n + " rel").c_str(), got_r, ref_r, 8e-3, 1e-6);
  ok &= check_close((n + " k").c_str(), got_k, ref_k, tol, 1e-3);
  ok &= check_close((n + " v").c_str(), got_v, ref_v, tol, 1e-3);
  return ok;
}

bool test_rel_logits(Rng& rng, int T) {
  const auto rel = rand_bf16(rng, size_t(T) * HQ * 16, 1.0);
  const auto proj = rand_f32(rng, 16 * 1024, 0.5);
  DeviceArray<uint16_t> d_rel(rel), out(size_t(T) * HQ * 1024);
  DeviceArray<float> d_proj(proj);
  pk::rel_logits(reinterpret_cast<const __nv_bfloat16*>(d_rel.get()), d_proj.get(), T,
                 reinterpret_cast<__nv_bfloat16*>(out.get()), nullptr);
  INGINE_TEST_CUDA(cudaDeviceSynchronize());
  const auto g = out.download();
  std::vector<float> got;
  std::vector<double> ref;
  for (int r = 0; r < T * HQ; ++r)
    for (int d = 0; d < 1024; ++d) {
      double acc = 0;
      for (int j = 0; j < 16; ++j) acc += double(bf16_val(rel[size_t(r) * 16 + j])) * rbf(proj[j * 1024 + d]);
      ref.push_back(acc), got.push_back(bf16_val(g[size_t(r) * 1024 + d]));
    }
  return check_close("rel_logits", got, ref, 8e-3, 1e-5);
}

bool test_conv(Rng& rng, int T) {
  const int C = H, ld = H + 64;
  const auto x = rand_bf16(rng, size_t(T) * ld, 1.0);
  const auto w = rand_f32(rng, size_t(C) * 4, 0.5);
  const auto state = rand_f32(rng, 3 * C, 1.0);
  DeviceArray<uint16_t> d_x(x);
  DeviceArray<float> d_w(w), d_state(state), out(size_t(T) * C);
  pk::static_conv(reinterpret_cast<const __nv_bfloat16*>(d_x.get()), ld, T, C, d_w.get(), d_state.get(), out.get(),
                  nullptr);
  pk::conv_state_update(d_state.get(), reinterpret_cast<const __nv_bfloat16*>(d_x.get()), ld, T, C, nullptr);
  INGINE_TEST_CUDA(cudaDeviceSynchronize());
  auto in = [&](int s, int c) { return s >= 0 ? double(bf16_val(x[size_t(s) * ld + c])) : double(state[(3 + s) * C + c]); };
  const auto g = out.download();
  std::vector<float> got(g.begin(), g.end());
  std::vector<double> ref;
  for (int t = 0; t < T; ++t)
    for (int c = 0; c < C; ++c) {
      double acc = 0;
      for (int j = 0; j < 4; ++j) acc += double(w[c * 4 + j]) * in(t - 3 + j, c);
      ref.push_back((acc + in(t, c)) / std::sqrt(5.0));
    }
  char name[48];
  std::snprintf(name, sizeof name, "static_conv T=%d", T);
  bool ok = check_close(name, got, ref, 1e-5, 1e-6);
  const auto gs = d_state.download();
  std::vector<float> got_s(gs.begin(), gs.end());
  std::vector<double> ref_s;
  for (int i = 0; i < 3; ++i)
    for (int c = 0; c < C; ++c) ref_s.push_back(in(T - 3 + i, c));
  std::snprintf(name, sizeof name, "conv_state_update T=%d", T);
  ok &= check_close(name, got_s, ref_s, 0, 0);
  return ok;
}

bool test_gates(Rng& rng, int T) {
  const int G = 3;
  std::vector<std::vector<float>> h(G), m(G);
  for (int g = 0; g < G; ++g) h[g] = rand_f32(rng, size_t(T) * H, 2.0), m[g] = rand_f32(rng, size_t(T) * H, 0.5);
  std::vector<DeviceArray<float>> dh, dm;
  pk::GateRows rows;
  for (int g = 0; g < G; ++g) {
    dh.emplace_back(h[g]), dm.emplace_back(m[g]);
    rows.h[g] = dh.back().get(), rows.msg[g] = dm.back().get();
  }
  DeviceArray<uint16_t> gin(size_t(G) * T * 2 * H);
  pk::gate_inputs(rows, G, T, reinterpret_cast<__nv_bfloat16*>(gin.get()), nullptr);
  const auto f = rand_f32(rng, size_t(T) * H, 1.0), gn = rand_f32(rng, size_t(T) * H, 1.0);
  const auto nw = rand_f32(rng, H, 0.2, 1.0);
  DeviceArray<float> df(f), dg(gn), dnw(nw);
  DeviceArray<uint16_t> out(size_t(T) * H);
  pk::gate_apply(dh[1].get(), dm[1].get(), df.get(), dg.get(), dnw.get(), T, reinterpret_cast<__nv_bfloat16*>(out.get()),
                 nullptr);
  INGINE_TEST_CUDA(cudaDeviceSynchronize());
  const auto gi = gin.download(), go = out.download();
  std::vector<float> got_i, got_o;
  std::vector<double> ref_i, ref_o;
  for (int g = 0; g < G; ++g)
    for (int t = 0; t < T; ++t) {
      double sh = 0, sm = 0;
      for (int c = 0; c < H; ++c) sh += double(h[g][size_t(t) * H + c]) * h[g][size_t(t) * H + c],
                                  sm += double(m[g][size_t(t) * H + c]) * m[g][size_t(t) * H + c];
      const double rh = 1 / std::sqrt(sh / H + 1e-6), rm = 1 / std::sqrt(sm / H + 1e-6);
      for (int c = 0; c < H; ++c) {
        ref_i.push_back(h[g][size_t(t) * H + c] * rh), got_i.push_back(bf16_val(gi[(size_t(g) * T + t) * 2 * H + c]));
        ref_i.push_back(m[g][size_t(t) * H + c] * rm), got_i.push_back(bf16_val(gi[(size_t(g) * T + t) * 2 * H + H + c]));
      }
    }
  for (int t = 0; t < T; ++t) {
    std::vector<double> y(H);
    double ss = 0;
    for (int c = 0; c < H; ++c) {
      const size_t i = size_t(t) * H + c;
      y[c] = (1 - std::tanh(f[i])) * h[1][i] + std::tanh(gn[i]) * m[1][i];
      ss += y[c] * y[c];
    }
    const double r = 1 / std::sqrt(ss / H + 1e-6);
    for (int c = 0; c < H; ++c) ref_o.push_back(y[c] * r * nw[c]), got_o.push_back(bf16_val(go[size_t(t) * H + c]));
  }
  bool ok = check_close("gate_inputs", got_i, ref_i, 4e-3, 1e-6);
  ok &= check_close("gate_apply", got_o, ref_o, 4e-3, 1e-6);
  return ok;
}

bool test_head(Rng& rng, bool fp8) {
  const int V = 3001;
  const auto x = rand_bf16(rng, H, 1.0);
  std::vector<double> wv(size_t(V) * H);
  std::vector<uint16_t> wb;
  std::vector<uint8_t> w8;
  std::vector<float> scale;
  if (fp8) {
    w8.resize(wv.size());
    scale.resize(V);
    for (int v = 0; v < V; ++v) {
      scale[v] = float(rng.uniform(0.001, 0.003));
      for (int k = 0; k < H; ++k) {
        w8[size_t(v) * H + k] = e4m3_bits(float(rng.normal() * 20));
        wv[size_t(v) * H + k] = rbf(e4m3_to_float(w8[size_t(v) * H + k]) * scale[v]);
      }
    }
  } else {
    wb = rand_bf16(rng, wv.size(), 0.04);
    for (size_t i = 0; i < wv.size(); ++i) wv[i] = bf16_val(wb[i]);
  }
  DeviceArray<uint16_t> dx(x), dwb(fp8 ? std::vector<uint16_t>(1) : wb);
  DeviceArray<uint8_t> dw8(fp8 ? w8 : std::vector<uint8_t>(1));
  DeviceArray<float> ds(fp8 ? scale : std::vector<float>(1)), out(V);
  pk::head_logits(reinterpret_cast<const __nv_bfloat16*>(dx.get()), fp8 ? (const void*)dw8.get() : (const void*)dwb.get(),
                  fp8 ? ds.get() : nullptr, V, out.get(), nullptr);
  INGINE_TEST_CUDA(cudaDeviceSynchronize());
  const auto g = out.download();
  std::vector<float> got(g.begin(), g.end());
  std::vector<double> ref;
  for (int v = 0; v < V; ++v) {
    double acc = 0;
    for (int k = 0; k < H; ++k) acc += wv[size_t(v) * H + k] * bf16_val(x[k]);
    ref.push_back(std::tanh(rbf(acc) / 30) * 30);
  }
  return check_close(fp8 ? "head fp8" : "head bf16", got, ref, 8e-3, 1e-4);
}

bool test_gemm(Rng& rng) {
  const int M = 37, N = 200, K = 2304, lda = K + 8, ldc = N + 24;
  const auto A = rand_bf16(rng, size_t(M) * lda, 1.0), B = rand_bf16(rng, size_t(N) * K, 0.05);
  DeviceArray<uint16_t> dA(A), dB(B), dC(size_t(M) * ldc);
  DeviceArray<float> dF(size_t(M) * N);
  pk::Gemm gemm;
  gemm.run(reinterpret_cast<const __nv_bfloat16*>(dA.get()), lda, reinterpret_cast<const __nv_bfloat16*>(dB.get()), K,
           dC.get(), ldc, false, M, N, K, nullptr);
  gemm.run(reinterpret_cast<const __nv_bfloat16*>(dA.get()), lda, reinterpret_cast<const __nv_bfloat16*>(dB.get()), K,
           dF.get(), N, true, M, N, K, nullptr);
  INGINE_TEST_CUDA(cudaDeviceSynchronize());
  const auto gc = dC.download();
  const auto gf = dF.download();
  std::vector<float> got_b, got_f;
  std::vector<double> ref_b, ref_f;
  for (int m = 0; m < M; ++m)
    for (int n = 0; n < N; ++n) {
      double acc = 0;
      for (int k = 0; k < K; ++k) acc += double(bf16_val(A[size_t(m) * lda + k])) * bf16_val(B[size_t(n) * K + k]);
      ref_b.push_back(acc), got_b.push_back(bf16_val(gc[size_t(m) * ldc + n]));
      ref_f.push_back(acc), got_f.push_back(gf[size_t(m) * N + n]);
    }
  bool ok = check_close("gemm bf16 out", got_b, ref_b, 4e-3, 1e-5);
  ok &= check_close("gemm fp32 out", got_f, ref_f, 1e-5, 1e-5);
  return ok;
}

bool test_taps(Rng& rng) {
  const int T = 5;
  const auto x = rand_f32(rng, size_t(T) * H, 1.0);
  DeviceArray<float> dx(x);
  DeviceArray<uint16_t> taps(size_t(T) * 4 * H);
  pk::store_tap(dx.get(), T, reinterpret_cast<__nv_bfloat16*>(taps.get()) + 2 * H, 4 * H, nullptr);
  INGINE_TEST_CUDA(cudaDeviceSynchronize());
  const auto g = taps.download();
  std::vector<float> got;
  std::vector<double> ref;
  for (int t = 0; t < T; ++t)
    for (int c = 0; c < H; ++c) got.push_back(bf16_val(g[(size_t(t) * 4 + 2) * H + c])), ref.push_back(rbf(x[size_t(t) * H + c]));
  return check_close("store_tap", got, ref, 0, 0);
}

}  // namespace

int main() {
  if (skip_without_gpu()) return kSkip;
  Rng rng(99);
  bool ok = true;
  ok &= test_attn_prep(rng, false, false, 9, 0);
  ok &= test_attn_prep(rng, false, false, 2, 37);
  ok &= test_attn_prep(rng, true, false, 7, 9000);
  ok &= test_attn_prep(rng, true, true, 7, 20000);
  ok &= test_rel_logits(rng, 21);
  ok &= test_rel_logits(rng, 300);
  ok &= test_conv(rng, 2);
  ok &= test_conv(rng, 37);
  ok &= test_gates(rng, 5);
  ok &= test_head(rng, false);
  ok &= test_head(rng, true);
  ok &= test_gemm(rng);
  ok &= test_taps(rng);
  std::printf("%s\n", ok ? "ALL PASS" : "SOME FAILED");
  return ok ? 0 : 1;
}
