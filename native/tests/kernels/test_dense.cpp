// Dense decode kernels (src/kernels/dense) against FP64 oracles of KERNELS.md area `dense`, at the model's
// shapes with synthetic weights (adapted from the PyTorch engine's test_dense.py: proj BF16/FP8 for several M and
// both projection layouts, proj_conv with its state update and history, head BF16/FP8, router with FP32 and
// BF16 weights, mix over its block/query/optional-tensor variants, message gates G = 1/3 with FP32/BF16
// forget/gain weights, embed, fp8_rows and dequant_rows).
#include "kernels/oracle.h"

#include "kernels/dense/dense.h"

#include <cuda_runtime.h>

#include <array>
#include <numeric>

using namespace ingine::test;
namespace dense = ingine::kernels::dense;

namespace {

constexpr int H = 2304;
cudaStream_t g_stream = nullptr;

void sync() { INGINE_TEST_CUDA(cudaStreamSynchronize(g_stream)); }

// A random weight matrix [rows, K] as the kernels take it: BF16 values, or FP8 codes + row scales.
struct Weights {
  int rows = 0, K = 0;
  bool fp8 = false;
  std::vector<double> value;  // effective value per element (code * scale applied at the end, see dot())
  std::vector<float> scale;   // fp8: per-row scale
  std::vector<double> code;   // fp8: decoded E4M3 value; bf16: the value
  std::optional<DeviceArray<uint8_t>> dev;
  std::optional<DeviceArray<float>> dev_scale;
  dense::Matrix matrix() const { return {dev->get(), fp8 ? dev_scale->get() : nullptr, rows}; }
  double dot(int r, const float* x) const {
    double s = 0;
    const double* c = &code[size_t(r) * K];
    for (int k = 0; k < K; ++k) s += c[k] * double(x[k]);
    return fp8 ? s * double(scale[r]) : s;
  }
};

Weights make_weights(Rng& rng, int rows, int K, bool fp8, double sd) {
  Weights w;
  w.rows = rows, w.K = K, w.fp8 = fp8;
  w.code.resize(size_t(rows) * K);
  if (fp8) {
    std::vector<uint8_t> codes(size_t(rows) * K);
    w.scale.resize(rows);
    for (int r = 0; r < rows; ++r) {
      w.scale[r] = float(sd / 64.0 * rng.uniform(0.5, 1.5));
      for (int k = 0; k < K; ++k) {
        const uint8_t c = float_to_e4m3(float(rng.normal() * 64.0));
        codes[size_t(r) * K + k] = c;
        w.code[size_t(r) * K + k] = e4m3_to_float(c);
      }
    }
    w.dev.emplace(codes);
    w.dev_scale.emplace(w.scale);
  } else {
    const std::vector<float> v = randn_bf16(rng, size_t(rows) * K, sd);
    for (size_t i = 0; i < v.size(); ++i) w.code[i] = v[i];
    const auto b = to_bf16(v);
    std::vector<uint8_t> bytes(b.size() * 2);
    std::memcpy(bytes.data(), b.data(), bytes.size());
    w.dev.emplace(bytes);
  }
  return w;
}

std::vector<int> layout(bool full, bool single) {
  const int rows[8] = {3072, 3072, 384, 24, 1024, 1024, 32, 32};
  const int input[8] = {0, 0, 0, 0, 1, 2, 1, 2};
  std::vector<int> t;
  int b = 0;
  for (int s = 0; s < (full ? 8 : 6); ++s) {
    t.insert(t.end(), {b, b + rows[s], single ? 0 : input[s]});
    b += rows[s];
  }
  return t;
}

bool test_proj(Rng& rng, int M, bool fp8, bool full, bool single) {
  const std::vector<int> seg = layout(full, single);
  const int nseg = int(seg.size() / 3), R = seg[seg.size() - 2];
  const Weights w = make_weights(rng, R, H, fp8, 0.03);
  const std::vector<float> x = randn_bf16(rng, size_t(3) * M * H, 1.0);
  DeviceArray<__nv_bfloat16> dx(to_bf16(x));
  DeviceArray<int> dseg(seg);
  DeviceArray<float> out(size_t(M) * R);
  dense::proj(dx.get(), M, w.matrix(), dseg.get(), nseg, out.get(), true, g_stream);
  sync();
  const auto got = out.download();
  std::vector<double> ref(got.size()), rnd(got.size());
  for (int s = 0; s < nseg; ++s)
    for (int r = seg[3 * s]; r < seg[3 * s + 1]; ++r)
      for (int m = 0; m < M; ++m) {
        ref[size_t(m) * R + r] = w.dot(r, &x[(size_t(seg[3 * s + 2]) * M + m) * H]);
        rnd[size_t(m) * R + r] = rbf(ref[size_t(m) * R + r]);
      }
  char what[96];
  std::snprintf(what, sizeof what, "proj M=%d %s %s%s", M, fp8 ? "fp8" : "bf16", full ? "full" : "sliding",
                single ? " single-input" : "");
  return Check{what, 1.0 / 128, 1e-6, 0.01}(got, ref, &rnd);
}

bool test_proj_conv(Rng& rng, int M, bool fp8) {
  const int K = 3072;
  const Weights w = make_weights(rng, H, K, fp8, 0.02);
  const std::vector<float> x = randn_bf16(rng, size_t(M) * K, 1.0);
  const std::vector<float> cw = randn(rng, size_t(H) * 4, 0.5);
  const std::vector<float> st0 = randn_bf16(rng, size_t(3) * H, 1.0);
  DeviceArray<__nv_bfloat16> dx(to_bf16(x));
  DeviceArray<float> dcw(cw), dst(st0), out(size_t(M) * H), hist(size_t(M) * H);
  dense::proj_conv(dx.get(), M, w.matrix(), dcw.get(), dst.get(), out.get(), hist.get(), true, g_stream);
  sync();
  const auto y = hist.download(), o = out.download(), st = dst.download();
  // the conv inputs: BF16-rounded projections
  std::vector<double> ry(y.size()), rr(y.size());
  for (int m = 0; m < M; ++m)
    for (int c = 0; c < H; ++c) {
      ry[size_t(m) * H + c] = w.dot(c, &x[size_t(m) * K]);
      rr[size_t(m) * H + c] = rbf(ry[size_t(m) * H + c]);
    }
  bool ok = Check{"proj_conv conv inputs", 1.0 / 128, 1e-6, 0.01}(y, ry, &rr);
  // the conv over [state, y] (oracle from the kernel's own y: checks the epilogue tightly)
  std::vector<double> ro(o.size()), rs(st.size());
  for (int c = 0; c < H; ++c) {
    std::vector<double> ext(M + 3);
    for (int j = 0; j < 3; ++j) ext[j] = st0[size_t(j) * H + c];
    for (int m = 0; m < M; ++m) ext[3 + m] = y[size_t(m) * H + c];
    for (int m = 0; m < M; ++m) {
      double s = 0;
      for (int j = 0; j < 4; ++j) s += double(cw[size_t(c) * 4 + j]) * ext[m + j];
      ro[size_t(m) * H + c] = (s + ext[m + 3]) / std::sqrt(5.0);
    }
    for (int j = 0; j < 3; ++j) rs[size_t(j) * H + c] = ext[M + j];
  }
  char what[64];
  std::snprintf(what, sizeof what, "proj_conv M=%d %s out", M, fp8 ? "fp8" : "bf16");
  ok &= Check{what, 1e-5, 1e-6, 1.0}(o, ro);
  ok &= Check{"proj_conv state", 0, 0, 0}(st, rs);
  return ok;
}

bool test_head(Rng& rng, int M, bool fp8) {
  const int V = 5000;
  const Weights w = make_weights(rng, V, H, fp8, 0.05);
  const std::vector<float> x = randn_bf16(rng, size_t(M) * H, 1.0);
  DeviceArray<__nv_bfloat16> dx(to_bf16(x));
  DeviceArray<float> out(size_t(M) * V);
  dense::head(dx.get(), M, w.matrix(), out.get(), true, g_stream);
  sync();
  const auto got = out.download();
  std::vector<double> ref(got.size());
  for (int m = 0; m < M; ++m)
    for (int v = 0; v < V; ++v) ref[size_t(m) * V + v] = 30.0 * std::tanh(w.dot(v, &x[size_t(m) * H]) / 30.0);
  char what[48];
  std::snprintf(what, sizeof what, "head M=%d %s", M, fp8 ? "fp8" : "bf16");
  return Check{what, 1.0 / 128, 1e-5, 1.0}(got, ref);
}

// ties: groups of experts with identical rows and biases (identical scores, the lower index must win), one
// group pushed into the top 8 so that the tie straddles the cut; some biases below -1 (negative scores).
bool test_router(Rng& rng, int M, bool wf32, bool ties = false) {
  std::vector<float> w = randn_bf16(rng, size_t(128) * H, 0.02);
  std::vector<float> bias = randn(rng, 128, 0.1);
  if (ties) {
    const int groups[3][4] = {{5, 40, 41, 127}, {0, 63, 64, 100}, {17, 18, 19, 90}};
    for (int g = 0; g < 3; ++g)
      for (int k = 1; k < 4; ++k) {
        std::copy_n(&w[size_t(groups[g][0]) * H], H, &w[size_t(groups[g][k]) * H]);
        bias[groups[g][k]] = bias[groups[g][0]];
      }
    for (int k = 0; k < 4; ++k) bias[groups[0][k]] = 2.f;   // all four in the top 8
    bias[50] = 2.5f;                                        // rank 0 or 4
    for (int k = 0; k < 4; ++k) bias[groups[1][k]] = 1.5f;  // four tied for ranks 5..7: expert 100 is cut
    for (int e : {7, 8, 9, 77}) bias[e] = -1.25f;
    for (int e : {30, 31, 32, 33, 34, 35}) bias[e] = 1.5f - 1e-3f * float(e);  // crowd around the cut
  }
  const std::vector<float> x = randn_bf16(rng, size_t(M) * H, 1.0);
  DeviceArray<__nv_bfloat16> dx(to_bf16(x)), dwb(to_bf16(w));
  DeviceArray<float> dwf(w), db(bias), wts(size_t(M) * 8);
  DeviceArray<int> idx(size_t(M) * 8);
  dense::router(dx.get(), M, wf32 ? static_cast<const void*>(dwf.get()) : static_cast<const void*>(dwb.get()), wf32,
                db.get(), idx.get(), wts.get(), true, g_stream);
  sync();
  const auto gi = idx.download();
  const auto gw = wts.download();
  bool ok = true;
  std::vector<double> rw(gw.size());
  for (int m = 0; m < M; ++m) {
    std::vector<double> s(128), v(128);
    for (int e = 0; e < 128; ++e) {
      double l = 0;
      for (int k = 0; k < H; ++k) l += double(w[size_t(e) * H + k]) * x[size_t(m) * H + k];
      s[e] = 1.0 / (1.0 + std::exp(-l));
      v[e] = s[e] + bias[e];
    }
    std::vector<int> order(128);
    std::iota(order.begin(), order.end(), 0);
    std::stable_sort(order.begin(), order.end(), [&](int a, int b) { return v[a] > v[b]; });
    double sum = 0;
    for (int k = 0; k < 8; ++k) sum += s[order[k]];
    for (int k = 0; k < 8; ++k) {
      if (gi[size_t(m) * 8 + k] != order[k]) ok = false;
      rw[size_t(m) * 8 + k] = s[order[k]] / (sum + 1e-20) * 8.0;
    }
  }
  std::printf("%s router M=%d %s%s: expert ids %s\n", ok ? "PASS" : "FAIL", M, wf32 ? "fp32" : "bf16",
              ties ? " ties" : "", ok ? "match" : "differ");
  return Check{"router weights", 1e-5, 1e-6, 1.0}(gw, rw) && ok;
}

struct MixCase {
  int nb, Q;
  bool base, write, abl, bf16, f32, partial;
  bool side = false;  // side outputs: bf16(new) and a copy of out_bf16 (Q = 1) at row stride 2H, two row copies
};

bool test_mix(Rng& rng, int M, const MixCase& c) {
  const size_t n = size_t(M) * H;
  std::vector<std::vector<float>> blocks;
  std::vector<DeviceArray<float>> dblocks;
  for (int i = 0; i < c.nb; ++i) {
    blocks.push_back(randn(rng, n, rng.uniform(0.5, 3.0)));
    dblocks.emplace_back(blocks.back());
  }
  std::vector<const float*> ptrs;
  for (auto& d : dblocks) ptrs.push_back(d.get());
  std::vector<uint64_t> ptr_bits(ptrs.size());
  std::memcpy(ptr_bits.data(), ptrs.data(), ptrs.size() * 8);
  DeviceArray<uint64_t> table(ptr_bits.empty() ? std::vector<uint64_t>{0} : ptr_bits);
  const std::vector<float> base = randn(rng, n, 1.0), write = randn(rng, n, 0.7);
  std::vector<float> r = randn(rng, H, 1.0);
  double rn = 0;
  for (float v : r) rn += double(v) * v;
  for (auto& v : r) v = float(v / std::sqrt(rn));
  const std::vector<float> aw = {0.8f};
  const std::vector<float> q = randn(rng, size_t(c.Q) * H, 0.05), g = randn(rng, size_t(c.Q) * H, 0.1);
  std::vector<float> gw(g);
  for (auto& v : gw) v += 1.f;
  DeviceArray<float> dbase(base), dwrite(write), dr(r), daw(aw), dq(q), dg(gw), part(n), of(size_t(c.Q) * n);
  DeviceArray<__nv_bfloat16> ob(size_t(c.Q) * n), nb16(size_t(M) * 2 * H), oc16(size_t(M) * 2 * H);
  const std::vector<float> csrc = randn(rng, size_t(M) * 3 * H, 1.0);
  DeviceArray<float> dcsrc(csrc), dcdst(size_t(M) * 2 * 1024);
  dense::MixParams p;
  p.blocks = c.nb ? reinterpret_cast<const float* const*>(table.get()) : nullptr;
  p.nblocks = c.nb;
  p.base = c.base ? dbase.get() : nullptr;
  p.write = c.write ? dwrite.get() : nullptr;
  p.abl_r = c.abl ? dr.get() : nullptr;
  p.abl_w = c.abl ? daw.get() : nullptr;
  p.partial_out = c.partial ? part.get() : nullptr;
  const bool skip = c.nb == 0 && c.Q == 1;
  p.queries = skip ? nullptr : dq.get();
  p.Q = c.Q;
  p.norm_w = dg.get();
  p.out_bf16 = c.bf16 ? ob.get() : nullptr;
  p.out_f32 = c.f32 ? of.get() : nullptr;
  p.M = M;
  const bool copy_out = c.side && c.bf16 && c.Q == 1;
  if (c.side) {
    p.new_bf16 = nb16.get() + H;  // second half of each 2H row
    p.new_bf16_stride = 2 * H;
    if (copy_out) p.out_copy = oc16.get(), p.out_copy_stride = 2 * H;
    // the K/V segments of a [M, 3H] row-major "projection" into two [M, 1024] slabs
    p.copies[0] = dense::MixRowCopy{dcsrc.get() + 100 * 4, 3 * H, dcdst.get(), 1024, 1024};
    p.copies[1] = dense::MixRowCopy{dcsrc.get() + 1000 * 4, 3 * H, dcdst.get() + size_t(M) * 1024, 1024, 1024};
  }
  dense::mix(p, true, g_stream);
  sync();
  // oracle
  std::vector<double> nw(n), mixr(size_t(c.Q) * n), outb(size_t(c.Q) * n), outbr(size_t(c.Q) * n);
  for (int m = 0; m < M; ++m) {
    const float* y = &write[size_t(m) * H];
    double d = 0;
    for (int k = 0; k < H; ++k) d += double(y[k]) * r[k];
    for (int k = 0; k < H; ++k) {
      double v = c.base ? base[size_t(m) * H + k] : 0.0;
      if (c.write) v += c.abl ? y[k] - aw[0] * d * r[k] : y[k];
      nw[size_t(m) * H + k] = v;
    }
    std::vector<const double*> states;
    std::vector<std::vector<double>> blk(c.nb);
    for (int i = 0; i < c.nb; ++i) {
      blk[i].assign(blocks[i].begin() + std::ptrdiff_t(m) * H, blocks[i].begin() + std::ptrdiff_t(m + 1) * H);
      states.push_back(blk[i].data());
    }
    states.push_back(&nw[size_t(m) * H]);
    for (int j = 0; j < c.Q; ++j) {
      std::vector<double> mixv(H);
      if (skip) {
        for (int k = 0; k < H; ++k) mixv[k] = nw[size_t(m) * H + k];
      } else {
        const int NS = int(states.size());
        std::vector<double> rinv(NS), lg(NS);
        double mx = -1e300;
        for (int i = 0; i < NS; ++i) {
          double ss = 0, dq = 0;
          for (int k = 0; k < H; ++k) ss += states[i][k] * states[i][k], dq += double(q[size_t(j) * H + k]) * states[i][k];
          rinv[i] = 1.0 / std::sqrt(ss / H + 1e-6);
          lg[i] = dq * rinv[i] / 48.0;
          mx = std::max(mx, lg[i]);
        }
        double den = 0;
        for (auto& l : lg) den += (l = std::exp(l - mx));
        for (int k = 0; k < H; ++k) {
          double s = 0;
          for (int i = 0; i < NS; ++i) s += lg[i] / den * states[i][k] * rinv[i];
          mixv[k] = s;
        }
      }
      double ss = 0;
      for (double v : mixv) ss += v * v;
      const double ri = 1.0 / std::sqrt(ss / H + 1e-6);
      for (int k = 0; k < H; ++k) {
        const size_t o = (size_t(j) * M + m) * H + k;
        mixr[o] = mixv[k];
        outb[o] = mixv[k] * ri * gw[size_t(j) * H + k];
        outbr[o] = rbf(outb[o]);
      }
    }
  }
  char what[96];
  std::snprintf(what, sizeof what, "mix M=%d blocks=%d Q=%d%s%s%s%s", M, c.nb, c.Q, c.base ? " base" : "",
                c.write ? " write" : "", c.abl ? " abl" : "", c.side ? " side" : "");
  bool ok = true;
  if (c.partial) ok &= Check{(std::string(what) + " partial").c_str(), 1e-5, 1e-5, 1.0}(part.download(), nw);
  if (c.f32) ok &= Check{(std::string(what) + " f32").c_str(), 1e-4, 1e-5, 1.0}(of.download(), mixr);
  if (c.bf16) ok &= Check{(std::string(what) + " bf16").c_str(), 1.0 / 128, 1e-5, 0.01}(bf16_download(ob), outb, &outbr);
  if (c.side) {
    // bf16(new) and the out_bf16 copy are bit-exact with the kernel's own FP32 `new` / out_bf16
    const auto gn = bf16_download(nb16), gc = bf16_download(oc16), gb = bf16_download(ob);
    const auto gp = c.partial ? part.download() : std::vector<float>();
    std::vector<float> gnew(n);
    std::vector<double> rnew(n);
    size_t bad = 0;
    for (int m = 0; m < M; ++m)
      for (int k = 0; k < H; ++k) {
        const size_t i = size_t(m) * H + k;
        gnew[i] = gn[size_t(m) * 2 * H + H + k];
        rnew[i] = rbf(nw[i]);
        if (c.partial && gnew[i] != round_bf16(gp[i])) ++bad;
        if (gn[size_t(m) * 2 * H + k] != 0.f) ++bad;  // untouched first half
        if (copy_out && gc[size_t(m) * 2 * H + k] != gb[i]) ++bad;
      }
    const auto gd = dcdst.download();
    for (int m = 0; m < M; ++m)
      for (int k = 0; k < 1024; ++k) {
        bad += gd[size_t(m) * 1024 + k] != csrc[size_t(m) * 3 * H + 400 + k];
        bad += gd[(size_t(M) + m) * 1024 + k] != csrc[size_t(m) * 3 * H + 4000 + k];
      }
    ok &= Check{(std::string(what) + " new_bf16").c_str(), 1.0 / 128, 1e-5, 0.01}(gnew, nw, &rnew);
    std::printf("%s %s side outputs: %zu mismatches\n", bad ? "FAIL" : "PASS", what, bad);
    ok &= bad == 0;
  }
  return ok;
}

bool test_gates(Rng& rng, int G, int M, bool fg32, bool h_shared = false) {
  const size_t n = size_t(G) * M * H;
  const std::vector<float> h = randn(rng, h_shared ? size_t(M) * H : n, 1.0), msg = randn(rng, n, 1.0);
  const std::vector<float> wd = randn_bf16(rng, size_t(G) * 64 * 2 * H, 0.02);
  const std::vector<float> wf = randn_bf16(rng, size_t(G) * H * 64, 0.1), wg = randn_bf16(rng, size_t(G) * H * 64, 0.1);
  std::vector<float> nw = randn(rng, size_t(G) * H, 0.1);
  for (auto& v : nw) v += 1.f;
  DeviceArray<float> dh(h), dm(msg), dwf32(wf), dwg32(wg), dn(nw), of(n);
  DeviceArray<__nv_bfloat16> dwd(to_bf16(wd)), dwf16(to_bf16(wf)), dwg16(to_bf16(wg)), ob(n);
  dense::message_gates(dh.get(), h_shared, dm.get(), G, M, dwd.get(),
                       fg32 ? (const void*)dwf32.get() : (const void*)dwf16.get(),
                       fg32 ? (const void*)dwg32.get() : (const void*)dwg16.get(), fg32, dn.get(), ob.get(), of.get(),
                       true, g_stream);
  sync();
  std::vector<double> gated(n), out(n), outr(n);
  for (int g = 0; g < G; ++g)
    for (int m = 0; m < M; ++m) {
      const float* hv = &h[((h_shared ? 0 : size_t(g)) * M + m) * H];
      const float* mv = &msg[(size_t(g) * M + m) * H];
      double sh = 0, sm = 0;
      for (int k = 0; k < H; ++k) sh += double(hv[k]) * hv[k], sm += double(mv[k]) * mv[k];
      const double rh = 1.0 / std::sqrt(sh / H + 1e-6), rm = 1.0 / std::sqrt(sm / H + 1e-6);
      std::vector<double> xs(2 * H), gd(64);
      for (int k = 0; k < H; ++k) xs[k] = rbf(hv[k] * rh), xs[H + k] = rbf(mv[k] * rm);
      for (int d = 0; d < 64; ++d) {
        double s = 0;
        for (int k = 0; k < 2 * H; ++k) s += double(wd[(size_t(g) * 64 + d) * 2 * H + k]) * xs[k];
        gd[d] = rbf(s);
      }
      double ss = 0;
      for (int c = 0; c < H; ++c) {
        double df = 0, dg = 0;
        for (int d = 0; d < 64; ++d)
          df += double(wf[(size_t(g) * H + c) * 64 + d]) * gd[d], dg += double(wg[(size_t(g) * H + c) * 64 + d]) * gd[d];
        const double v = (1 - std::tanh(df)) * hv[c] + std::tanh(dg) * mv[c];
        gated[(size_t(g) * M + m) * H + c] = v;
        ss += v * v;
      }
      const double r = 1.0 / std::sqrt(ss / H + 1e-6);
      for (int c = 0; c < H; ++c) {
        const size_t o = (size_t(g) * M + m) * H + c;
        out[o] = gated[o] * r * nw[size_t(g) * H + c];
        outr[o] = rbf(out[o]);
      }
    }
  char what[64];
  std::snprintf(what, sizeof what, "message_gates G=%d M=%d %s%s", G, M, fg32 ? "fp32" : "bf16",
                h_shared ? " shared-h" : "");
  // the BF16 roundings of the normalised inputs and of g may flip against the oracle: loose bounds
  bool ok = Check{(std::string(what) + " gated").c_str(), 2e-3, 2e-3, 1.0}(of.download(), gated);
  ok &= Check{(std::string(what) + " out").c_str(), 1.0 / 64, 4e-3, 0.05}(bf16_download(ob), out, &outr);
  return ok;
}

bool test_embed(Rng& rng, int M) {
  const int V = 1000;
  const std::vector<float> table = randn_bf16(rng, size_t(V) * H, 0.05);
  std::vector<float> nw = randn(rng, H, 0.2);
  for (auto& v : nw) v += 1.f;
  std::vector<int64_t> tok(M);
  for (auto& t : tok) t = int64_t(rng.next() % V);
  DeviceArray<__nv_bfloat16> dt(to_bf16(table));
  DeviceArray<float> dn(nw), out(size_t(M) * H);
  DeviceArray<int64_t> dtok(tok);
  dense::embed(dtok.get(), M, dt.get(), dn.get(), out.get(), true, g_stream);
  sync();
  std::vector<double> ref(size_t(M) * H), rnd(ref.size());
  for (int m = 0; m < M; ++m) {
    const float* x = &table[size_t(tok[m]) * H];
    double ss = 0;
    for (int k = 0; k < H; ++k) ss += double(x[k]) * x[k];
    const double r = 1.0 / std::sqrt(ss / H + 1e-6);
    for (int k = 0; k < H; ++k) ref[size_t(m) * H + k] = x[k] * r * nw[k], rnd[size_t(m) * H + k] = rbf(ref[size_t(m) * H + k]);
  }
  char what[32];
  std::snprintf(what, sizeof what, "embed M=%d", M);
  return Check{what, 1.0 / 128, 1e-6, 0.01}(out.download(), ref, &rnd);
}

bool test_fp8_rows(Rng& rng) {
  const int rows = 64, K = H;
  std::vector<float> w = randn_bf16(rng, size_t(rows) * K, 0.03);
  for (int k = 0; k < K; ++k) w[size_t(5) * K + k] = 0.f;  // an all-zero row: scale 0, codes 0
  DeviceArray<__nv_bfloat16> dw(to_bf16(w));
  DeviceArray<uint8_t> q(size_t(rows) * K);
  DeviceArray<float> s(rows);
  DeviceArray<__nv_bfloat16> back(size_t(rows) * K);
  dense::fp8_rows(dw.get(), rows, K, q.get(), s.get(), g_stream);
  dense::dequant_rows(q.get(), s.get(), rows, K, back.get(), g_stream);
  sync();
  const auto gq = q.download();
  const auto gs = s.download();
  const auto gb = bf16_download(back);
  size_t bad_scale = 0, bad_code = 0, bad_back = 0;
  for (int r = 0; r < rows; ++r) {
    float amax = 0;
    for (int k = 0; k < K; ++k) amax = std::max(amax, std::fabs(w[size_t(r) * K + k]));
    const float sc = amax / 448.f;
    bad_scale += gs[r] != sc;
    for (int k = 0; k < K; ++k) {
      const size_t i = size_t(r) * K + k;
      const uint8_t want = sc > 0 ? float_to_e4m3(w[i] / sc) : 0;
      bad_code += gq[i] != want;
      bad_back += gb[i] != round_bf16(e4m3_to_float(gq[i]) * gs[r]);
    }
  }
  const bool ok = bad_scale == 0 && bad_code == 0 && bad_back == 0;
  std::printf("%s fp8_rows/dequant_rows: scale mismatches %zu, code mismatches %zu, dequant mismatches %zu\n",
              ok ? "PASS" : "FAIL", bad_scale, bad_code, bad_back);
  return ok;
}

// ---- Bitwise lock. mix, message_gates and router are exact: on fixed inputs they must keep giving the bits
// recorded in kLock (taken from kernels checked byte for byte against the implementation they were ported
// from). A changed hash means the arithmetic changed (order of a sum, an FMA contraction the compiler chose
// differently, a rounding point). Either undo that, or, when the change is deliberate and qualified against
// the model (ingine-validate), record the new hashes this test prints.
struct LockGen {  // integer generator, exact float conversion: the inputs do not depend on libm
  uint64_t s;
  uint64_t next() {
    s += 0x9e3779b97f4a7c15ull;
    uint64_t z = s;
    z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ull;
    z = (z ^ (z >> 27)) * 0x94d049bb133111ebull;
    return z ^ (z >> 31);
  }
  float unit() { return float(int32_t(next() >> 40) - (1 << 23)) * (1.f / float(1 << 23)); }  // [-1, 1)
  std::vector<float> vec(size_t n, float scale, float offset = 0.f) {
    std::vector<float> v(n);
    for (auto& x : v) x = unit() * scale + offset;
    return v;
  }
  std::vector<float> vec_bf16(size_t n, float scale) {
    std::vector<float> v = vec(n, scale);
    for (auto& x : v) x = round_bf16(x);
    return v;
  }
};

struct Fnv {
  uint64_t h = 1469598103934665603ull;
  template <class T>
  void add(const std::vector<T>& v) {
    const unsigned char* b = reinterpret_cast<const unsigned char*>(v.data());
    for (size_t i = 0; i < v.size() * sizeof(T); ++i) h = (h ^ b[i]) * 1099511628211ull;
  }
};

// mix over the call shapes of the decode step, the pair launch included (same bits as two launches)
uint64_t lock_mix(LockGen& g, int M, bool& pair_ok) {
  const size_t n = size_t(M) * H;
  std::vector<DeviceArray<float>> blocks;
  std::vector<uint64_t> ptrs;
  for (int i = 0; i < 5; ++i) {
    blocks.emplace_back(g.vec(n, 2.f));
    ptrs.push_back(reinterpret_cast<uint64_t>(blocks.back().get()));
  }
  DeviceArray<uint64_t> table(ptrs), table4(std::vector<uint64_t>{ptrs[4]});
  DeviceArray<float> base(g.vec(n, 1.5f)), write(g.vec(n, 1.f)), r(g.vec(H, 0.03f)), aw(std::vector<float>{0.8125f});
  DeviceArray<float> q(g.vec(size_t(4) * H, 0.08f)), nw(g.vec(size_t(4) * H, 0.25f, 1.f));
  Fnv f;
  auto run = [&](int nb, int Q, bool has_base, bool has_write, bool abl, bool bf16, bool f32) {
    DeviceArray<float> part(n), of(size_t(Q) * n);
    DeviceArray<__nv_bfloat16> ob(size_t(Q) * n), nb16(size_t(M) * 2 * H);
    dense::MixParams p;
    p.blocks = nb ? reinterpret_cast<const float* const*>(table.get()) : nullptr;
    p.nblocks = nb;
    p.base = has_base ? base.get() : nullptr;
    p.write = has_write ? write.get() : nullptr;
    p.abl_r = abl ? r.get() : nullptr;
    p.abl_w = abl ? aw.get() : nullptr;
    p.partial_out = part.get();
    p.queries = nb == 0 && Q == 1 ? nullptr : q.get();
    p.Q = Q;
    p.norm_w = nw.get();
    p.out_bf16 = bf16 ? ob.get() : nullptr;
    p.out_f32 = f32 ? of.get() : nullptr;
    p.new_bf16 = nb16.get() + H;
    p.new_bf16_stride = 2 * H;
    p.M = M;
    dense::mix(p, true, g_stream);
    sync();
    f.add(part.download()), f.add(of.download()), f.add(ob.download()), f.add(nb16.download());
  };
  run(0, 1, true, false, false, true, false);  // layer 0
  run(1, 1, true, true, true, true, false);    // sliding attention read
  run(2, 3, true, true, true, true, true);     // full attention read
  run(3, 1, false, true, true, true, false);   // MLP read at a block boundary
  run(5, 1, true, true, false, true, false);   // the most blocks
  // a message visit's two reads: one launch and two launches
  std::vector<float> got[2][4];
  for (int pair = 0; pair < 2; ++pair) {
    DeviceArray<float> part(n), of(n), mpart(n), msgs(size_t(4) * n);
    dense::MixParams a, b;
    a.blocks = reinterpret_cast<const float* const*>(table.get()), a.nblocks = 2;
    a.base = base.get(), a.write = write.get(), a.abl_r = r.get(), a.abl_w = aw.get();
    a.partial_out = part.get(), a.queries = q.get(), a.Q = 1, a.out_f32 = of.get(), a.M = M;
    b.blocks = reinterpret_cast<const float* const*>(table4.get()), b.nblocks = 1;
    b.base = blocks[3].get(), b.write = write.get(), b.abl_r = r.get(), b.abl_w = aw.get();
    b.partial_out = mpart.get(), b.queries = q.get(), b.Q = 4, b.out_f32 = msgs.get(), b.M = M;
    if (pair) {
      dense::mix_pair(a, b, true, g_stream);
    } else {
      dense::mix(b, true, g_stream);
      dense::mix(a, true, g_stream);
    }
    sync();
    got[pair][0] = part.download(), got[pair][1] = of.download(), got[pair][2] = mpart.download();
    got[pair][3] = msgs.download();
  }
  for (int k = 0; k < 4; ++k) {
    f.add(got[0][k]);
    pair_ok = pair_ok && std::memcmp(got[0][k].data(), got[1][k].data(), got[0][k].size() * 4) == 0;
  }
  return f.h;
}

uint64_t lock_gates(LockGen& g, int M) {
  Fnv f;
  for (int G : {3, 1}) {
    const bool shared = G == 3;
    const size_t n = size_t(G) * M * H;
    DeviceArray<float> h(g.vec(shared ? size_t(M) * H : n, 2.f)), msg(g.vec(n, 2.f)), nw(g.vec(size_t(G) * H, 0.25f, 1.f));
    DeviceArray<float> of(n);
    DeviceArray<__nv_bfloat16> wd(to_bf16(g.vec_bf16(size_t(G) * 64 * 2 * H, 0.04f)));
    DeviceArray<__nv_bfloat16> wf(to_bf16(g.vec_bf16(size_t(G) * H * 64, 0.2f))), wg(to_bf16(g.vec_bf16(size_t(G) * H * 64, 0.2f)));
    DeviceArray<__nv_bfloat16> ob(n);
    dense::message_gates(h.get(), shared, msg.get(), G, M, wd.get(), wf.get(), wg.get(), false, nw.get(), ob.get(),
                         of.get(), true, g_stream);
    sync();
    f.add(ob.download()), f.add(of.download());
  }
  return f.h;
}

uint64_t lock_router(LockGen& g, int M) {
  std::vector<float> w = g.vec_bf16(size_t(128) * H, 0.04f), bias = g.vec(128, 0.2f);
  for (int e : {9, 70, 71, 120}) {  // tied experts
    std::copy_n(&w[size_t(3) * H], H, &w[size_t(e) * H]);
    bias[e] = bias[3];
  }
  DeviceArray<__nv_bfloat16> x(to_bf16(g.vec_bf16(size_t(M) * H, 2.f))), dw(to_bf16(w));
  DeviceArray<float> db(bias), wts(size_t(M) * 8);
  DeviceArray<int> idx(size_t(M) * 8);
  dense::router(x.get(), M, dw.get(), false, db.get(), idx.get(), wts.get(), true, g_stream);
  sync();
  Fnv f;
  f.add(idx.download()), f.add(wts.download());
  return f.h;
}

// [M - 1][mix, message_gates, router]
constexpr uint64_t kLock[8][3] = {
    {0x2b27dd32a1dd100eull, 0x5efdd3c8b07ce184ull, 0x51c35a315b7e90b3ull},
    {0x75334db64229e2e1ull, 0x1940463968d37814ull, 0xea1ef0b8721fb3ffull},
    {0xbcbd0a5db598e31eull, 0xc5d6d6cbeced9705ull, 0x92540ffd2106daacull},
    {0xa6717a49034460f7ull, 0xc2cec0ccbcefa691ull, 0x70dd87ceaf1a3d15ull},
    {0xee8f3095dd53a1c1ull, 0xfa3a8c6799d1da0aull, 0xe7e125992e15339eull},
    {0xbe73b36287e82a47ull, 0x61ac1608d61c8021ull, 0xc8df312f5d2c4bc3ull},
    {0x82fcad4fa28a5d54ull, 0x908da1a2466cd8ebull, 0x76e91bc67374d743ull},
    {0x485152f0b0ead6f4ull, 0xd6dded13cf36fa5cull, 0xffe0d044657686f0ull},
};

bool test_bit_lock() {
  bool ok = true;
  for (int M = 1; M <= 8; ++M) {
    LockGen gm{0x5eed0000ull + uint64_t(M)}, gg{0x5eed1000ull + uint64_t(M)}, gr{0x5eed2000ull + uint64_t(M)};
    bool pair_ok = true;
    const uint64_t got[3] = {lock_mix(gm, M, pair_ok), lock_gates(gg, M), lock_router(gr, M)};
    const bool same = got[0] == kLock[M - 1][0] && got[1] == kLock[M - 1][1] && got[2] == kLock[M - 1][2];
    std::printf("%s bit lock M=%d: {0x%016llxull, 0x%016llxull, 0x%016llxull}%s%s\n", same && pair_ok ? "PASS" : "FAIL", M,
                static_cast<unsigned long long>(got[0]), static_cast<unsigned long long>(got[1]),
                static_cast<unsigned long long>(got[2]), same ? "" : " differs from the recorded hashes",
                pair_ok ? "" : "; mix_pair differs from two mix launches");
    ok = ok && same && pair_ok;
  }
  return ok;
}

}  // namespace

int main() {
  if (skip_without_gpu()) return kSkip;
  INGINE_TEST_CUDA(cudaStreamCreateWithFlags(&g_stream, cudaStreamNonBlocking));
  Rng rng(20261120);
  bool ok = true;
  for (int M : {1, 3, 8}) {
    ok &= test_proj(rng, M, false, false, true);
    ok &= test_proj(rng, M, true, true, false);
  }
  ok &= test_proj(rng, 2, false, false, false);
  ok &= test_proj(rng, 5, true, false, true);
  for (int M : {1, 4}) {
    ok &= test_proj_conv(rng, M, false);
    ok &= test_proj_conv(rng, M, true);
  }
  for (int M : {1, 8}) {
    ok &= test_head(rng, M, false);
    ok &= test_head(rng, M, true);
    ok &= test_router(rng, M, true);
    ok &= test_router(rng, M, false);
  }
  ok &= test_router(rng, 3, false);
  ok &= test_router(rng, 2, false, true);
  ok &= test_router(rng, 8, true, true);
  const MixCase cases[] = {
      {0, 1, true, false, false, true, false, false},   // layer 0: base only, no mix
      {0, 1, true, true, true, true, true, true},       // skip path with an ablated write
      {1, 1, true, true, true, true, false, true},      // sliding attention read
      {3, 3, false, true, true, true, true, true},      // full layer at a block boundary
      {1, 4, true, true, true, false, true, true},      // loop messages (Q = 4)
      {5, 1, true, true, false, true, false, false},    // the most blocks a mix takes (5)
      {2, 1, true, true, true, true, false, true, true},     // side outputs (taps, history copies)
      {0, 1, true, true, true, true, false, false, true},    // side outputs on the skip path, no partial
      {4, 3, true, true, true, false, true, true, true},     // side outputs with Q = 3 (no out copy)
  };
  for (int M : {1, 3})
    for (const auto& c : cases) ok &= test_mix(rng, M, c);
  ok &= test_gates(rng, 3, 1, false);
  ok &= test_gates(rng, 3, 2, true);
  ok &= test_gates(rng, 1, 8, false);
  ok &= test_gates(rng, 3, 4, false, true);
  ok &= test_gates(rng, 3, 8, false, true);
  ok &= test_gates(rng, 1, 5, true);  // FP32 forget/gain with several tokens: one staging buffer
  for (int M : {1, 3, 8}) ok &= test_embed(rng, M);
  ok &= test_fp8_rows(rng);
  ok &= test_bit_lock();
  INGINE_TEST_CUDA(cudaStreamDestroy(g_stream));
  std::printf("%s\n", ok ? "ALL PASS" : "SOME FAILED");
  return ok ? 0 : 1;
}
