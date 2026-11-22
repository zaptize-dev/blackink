// Routed-expert kernels (src/kernels/moe) against FP64 oracles of KERNELS.md area `moe` (adapted from the
// PyTorch engine's test_moe.py): moe_up and moe_down for NVFP4 and MXFP8 experts, M = 1, 3, 8 with experts
// shared between tokens (deduplicated passes), the post-MLP conv state update and conv history, and
// dequant_experts. Synthetic experts in checkpoint encoding; only experts 0..E-1 are allocated (routing
// stays below E), which keeps GPU memory small.
#include "kernels/oracle.h"

#include "kernels/moe/moe.h"

#include <cuda_runtime.h>

#include <numeric>

using namespace ingine::test;
namespace moe = ingine::kernels::moe;

namespace {

constexpr int H = 2304, I = 1152, TOPK = 8, E = 10;
cudaStream_t g_stream = nullptr;

void sync() { INGINE_TEST_CUDA(cudaStreamSynchronize(g_stream)); }

// One expert weight [E, rows, cols] in checkpoint encoding plus its decoded values (model.decode_experts).
struct ExpertWeight {
  int rows = 0, cols = 0;
  bool nv = true;
  std::vector<float> value;  // [E, rows, cols] decoded BF16 values
  std::optional<DeviceArray<uint8_t>> codes, scales;
  std::optional<DeviceArray<float>> factor;  // [128]
};

ExpertWeight make_expert_weight(Rng& rng, int rows, int cols, bool nv) {
  ExpertWeight w;
  w.rows = rows, w.cols = cols, w.nv = nv;
  const size_t n = size_t(E) * rows * cols;
  w.value.resize(n);
  std::vector<float> factor(128, 0.f);
  if (nv) {
    std::vector<uint8_t> codes(n / 2), scales(n / 16);
    for (auto& c : codes) c = uint8_t(rng.next());
    for (auto& s : scales) s = float_to_e4m3(float(rng.uniform(16.0, 448.0)));
    for (int e = 0; e < E; ++e) factor[e] = float(rng.uniform(0.5, 1.5) * 0.002 / 448.0 * 64.0);
    for (size_t i = 0; i < n; ++i) {
      const int e = int(i / (size_t(rows) * cols));
      const uint8_t byte = codes[i / 2];
      const float code = kE2M1[(i & 1) ? byte >> 4 : byte & 0xF];
      const float cs = code * e4m3_to_float(scales[i / 16]);  // exact (as in FP16)
      w.value[i] = round_bf16(cs * factor[e]);
    }
    w.codes.emplace(codes);
    w.scales.emplace(scales);
  } else {
    std::vector<uint8_t> codes(n), scales(n / 32);
    for (auto& c : codes) {
      c = uint8_t(rng.next());
      if ((c & 0x7f) == 0x7f) c ^= 1;  // no NaN
    }
    for (auto& s : scales) s = uint8_t(112 + rng.next() % 6);  // weights of real magnitude (|w| < 0.5)
    for (size_t i = 0; i < n; ++i) w.value[i] = round_bf16(e4m3_to_float(codes[i]) * std::ldexp(1.0f, scales[i / 32] - 127));
    w.codes.emplace(codes);
    w.scales.emplace(scales);
  }
  w.factor.emplace(factor);
  return w;
}

// Routing: 8 distinct experts of 0..E-1 per token (tokens share experts).
std::vector<int> routing(Rng& rng, int M) {
  std::vector<int> idx;
  for (int m = 0; m < M; ++m) {
    std::vector<int> p(E);
    std::iota(p.begin(), p.end(), 0);
    for (int i = E - 1; i > 0; --i) std::swap(p[i], p[rng.next() % (i + 1)]);
    idx.insert(idx.end(), p.begin(), p.begin() + TOPK);
  }
  return idx;
}

double softcap(double x, double c) { return std::tanh(x / c) * c; }

bool test_up(Rng& rng, const ExpertWeight& w13, int M) {
  const std::vector<float> x = randn_bf16(rng, size_t(M) * H, 1.0);
  const std::vector<int> idx = routing(rng, M);
  DeviceArray<__nv_bfloat16> dx(to_bf16(x)), out(size_t(M) * TOPK * I);
  DeviceArray<int> didx(idx);
  moe::moe_up(dx.get(), M, didx.get(), w13.codes->get(), w13.scales->get(), w13.nv ? w13.factor->get() : nullptr,
              w13.nv ? moe::Format::NVFP4 : moe::Format::MXFP8, out.get(), true, g_stream);
  sync();
  std::vector<double> ref(size_t(M) * TOPK * I), rnd(ref.size());
  for (int m = 0; m < M; ++m)
    for (int k = 0; k < TOPK; ++k) {
      const int e = idx[size_t(m) * TOPK + k];
      for (int i = 0; i < I; ++i) {
        double gu[2] = {0, 0};
        for (int j = 0; j < 2; ++j) {
          const float* row = &w13.value[(size_t(e) * 2 * I + 2 * i + j) * H];
          for (int h = 0; h < H; ++h) gu[j] += double(row[h]) * x[size_t(m) * H + h];
        }
        const double g = rbf(gu[0]), u = rbf(gu[1]);
        const size_t o = (size_t(m) * TOPK + k) * I + i;
        ref[o] = softcap(g, 4.0) / (1.0 + std::exp(-g)) * softcap(u, 25.0);
        rnd[o] = rbf(ref[o]);
      }
    }
  char what[48];
  std::snprintf(what, sizeof what, "moe_up %s M=%d", w13.nv ? "nvfp4" : "mxfp8", M);
  return Check{what, 1.0 / 64, 2e-3, 0.02}(bf16_download(out), ref, &rnd);
}

bool test_down(Rng& rng, const ExpertWeight& w2, int M) {
  const std::vector<float> act = randn_bf16(rng, size_t(M) * TOPK * I, 0.5);
  const std::vector<int> idx = routing(rng, M);
  std::vector<float> wts(size_t(M) * TOPK);
  for (auto& v : wts) v = float(rng.uniform(0.2, 2.0));
  const std::vector<float> cw = randn(rng, size_t(H) * 4, 0.5);
  const std::vector<float> st0 = randn_bf16(rng, size_t(3) * H, 0.3);
  DeviceArray<__nv_bfloat16> dact(to_bf16(act));
  DeviceArray<int> didx(idx);
  DeviceArray<float> dw(wts), dcw(cw), dst(st0), out(size_t(M) * H), hist(size_t(M) * H);
  moe::moe_down(dact.get(), M, didx.get(), dw.get(), w2.codes->get(), w2.scales->get(), w2.nv ? w2.factor->get() : nullptr,
                w2.nv ? moe::Format::NVFP4 : moe::Format::MXFP8, dcw.get(), dst.get(), out.get(), hist.get(), true,
                g_stream);
  sync();
  const auto s = hist.download(), o = out.download(), st = dst.download();
  std::vector<double> rs(s.size()), rsr(s.size());
  for (int m = 0; m < M; ++m)
    for (int c = 0; c < H; ++c) {
      double sum = 0;
      for (int k = 0; k < TOPK; ++k) {
        const int e = idx[size_t(m) * TOPK + k];
        const float* row = &w2.value[(size_t(e) * H + c) * I];
        double y = 0;
        for (int i = 0; i < I; ++i) y += double(row[i]) * act[(size_t(m) * TOPK + k) * I + i];
        sum += double(wts[size_t(m) * TOPK + k]) * rbf(y);
      }
      rs[size_t(m) * H + c] = sum;
      rsr[size_t(m) * H + c] = rbf(sum);
    }
  char what[64];
  std::snprintf(what, sizeof what, "moe_down %s M=%d expert sum", w2.nv ? "nvfp4" : "mxfp8", M);
  bool ok = Check{what, 1.0 / 64, 1e-4, 0.02}(s, rs, &rsr);
  std::vector<double> ro(o.size()), rst(st.size());
  for (int c = 0; c < H; ++c) {
    std::vector<double> ext(M + 3);
    for (int j = 0; j < 3; ++j) ext[j] = st0[size_t(j) * H + c];
    for (int m = 0; m < M; ++m) ext[3 + m] = s[size_t(m) * H + c];
    for (int m = 0; m < M; ++m) {
      double a = 0;
      for (int j = 0; j < 4; ++j) a += double(cw[size_t(c) * 4 + j]) * ext[m + j];
      ro[size_t(m) * H + c] = (a + ext[m + 3]) / std::sqrt(5.0);
    }
    for (int j = 0; j < 3; ++j) rst[size_t(j) * H + c] = ext[M + j];
  }
  // FP32 sums of up to five terms: the absolute error follows the largest input, not the (cancelling) result
  double big = 0;
  for (float v : s) big = std::max(big, double(std::fabs(v)));
  for (float v : st0) big = std::max(big, double(std::fabs(v)));
  ok &= Check{"moe_down conv out", 1e-5, 4e-7 * big + 1e-6, 1.0}(o, ro);
  ok &= Check{"moe_down conv state", 0, 0, 0}(st, rst);
  return ok;
}

bool test_dequant(const ExpertWeight& w) {
  const std::vector<int> experts = {E - 1, 0, 3};
  DeviceArray<int> de(experts);
  DeviceArray<__nv_bfloat16> out(experts.size() * w.rows * w.cols);
  moe::dequant_experts(w.codes->get(), w.scales->get(), w.nv ? w.factor->get() : nullptr, de.get(), int(experts.size()),
                       w.rows, w.cols, w.nv ? moe::Format::NVFP4 : moe::Format::MXFP8, out.get(), g_stream);
  sync();
  const auto got = bf16_download(out);
  size_t bad = 0;
  for (size_t s = 0; s < experts.size(); ++s)
    for (size_t i = 0; i < size_t(w.rows) * w.cols; ++i)
      bad += got[s * w.rows * w.cols + i] != w.value[size_t(experts[s]) * w.rows * w.cols + i];
  std::printf("%s dequant_experts %s [%d, %d]: %zu mismatches\n", bad ? "FAIL" : "PASS", w.nv ? "nvfp4" : "mxfp8",
              w.rows, w.cols, bad);
  return bad == 0;
}

}  // namespace

int main() {
  if (skip_without_gpu()) return kSkip;
  INGINE_TEST_CUDA(cudaStreamCreateWithFlags(&g_stream, cudaStreamNonBlocking));
  Rng rng(1152);
  bool ok = true;
  for (bool nv : {true, false}) {
    const ExpertWeight w13 = make_expert_weight(rng, 2 * I, H, nv);
    for (int M : {1, 3, 8}) ok &= test_up(rng, w13, M);
    ok &= test_dequant(w13);
  }
  for (bool nv : {true, false}) {
    const ExpertWeight w2 = make_expert_weight(rng, H, I, nv);
    for (int M : {1, 3, 8}) ok &= test_down(rng, w2, M);
    ok &= test_dequant(w2);
  }
  INGINE_TEST_CUDA(cudaStreamDestroy(g_stream));
  std::printf("%s\n", ok ? "ALL PASS" : "SOME FAILED");
  return ok ? 0 : 1;
}
