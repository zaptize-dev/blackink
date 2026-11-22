// Model-level checks of the Prefiller with the real weights (orchestrator: needs the whole model on the GPU).
//   argv[1] = "bf16" | "fp8" (full-attention cache precision).
// 1. chunking: N tokens prefilled as 2048 + rest vs. chunks of 700: last-token logits agree (KL, top-1);
// 2. prefill vs. decode: N - 8 tokens prefilled, then 8 decode steps (the decode kernels on the prefilled
//    state): last-token logits agree with the full prefill (KL, top-1) and so do the DSpark taps of that row.
// Tokens: the first 3000 of the t5000 model-golden prompt (repository text).
// Thresholds are set from the Python reference itself (engine/ingine model.prefill + fast decode, same tokens,
// ablated, dense FP8): chunking KL 3.7e-3 (BF16 KV) / 2.3e-3 (FP8 KV); prefill vs decode KL 1.6e-3 / 1.4e-3;
// tap rms rel diffs 3e-3, 5e-2, 9e-2, 1.6e-1 (BF16) / 3e-3, 5e-2, 7e-2, 1e-1 (FP8). The 48-visit stack amplifies
// last-bit differences (MoE routing flips, FP8 cache rounding) to that level; Python with exact FP32 attention
// instead of flex's TF32 differs from its flex prefill by up to KL 8e-2 on these prompts.
#include "model/prefill.h"
#include "model/runtime.h"
#include <nlohmann/json.hpp>
#include "prefill/prefill_test.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <fstream>
#include <string>
#include <vector>

using namespace ingine;
using namespace ingine::model;
using namespace ingine::test;
using namespace ingine::test::prefill;

namespace {

std::vector<float> download(const float* p, size_t n) {
  std::vector<float> h(n);
  INGINE_TEST_CUDA(cudaMemcpy(h.data(), p, n * 4, cudaMemcpyDeviceToHost));
  return h;
}

double kl(const std::vector<float>& a, const std::vector<float>& b) {
  const double ma = *std::max_element(a.begin(), a.end()), mb = *std::max_element(b.begin(), b.end());
  double za = 0, zb = 0;
  for (size_t i = 0; i < a.size(); ++i) za += std::exp(a[i] - ma), zb += std::exp(b[i] - mb);
  double k = 0;
  for (size_t i = 0; i < a.size(); ++i) {
    const double la = a[i] - ma - std::log(za), lb = b[i] - mb - std::log(zb);
    k += std::exp(la) * (la - lb);
  }
  return k;
}

int argmax(const std::vector<float>& a) { return int(std::max_element(a.begin(), a.end()) - a.begin()); }

bool compare(const char* what, const std::vector<float>& a, const std::vector<float>& b, double max_kl) {
  const double k = kl(a, b);
  const bool ok = k <= max_kl && argmax(a) == argmax(b);
  std::printf("%s %s: KL %.3e (max %.1e), top-1 %d / %d\n", ok ? "PASS" : "FAIL", what, k, max_kl, argmax(a), argmax(b));
  return ok;
}

}  // namespace

int main(int argc, char** argv) {
  if (skip_without_gpu()) return kSkip;
  const std::string model_dir = INGINE_MODEL_DIR;
  if (!std::ifstream(model_dir + "/config.json")) {
    std::printf("skip: no model at %s\n", model_dir.c_str());
    return kSkip;
  }
  const bool kv_fp8 = argc > 1 && std::string(argv[1]) == "fp8";
  RuntimeOptions opt;
  opt.model_dir = model_dir;
  const std::string abl = std::string(INGINE_SOURCE_DIR) + "/../ablation/inkling-2-mini-abliterated.safetensors";
  if (std::ifstream(abl)) opt.ablation = abl;
  opt.max_ctx = 8192;
  opt.dense_fp8 = true;
  opt.kv_fp8 = kv_fp8;
  Runtime rt(opt);
  const cudaStream_t s = rt.stream();
  Prefiller pre(rt.weights(), rt.ablation(), 2048, s);
  std::printf("kv %s, ablation %s, prefill scratch %.0f MiB\n", kv_fp8 ? "fp8" : "bf16",
              opt.ablation.empty() ? "none" : "on", pre.device_bytes() / 1048576.0);

  std::ifstream gf(std::string(INGINE_GOLDEN_DIR) + "/model/abl_kv16_t5000.json");
  if (!gf) {
    std::printf("skip: no model golden abl_kv16_t5000.json\n");
    return kSkip;
  }
  const int N = 3000;
  std::vector<int32_t> ids = nlohmann::json::parse(gf)["prompt_ids"].get<std::vector<int32_t>>();
  INGINE_CHECK(int(ids.size()) >= N);
  ids.resize(N);

  int32_t* d_ids = nullptr;
  float* d_logits = nullptr;
  __nv_bfloat16* d_taps = nullptr;
  INGINE_TEST_CUDA(cudaMalloc(&d_ids, N * 4));
  INGINE_TEST_CUDA(cudaMalloc(&d_logits, size_t(UNPADDED_VOCAB) * 4));
  INGINE_TEST_CUDA(cudaMalloc(&d_taps, size_t(2048) * NUM_TAPS * HIDDEN * 2));
  INGINE_TEST_CUDA(cudaMemcpy(d_ids, ids.data(), N * 4, cudaMemcpyHostToDevice));
  State& st = rt.state();

  auto prefill = [&](int upto, int chunk, std::vector<uint16_t>* last_taps) {
    reset_state(st, s);
    for (int p0 = 0; p0 < upto; p0 += chunk) {
      const int T = std::min(chunk, upto - p0);
      Tensor taps = rm(d_taps, DType::BF16, {T, NUM_TAPS, HIDDEN});
      pre.run(st, d_ids + p0, T, p0, d_logits, last_taps ? &taps : nullptr);
      if (last_taps && p0 + T == upto) {
        last_taps->resize(size_t(NUM_TAPS) * HIDDEN);
        INGINE_TEST_CUDA(cudaMemcpyAsync(last_taps->data(), d_taps + size_t(T - 1) * NUM_TAPS * HIDDEN,
                                         last_taps->size() * 2, cudaMemcpyDeviceToHost, s));
      }
    }
    INGINE_TEST_CUDA(cudaStreamSynchronize(s));
    INGINE_CHECK(st.length == upto);
    return download(d_logits, UNPADDED_VOCAB);
  };

  bool ok = true;
  std::vector<uint16_t> taps_a;
  const auto a = prefill(N, 2048, &taps_a);
  const auto b = prefill(N, 700, nullptr);
  ok &= compare("chunks 2048+952 vs 700s", a, b, 1e-2);

  prefill(N - 8, 2048, nullptr);
  const float* dl = nullptr;
  for (int i = N - 8; i < N; ++i) dl = rt.decode_logits(ids[i], i);
  INGINE_TEST_CUDA(cudaStreamSynchronize(s));
  const auto c = download(dl, UNPADDED_VOCAB);
  ok &= compare("prefill vs prefill + 8 decode steps", a, c, 1e-2);
  {
    std::vector<uint16_t> taps_c(size_t(NUM_TAPS) * HIDDEN);
    INGINE_TEST_CUDA(cudaMemcpy(taps_c.data(), rt.buffers(1).taps.data, taps_c.size() * 2, cudaMemcpyDeviceToHost));
    for (int k = 0; k < NUM_TAPS; ++k) {
      double num = 0, den = 0;
      for (int i = 0; i < HIDDEN; ++i) {
        const double x = bf16_val(taps_a[size_t(k) * HIDDEN + i]), y = bf16_val(taps_c[size_t(k) * HIDDEN + i]);
        num += (x - y) * (x - y), den += y * y;
      }
      const double rel = std::sqrt(num / std::max(den, 1e-30));
      const bool pass = rel < 0.25;
      std::printf("%s tap %d prefill vs decode: rms rel diff %.3e\n", pass ? "PASS" : "FAIL", k, rel);
      ok &= pass;
    }
  }
  std::printf("%s\n", ok ? "ALL PASS" : "SOME FAILED");
  return ok ? 0 : 1;
}
