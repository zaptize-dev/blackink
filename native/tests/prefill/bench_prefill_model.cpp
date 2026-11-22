// Prefill throughput with the real weights in the served profile (max_ctx 262144, FP8 dense, FP8 full caches):
// tokens/s of each 2048-token chunk up to 16K context (target: >= 3000 for the chunk ending at 16K) and, with
// INGINE_PREFILL_LONG=1, the average over a 255K prompt (target: >= 1000). Runs only with INGINE_BENCH=1.
#include "model/prefill.h"
#include "model/runtime.h"
#include "common/test_util.h"

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <string>
#include <vector>

using namespace ingine;
using namespace ingine::model;
using namespace ingine::test;

int main() {
  const char* on = std::getenv("INGINE_BENCH");
  if (!on || std::string(on) != "1") {
    std::printf("skip: set INGINE_BENCH=1 to run the prefill model benchmark\n");
    return kSkip;
  }
  if (skip_without_gpu()) return kSkip;
  const std::string model_dir = INGINE_MODEL_DIR;
  if (!std::ifstream(model_dir + "/config.json")) {
    std::printf("skip: no model at %s\n", model_dir.c_str());
    return kSkip;
  }
  const char* long_env = std::getenv("INGINE_PREFILL_LONG");
  const bool longrun = long_env && std::string(long_env) == "1";
  RuntimeOptions opt;
  opt.model_dir = model_dir;
  const std::string abl = std::string(INGINE_SOURCE_DIR) + "/../ablation/inkling-2-mini-abliterated.safetensors";
  if (std::ifstream(abl)) opt.ablation = abl;
  opt.max_ctx = 262144;
  opt.dense_fp8 = true;
  opt.kv_fp8 = true;
  Runtime rt(opt);
  const cudaStream_t s = rt.stream();
  const int chunk = 2048;
  Prefiller pre(rt.weights(), rt.ablation(), chunk, s);
  const int N = longrun ? 255 * 1024 : 16384;
  std::vector<int32_t> ids(N);
  Rng rng(3);
  for (auto& t : ids) t = int32_t(rng.next() % 200000);
  int32_t* d_ids = nullptr;
  float* d_logits = nullptr;
  INGINE_TEST_CUDA(cudaMalloc(&d_ids, size_t(N) * 4));
  INGINE_TEST_CUDA(cudaMalloc(&d_logits, size_t(UNPADDED_VOCAB) * 4));
  INGINE_TEST_CUDA(cudaMemcpy(d_ids, ids.data(), size_t(N) * 4, cudaMemcpyHostToDevice));
  State& st = rt.state();

  // warm-up chunk (cuBLASLt heuristics, block tables)
  reset_state(st, s);
  pre.run(st, d_ids, chunk, 0, d_logits, nullptr);
  INGINE_TEST_CUDA(cudaStreamSynchronize(s));

  reset_state(st, s);
  cudaEvent_t e0, e1;
  cudaEventCreate(&e0), cudaEventCreate(&e1);
  double total_ms = 0;
  for (int p0 = 0; p0 < N; p0 += chunk) {
    const int T = std::min(chunk, N - p0);
    cudaEventRecord(e0, s);
    pre.run(st, d_ids + p0, T, p0, d_logits, nullptr);
    cudaEventRecord(e1, s);
    INGINE_TEST_CUDA(cudaEventSynchronize(e1));
    float ms;
    cudaEventElapsedTime(&ms, e0, e1);
    total_ms += ms;
    if (!longrun || p0 + T == N || (p0 / chunk) % 16 == 0)
      std::printf("chunk ending at %6d: %8.1f ms  %7.0f tok/s\n", p0 + T, ms, T / ms * 1e3);
  }
  std::printf("total %d tokens: %.2f s, average %.0f tok/s\n", N, total_ms / 1e3, N / total_ms * 1e3);
  return 0;
}
