// Prefill timing on the real model at chosen (chunk tokens, context) points, for nsys kernel breakdowns:
//   prefill_profile T@ctx [T@ctx ...]
// Loads the weights (dense FP8, FP8 full caches, ablation) without the decode runtime. Context is simulated: the
// rings hold positions ctx-1024..ctx-1 (zero keys/values) and the full caches are read up to ctx, so every
// kernel does the work of a real append at that context; tokens are real text (the t5000 golden prompt), so the
// expert routing is realistic. Each point: one warm-up run, then 3 timed runs inside the NVTX range p<T>_<ctx>
// (nsys stats --filter-nvtx p<T>_<ctx> for its kernels). Runs only with INGINE_BENCH=1.
#include "model/prefill.h"
#include "model/runtime.h"
#include "common/test_util.h"

#include <nlohmann/json.hpp>
#include <nvtx3/nvToolsExt.h>

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <string>
#include <utility>
#include <vector>

using namespace ingine;
using namespace ingine::model;
using namespace ingine::test;

int main(int argc, char** argv) {
  const char* on = std::getenv("INGINE_BENCH");
  if (!on || std::string(on) != "1") {
    std::printf("skip: set INGINE_BENCH=1 to run the prefill profile\n");
    return kSkip;
  }
  if (skip_without_gpu()) return kSkip;
  const std::string model_dir = INGINE_MODEL_DIR;
  std::ifstream gf(std::string(INGINE_GOLDEN_DIR) + "/model/abl_kv16_t5000.json");
  if (!std::ifstream(model_dir + "/config.json") || !gf) {
    std::printf("skip: no model or golden prompt\n");
    return kSkip;
  }
  std::vector<std::pair<int, int>> points;  // (T, ctx)
  for (int i = 1; i < argc; ++i) {
    int T = 0, ctx = 0;
    if (std::sscanf(argv[i], "%d@%d", &T, &ctx) != 2 || T < 1 || ctx < 0) {
      std::fprintf(stderr, "bad point %s (want T@ctx)\n", argv[i]);
      return 2;
    }
    points.emplace_back(T, ctx);
  }
  if (points.empty()) points = {{64, 20480}, {256, 20480}, {1024, 20480}, {2048, 20480}, {2048, 131072}};
  int max_ctx = 0, max_t = 0;
  for (auto [T, ctx] : points) max_ctx = std::max(max_ctx, ctx + T), max_t = std::max(max_t, T);
  max_ctx = (max_ctx + 1023) / 1024 * 1024;

  cudaStream_t s;
  INGINE_TEST_CUDA(cudaStreamCreateWithFlags(&s, cudaStreamNonBlocking));
  const Weights w = load_weights(model_dir, true, s);
  const std::string abl_path = std::string(INGINE_SOURCE_DIR) + "/../ablation/inkling-2-mini-abliterated.safetensors";
  const Ablation abl = load_ablation(std::ifstream(abl_path) ? abl_path : std::string());
  State st = make_state(max_ctx, true);
  const int chunk = std::max(2048, max_t);
  Prefiller pre(w, abl, chunk, s);

  std::vector<int32_t> text = nlohmann::json::parse(gf)["prompt_ids"].get<std::vector<int32_t>>();
  std::vector<int32_t> ids(chunk);
  for (int i = 0; i < chunk; ++i) ids[i] = text[i % text.size()];
  int32_t* d_ids = nullptr;
  float* d_logits = nullptr;
  INGINE_TEST_CUDA(cudaMalloc(&d_ids, size_t(chunk) * 4));
  INGINE_TEST_CUDA(cudaMalloc(&d_logits, size_t(UNPADDED_VOCAB) * 4));
  INGINE_TEST_CUDA(cudaMemcpy(d_ids, ids.data(), size_t(chunk) * 4, cudaMemcpyHostToDevice));

  cudaEvent_t e0, e1;
  cudaEventCreate(&e0), cudaEventCreate(&e1);
  std::vector<int64_t> rp(size_t(NUM_SLIDING_VISITS) * WINDOW);
  for (auto [T, ctx] : points) {
    for (int r = 0; r < NUM_SLIDING_VISITS; ++r)
      for (int slot = 0; slot < WINDOW; ++slot) {
        // the position ctx-1024 .. ctx-1 that maps to this slot (or -1 before position 0)
        const int p = ctx - WINDOW + ((slot - (ctx - WINDOW)) % WINDOW + WINDOW) % WINDOW;
        rp[size_t(r) * WINDOW + slot] = p >= 0 ? p : -1;
      }
    auto run = [&] {
      INGINE_TEST_CUDA(cudaMemcpyAsync(st.ring_pos.data, rp.data(), rp.size() * 8, cudaMemcpyHostToDevice, s));
      pre.run(st, d_ids, T, ctx, d_logits, nullptr);
    };
    run();
    INGINE_TEST_CUDA(cudaStreamSynchronize(s));
    const std::string range = "p" + std::to_string(T) + "_" + std::to_string(ctx);
    nvtxRangePushA(range.c_str());
    float total = 0;
    for (int rep = 0; rep < 3; ++rep) {
      cudaEventRecord(e0, s);
      run();
      cudaEventRecord(e1, s);
      INGINE_TEST_CUDA(cudaEventSynchronize(e1));
      float ms;
      cudaEventElapsedTime(&ms, e0, e1);
      total += ms;
    }
    nvtxRangePop();
    std::printf("T=%5d ctx=%6d: %8.2f ms  %7.0f tok/s\n", T, ctx, total / 3, T / (total / 3) * 1e3);
  }
  return 0;
}
