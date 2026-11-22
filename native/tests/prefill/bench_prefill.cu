// Prefill kernel timings with real layer weights (dense FP8 rows, NVFP4 and MXFP8 experts), rotating over layers
// and caches so L2 does not serve the weights / keys from an earlier repetition. Per chunk size T: grouped MoE
// (router + gate/up + down on real routing of random hidden rows), dense projections (proj + wo), relative
// logits, sliding attention, full attention over an FP8 cache at 20K context, and full attention at 131K for
// T = 2048. Prints ms, TFLOP/s of useful MMA work and GB/s of weight bytes. Needs the model (MODEL label) and
// ~5 GB of GPU memory. Runs only with INGINE_BENCH=1.
#include "kernels/dense/dense.h"
#include "kernels/prefill/prefill.h"
#include "model/runtime.h"
#include "common/test_util.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <functional>
#include <string>
#include <vector>

using namespace ingine;
using namespace ingine::model;
using namespace ingine::test;
namespace pk = ingine::kernels::prefill;

namespace {

__global__ void fill_bf16(uint16_t* p, size_t n, uint32_t seed, float scale) {
  for (size_t i = blockIdx.x * size_t(blockDim.x) + threadIdx.x; i < n; i += size_t(gridDim.x) * blockDim.x) {
    uint32_t h = uint32_t(i) * 2654435761u ^ seed;
    h ^= h >> 15, h *= 2246822519u, h ^= h >> 13;
    p[i] = __bfloat16_as_ushort(__float2bfloat16(((h & 0xffffff) / 16777216.f - 0.5f) * scale));
  }
}
__global__ void fill_e4m3(uint8_t* p, size_t n, uint32_t seed) {
  for (size_t i = blockIdx.x * size_t(blockDim.x) + threadIdx.x; i < n; i += size_t(gridDim.x) * blockDim.x) {
    uint32_t h = uint32_t(i) * 2654435761u ^ seed;
    h ^= h >> 15, h *= 2246822519u, h ^= h >> 13;
    p[i] = uint8_t((h & 0x80) | ((4 + ((h >> 8) & 3)) << 3) | ((h >> 12) & 7));  // |v| in [1/8, 1)
  }
}

template <class T>
T* dalloc(size_t n) {
  T* p = nullptr;
  INGINE_TEST_CUDA(cudaMalloc(&p, n * sizeof(T) + 256));
  INGINE_TEST_CUDA(cudaMemset(p, 0, n * sizeof(T) + 256));
  return p;
}

cudaStream_t g_stream;

// median of `reps` timings of f(rep) (each call gets its repetition index to rotate inputs)
float time_ms(const std::function<void(int)>& f, int reps = 6) {
  cudaEvent_t a, b;
  cudaEventCreate(&a), cudaEventCreate(&b);
  f(0);
  INGINE_TEST_CUDA(cudaStreamSynchronize(g_stream));
  std::vector<float> t;
  for (int i = 0; i < reps; ++i) {
    cudaEventRecord(a, g_stream);
    f(i + 1);
    cudaEventRecord(b, g_stream);
    cudaEventSynchronize(b);
    float ms;
    cudaEventElapsedTime(&ms, a, b);
    t.push_back(ms);
  }
  INGINE_TEST_CUDA(cudaGetLastError());
  std::sort(t.begin(), t.end());
  cudaEventDestroy(a), cudaEventDestroy(b);
  return t[t.size() / 2];
}

const __nv_bfloat16* bf(const void* p) { return static_cast<const __nv_bfloat16*>(p); }

}  // namespace

int main(int argc, char** argv) {
  const char* on = std::getenv("INGINE_BENCH");
  if (!on || std::string(on) != "1") {
    std::printf("skip: set INGINE_BENCH=1 to run the prefill benchmark\n");
    return kSkip;
  }
  if (skip_without_gpu()) return kSkip;
  const std::string model_dir = INGINE_MODEL_DIR;
  if (!std::ifstream(model_dir + "/config.json")) {
    std::printf("skip: no model at %s\n", model_dir.c_str());
    return kSkip;
  }
  // sections: moe dense attn (default all)
  std::string only = argc > 1 ? argv[1] : "all";
  auto want = [&](const char* s) { return only == "all" || only.find(s) != std::string::npos; };
  INGINE_TEST_CUDA(cudaStreamCreateWithFlags(&g_stream, cudaStreamNonBlocking));
  const cudaStream_t s = g_stream;
  const int H = HIDDEN, maxT = 2048;
  std::vector<int> Ts = {64, 256, 1024, 2048};
  if (const char* ts = std::getenv("INGINE_BENCH_T")) Ts = {std::atoi(ts)};  // one chunk size (profiling)

  std::vector<DeviceBuffer> storage;
  std::vector<Layer> nv;  // NVFP4 layers (sliding 1, 2, 3 and full 5)
  for (int l : {1, 2, 3, 5}) nv.push_back(load_layer(model_dir, l, true, s, storage));
  std::vector<Layer> mx;
  if (want("moe")) mx.push_back(load_layer(model_dir, 20, true, s, storage));
  INGINE_TEST_CUDA(cudaStreamSynchronize(s));
  pk::Gemm gemm;

  auto* x = dalloc<uint16_t>(size_t(maxT) * H);
  fill_bf16<<<1024, 256, 0, s>>>(x, size_t(maxT) * H, 1, 3.4f);

  if (want("moe")) {
    auto* logits = dalloc<float>(size_t(maxT) * 128);
    auto* idx = dalloc<int>(size_t(maxT) * 8);
    auto* wts = dalloc<float>(size_t(maxT) * 8);
    pk::MoeWork w{dalloc<int>(128), dalloc<int>(1), dalloc<int4>(pk::moe_tiles_capacity(maxT)), dalloc<int>(size_t(maxT) * 8),
                  dalloc<int>(size_t(maxT) * 8)};
    auto* act = dalloc<uint16_t>(size_t(maxT) * 8 * 1152);
    auto* down = dalloc<uint16_t>(size_t(maxT) * 8 * H);
    auto route = [&](const Layer& L, int T) {
      gemm.run(bf(x), H, bf(L.router_bf16.data), H, logits, 128, true, T, 128, H, s);
      cudaMemsetAsync(w.counts, 0, 512, s);
      pk::moe_topk(logits, static_cast<const float*>(L.router_bias.data), T, idx, wts, w.counts, s);
      pk::moe_route(idx, T, w, s);
    };
    auto experts = [](const Layer& L) {
      const Experts& e = L.experts;
      return pk::MoeExperts{e.fmt == ExpertFormat::NVFP4 ? 0 : 1, e.w13.data, e.w13_scale.data,
                            static_cast<const float*>(e.w13_factor.data), e.w2.data, e.w2_scale.data,
                            static_cast<const float*>(e.w2_factor.data)};
    };
    for (auto* set : {&nv, &mx}) {
      const bool nvfp4 = set == &nv;
      // per layer: route once (routing differs per layer), then time gate/up and down rotating over the layers
      for (int T : Ts) {
        std::vector<int> touched(set->size());
        // each layer's routing lives in its own work buffers? Routing is cheap: re-route inside the timed loop
        // would add its time; instead time route+gemm and route alone.
        const float t_route = time_ms([&](int r) { route((*set)[r % set->size()], T); });
        const float t_up = time_ms([&](int r) {
          const Layer& L = (*set)[r % set->size()];
          route(L, T);
          pk::moe_gate_up(experts(L), bf(x), T, w, reinterpret_cast<__nv_bfloat16*>(act), s);
        }) - t_route;
        const float t_down = time_ms([&](int r) {
          const Layer& L = (*set)[r % set->size()];
          route(L, T);
          pk::moe_down(experts(L), bf(act), T, w, reinterpret_cast<__nv_bfloat16*>(down), s);
        }) - t_route;
        // experts touched by the last routing
        std::vector<int> counts(128);
        INGINE_TEST_CUDA(cudaMemcpy(counts.data(), w.counts, 512, cudaMemcpyDeviceToHost));
        const int hit = int(std::count_if(counts.begin(), counts.end(), [](int c) { return c > 0; }));
        const double per_expert13 = nvfp4 ? 2304.0 * 1152 * (1 + 1.0 / 8) : 2304.0 * 2304 * (1 + 1.0 / 32);
        const double per_expert2 = per_expert13 / 2;
        const double fu = 2.0 * T * 8 * 2304 * 2304, fd = fu / 2;
        std::printf("moe %s T=%4d: route %.3f ms | gate_up %.3f ms %6.1f TFLOP/s %6.0f GB/s | down %.3f ms %6.1f TFLOP/s "
                    "%6.0f GB/s | %d experts\n",
                    nvfp4 ? "nvfp4" : "mxfp8", T, t_route, t_up, fu / t_up / 1e9, hit * per_expert13 / t_up / 1e6,
                    t_down, fd / t_down / 1e9, hit * per_expert2 / t_down / 1e6, hit);
      }
    }
  }

  if (want("dense")) {
    auto* scratch = dalloc<uint16_t>(size_t(PROJ_ROWS_FULL) * H);
    auto* proj = dalloc<uint16_t>(size_t(maxT) * PROJ_ROWS_FULL);
    auto* wo_out = dalloc<uint16_t>(size_t(maxT) * H);
    auto* attn = dalloc<uint16_t>(size_t(maxT) * 3072);
    fill_bf16<<<1024, 256, 0, s>>>(attn, size_t(maxT) * 3072, 2, 2.f);
    for (int T : Ts) {
      const float t = time_ms([&](int r) {
        const Layer& L = nv[r % 3];  // sliding layers: single-input projection
        kernels::dense::dequant_rows(L.wproj.w.data, static_cast<const float*>(L.wproj.scale.data), PROJ_ROWS_SLIDING, H,
                                     reinterpret_cast<__nv_bfloat16*>(scratch), s);
        gemm.run(bf(x), H, bf(scratch), H, proj, PROJ_ROWS_SLIDING, false, T, PROJ_ROWS_SLIDING, H, s);
        kernels::dense::dequant_rows(L.wo.w.data, static_cast<const float*>(L.wo.scale.data), H, 3072,
                                     reinterpret_cast<__nv_bfloat16*>(scratch), s);
        gemm.run(bf(attn), 3072, bf(scratch), 3072, wo_out, H, false, T, H, 3072, s);
      });
      const double flop = 2.0 * T * H * (PROJ_ROWS_SLIDING + 3072), bytes = double(H) * (PROJ_ROWS_SLIDING + 3072);
      std::printf("dense T=%4d: dequant + cuBLAS %.3f ms %6.1f TFLOP/s %6.0f GB/s (FP8 weight bytes)\n", T, t,
                  flop / t / 1e9, bytes / t / 1e6);
    }
  }

  if (want("attn")) {
    auto* q = dalloc<uint16_t>(size_t(maxT) * 3072);
    auto* rel16 = dalloc<uint16_t>(size_t(maxT) * 24 * 16);
    auto* rel = dalloc<uint16_t>(size_t(maxT) * 24 * 1024);
    auto* proj = dalloc<uint16_t>(size_t(maxT) * PROJ_ROWS_FULL);
    auto* out = dalloc<uint16_t>(size_t(maxT) * 3072);
    fill_bf16<<<1024, 256, 0, s>>>(q, size_t(maxT) * 3072, 3, 4.f);
    fill_bf16<<<1024, 256, 0, s>>>(rel16, size_t(maxT) * 24 * 16, 4, 4.f);
    fill_bf16<<<1024, 256, 0, s>>>(rel, size_t(maxT) * 24 * 1024, 5, 2.f);
    fill_bf16<<<1024, 256, 0, s>>>(proj, size_t(maxT) * PROJ_ROWS_FULL, 6, 2.f);
    pk::AttnParams p;
    p.q = bf(q), p.rel = bf(rel), p.proj = bf(proj), p.ld = PROJ_ROWS_FULL, p.out = reinterpret_cast<__nv_bfloat16*>(out);
    // rel_logits
    for (int T : Ts) {
      const float t = time_ms([&](int r) {
        pk::rel_logits(bf(rel16), static_cast<const float*>(nv[r % 3].rel_proj.data), T,
                       reinterpret_cast<__nv_bfloat16*>(rel), s);
      });
      std::printf("rel_logits T=%4d: %.3f ms\n", T, t);
    }
    // sliding: ring (full) + chunk rows, three copies rotating
    {
      const int L = 1024 + maxT, kr = (L + 63) / 64 * 64;
      std::vector<uint16_t*> kr_(3), vr_(3);
      std::vector<int*> kp_(3);
      std::vector<int4*> bd_(3);
      for (int i = 0; i < 3; ++i) {
        kr_[i] = dalloc<uint16_t>(size_t(kr) * 1024), vr_[i] = dalloc<uint16_t>(size_t(kr) * 1024);
        fill_bf16<<<1024, 256, 0, s>>>(kr_[i], size_t(kr) * 1024, 7 + i, 0.3f);
        fill_bf16<<<1024, 256, 0, s>>>(vr_[i], size_t(kr) * 1024, 17 + i, 2.f);
        kp_[i] = dalloc<int>(kr), bd_[i] = dalloc<int4>(kr / 64);
      }
      const int pos0 = 20480;
      for (int T : Ts) {
        std::vector<int> kpos(kr, -1);
        for (int j = 0; j < 1024; ++j) kpos[j] = pos0 - 1024 + j;
        for (int t = 0; t < T; ++t) kpos[1024 + t] = pos0 + t;
        for (int i = 0; i < 3; ++i) {
          INGINE_TEST_CUDA(cudaMemcpy(kp_[i], kpos.data(), kr * 4, cudaMemcpyHostToDevice));
          pk::key_tile_bounds(kp_[i], 1024 + T, bd_[i], s);
        }
        const float t = time_ms([&](int r) {
          pk::AttnParams a = p;
          const int i = r % 3;
          a.k = kr_[i], a.v = vr_[i], a.kv_fp8 = false, a.kpos = kp_[i], a.bounds = bd_[i], a.L = 1024 + T;
          a.T = T, a.pos0 = pos0, a.window = 1024;
          pk::attention(a, s);
        });
        const double flop = 24.0 * 512 * T * 1024;
        std::printf("attn sliding T=%4d: %.3f ms %6.1f TFLOP/s (window pairs)\n", T, t, flop / t / 1e9);
      }
    }
    // full FP8: 20K with three caches rotating, 131K with two
    for (int ctx : {20480, 131072}) {
      const int ncache = ctx > 32768 ? 2 : 3;
      std::vector<uint8_t*> kc(ncache), vc(ncache);
      for (int i = 0; i < ncache; ++i) {
        kc[i] = dalloc<uint8_t>(size_t(ctx + maxT) * 1024), vc[i] = dalloc<uint8_t>(size_t(ctx + maxT) * 1024);
        fill_e4m3<<<4096, 256, 0, s>>>(kc[i], size_t(ctx + maxT) * 1024, 30 + i);
        fill_e4m3<<<4096, 256, 0, s>>>(vc[i], size_t(ctx + maxT) * 1024, 40 + i);
      }
      for (int T : Ts) {
        if (ctx > 32768 && T != 2048) continue;
        const float t = time_ms([&](int r) {
          pk::AttnParams a = p;
          const int i = r % ncache;
          a.k = kc[i], a.v = vc[i], a.kv_fp8 = true, a.kpos = nullptr, a.bounds = nullptr, a.L = ctx + T;
          a.T = T, a.pos0 = ctx, a.window = 0;
          pk::attention(a, s);
        }, ctx > 32768 ? 3 : 6);
        const double flop = 24.0 * 512 * (double(T) * ctx + 0.5 * double(T) * T);
        std::printf("attn full fp8 ctx=%6d T=%4d: %.3f ms %6.1f TFLOP/s %6.0f GB/s (K/V bytes)\n", ctx, T, t,
                    flop / t / 1e9, 2.0 * (ctx + T) * 1024 / t / 1e6);
      }
      for (int i = 0; i < ncache; ++i) cudaFree(kc[i]), cudaFree(vc[i]);
    }
  }
  return 0;
}
