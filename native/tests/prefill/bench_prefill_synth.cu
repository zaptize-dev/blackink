// Whole-Prefiller timing without the checkpoint: every layer aliases one synthetic full-attention or sliding
// layer (random FP8 dense rows, NVFP4 experts shared by all layers, the FP8 unembedding aliasing the expert
// bytes), served profile otherwise (dense FP8, FP8 full caches). Prints tokens/s per 2048-token chunk up to
// 16K context and checks the logits stay finite. < 2 GB of GPU memory. Runs only with INGINE_BENCH=1.
#include "model/prefill.h"
#include "model/runtime.h"
#include "common/test_util.h"

#include <cuda_runtime.h>

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

using namespace ingine;
using namespace ingine::model;
using namespace ingine::test;

namespace {

__device__ uint32_t hash(size_t i, uint32_t seed) {
  uint32_t h = uint32_t(i) * 2654435761u ^ seed ^ uint32_t(i >> 32) * 40503u;
  h ^= h >> 15, h *= 2246822519u, h ^= h >> 13, h *= 3266489917u, h ^= h >> 16;
  return h;
}
__global__ void k_bf16(uint16_t* p, size_t n, uint32_t seed, float scale) {
  for (size_t i = blockIdx.x * size_t(blockDim.x) + threadIdx.x; i < n; i += size_t(gridDim.x) * blockDim.x)
    p[i] = __bfloat16_as_ushort(__float2bfloat16(((hash(i, seed) & 0xffffff) / 16777216.f - 0.5f) * 3.46f * scale));
}
__global__ void k_f32(float* p, size_t n, uint32_t seed, float scale, float offset) {
  for (size_t i = blockIdx.x * size_t(blockDim.x) + threadIdx.x; i < n; i += size_t(gridDim.x) * blockDim.x)
    p[i] = offset + ((hash(i, seed) & 0xffffff) / 16777216.f - 0.5f) * 3.46f * scale;
}
// E4M3 codes of magnitude ~[0.5, 2) with random sign / mantissa (exponent field 6..7)
__global__ void k_fp8(uint8_t* p, size_t n, uint32_t seed) {
  for (size_t i = blockIdx.x * size_t(blockDim.x) + threadIdx.x; i < n; i += size_t(gridDim.x) * blockDim.x) {
    const uint32_t h = hash(i, seed);
    p[i] = uint8_t((h & 0x80) | ((6 + ((h >> 8) & 1)) << 3) | ((h >> 12) & 7));
  }
}

struct Alloc {
  std::vector<DeviceBuffer>& storage;
  uint32_t seed = 1;
  void* raw(size_t bytes) {
    storage.emplace_back(bytes);
    return storage.back().p;
  }
  Tensor f32(std::initializer_list<int> shape, float scale, float offset = 0.f) {
    size_t n = 1;
    for (int d : shape) n *= size_t(d);
    auto* p = static_cast<float*>(raw(n * 4));
    k_f32<<<1024, 256>>>(p, n, seed++, scale, offset);
    return rm(p, DType::FP32, shape);
  }
  Tensor bf16(std::initializer_list<int> shape, float scale) {
    size_t n = 1;
    for (int d : shape) n *= size_t(d);
    auto* p = static_cast<uint16_t*>(raw(n * 2));
    k_bf16<<<1024, 256>>>(p, n, seed++, scale);
    return rm(p, DType::BF16, shape);
  }
  Dense fp8(int rows, int cols, float scale) {
    auto* p = static_cast<uint8_t*>(raw(size_t(rows) * cols));
    k_fp8<<<1024, 256>>>(p, size_t(rows) * cols, seed++);
    Dense d;
    d.w = rm(p, DType::FP8_E4M3FN, {rows, cols});
    d.scale = f32({rows}, 0.f, scale);
    return d;
  }
};

Layer make_layer(Alloc& a, bool full, const Experts& ex) {
  const int H = HIDDEN;
  Layer L;
  L.full = full;
  L.loop = true;
  L.q_norm = L.k_norm = L.v_norm = L.attn_norm = L.mlp_norm = a.f32({H}, 0.f, 1.f);
  L.wproj = a.fp8(proj_rows(full), H, 0.6f / 48);
  L.wo = a.fp8(H, HEADS * HEAD_DIM, 0.6f / 55);
  L.q_scale = a.f32({HEADS, HEAD_DIM}, 0.f, 1.f);
  L.r_scale = a.f32({HEADS, D_REL}, 0.f, 1.f);
  L.rel_proj = a.f32({D_REL, REL_EXTENT}, 0.1f);
  L.k_static = a.f32({CONV_K, KV_DIM}, 0.1f);
  L.v_static = a.f32({CONV_K, KV_DIM}, 0.1f);
  L.k_conv = a.f32({KV_DIM, CONV_K}, 0.1f);
  L.v_conv = a.f32({KV_DIM, CONV_K}, 0.1f);
  L.attn_conv = a.f32({H, CONV_K}, 0.1f);
  L.mlp_conv = a.f32({H, CONV_K}, 0.1f);
  L.router_bf16 = a.bf16({EXPERTS, H}, 0.02f);
  L.router_bias = a.f32({EXPERTS}, 0.f);
  L.attn_query = a.f32({full ? 3 : 1, H}, 0.02f);
  L.mlp_query = a.f32({1, H}, 0.02f);
  L.qkv_gate_down = a.bf16({3, GATE_RANK, 2 * H}, 0.02f);
  L.qkv_gate_forget16 = a.bf16({3, H, GATE_RANK}, 0.05f);
  L.qkv_gate_gain16 = a.bf16({3, H, GATE_RANK}, 0.05f);
  L.mlp1_gate_down = a.bf16({1, GATE_RANK, 2 * H}, 0.02f);
  L.mlp1_gate_forget16 = a.bf16({1, H, GATE_RANK}, 0.05f);
  L.mlp1_gate_gain16 = a.bf16({1, H, GATE_RANK}, 0.05f);
  L.attn_norm3 = a.f32({3, H}, 0.f, 1.f);
  L.attn_norm_mix = full ? L.attn_norm3 : a.f32({1, H}, 0.f, 1.f);
  L.mlp_norm1 = a.f32({1, H}, 0.f, 1.f);
  L.experts = ex;
  return L;
}

}  // namespace

int main() {
  const char* on = std::getenv("INGINE_BENCH");
  if (!on || std::string(on) != "1") {
    std::printf("skip: set INGINE_BENCH=1 to run the synthetic prefill benchmark\n");
    return kSkip;
  }
  if (skip_without_gpu()) return kSkip;
  const int H = HIDDEN, chunk = 2048, N = 16384;
  cudaStream_t s;
  INGINE_TEST_CUDA(cudaStreamCreateWithFlags(&s, cudaStreamNonBlocking));
  Weights w;
  Alloc a{w.storage};
  Experts ex;
  ex.fmt = ExpertFormat::NVFP4;
  {
    // one allocation for all expert tensors, also aliased by the FP8 unembedding (needs 461 MB); E4M3-valid
    // bytes everywhere (also valid NVFP4 code pairs and block scales)
    const size_t n13 = size_t(EXPERTS) * 2 * INTER * H / 2, s13 = n13 / 8, n2 = size_t(EXPERTS) * H * INTER / 2,
                 s2 = n2 / 8;
    auto* p = static_cast<uint8_t*>(a.raw(n13 + s13 + n2 + s2));
    k_fp8<<<4096, 256>>>(p, n13 + s13 + n2 + s2, a.seed++);
    ex.w13 = rm(p, DType::U8, {EXPERTS, 2 * INTER, H / 2});
    ex.w13_scale = rm(p + n13, DType::FP8_E4M3FN, {EXPERTS, 2 * INTER, H / 16});
    ex.w2 = rm(p + n13 + s13, DType::U8, {EXPERTS, H, INTER / 2});
    ex.w2_scale = rm(p + n13 + s13 + n2, DType::FP8_E4M3FN, {EXPERTS, H, INTER / 16});
    ex.w13_factor = a.f32({EXPERTS}, 0.f, 0.02f);
    ex.w2_factor = a.f32({EXPERTS}, 0.f, 0.02f);
    if (n13 + s13 + n2 + s2 < size_t(UNPADDED_VOCAB) * H) return 1;
  }
  const Layer full = make_layer(a, true, ex), sliding = make_layer(a, false, ex);
  for (int l = 0; l < LAYERS; ++l) {
    Layer L = is_full_layer(l) ? full : sliding;
    L.idx = l;
    L.loop = is_loop_layer(l);
    w.layers.push_back(L);
  }
  w.embed = a.bf16({4096, H}, 1.f);  // token ids stay below 4096
  w.embed_norm = a.f32({H}, 0.f, 1.f);
  w.final_query = a.f32({1, H}, 0.02f);
  w.final_norm = a.f32({H}, 0.f, 1.f);
  w.unembed.w = rm(ex.w13.data, DType::FP8_E4M3FN, {UNPADDED_VOCAB, H});  // any 461 MB of E4M3-like bytes
  w.unembed.scale = a.f32({UNPADDED_VOCAB}, 0.f, 1e-3f);
  w.message_queries = a.f32({LOOP_END - LOOP_START, 4, H}, 0.02f);
  w.dense_fp8 = true;
  const Ablation abl = load_ablation({});
  State st = make_state(N, true);
  INGINE_TEST_CUDA(cudaDeviceSynchronize());

  Prefiller pre(w, abl, chunk, s);
  size_t free_b = 0, total_b = 0;
  cudaMemGetInfo(&free_b, &total_b);
  std::printf("prefill scratch %.0f MiB\n", pre.device_bytes() / 1048576.0);
  std::vector<int32_t> ids(N);
  Rng rng(9);
  for (auto& t : ids) t = int32_t(rng.next() % 4096);
  int32_t* d_ids = nullptr;
  float* d_logits = nullptr;
  INGINE_TEST_CUDA(cudaMalloc(&d_ids, size_t(N) * 4));
  INGINE_TEST_CUDA(cudaMalloc(&d_logits, size_t(UNPADDED_VOCAB) * 4));
  INGINE_TEST_CUDA(cudaMemcpy(d_ids, ids.data(), size_t(N) * 4, cudaMemcpyHostToDevice));

  reset_state(st, s);
  pre.run(st, d_ids, chunk, 0, d_logits, nullptr);  // warm-up
  INGINE_TEST_CUDA(cudaStreamSynchronize(s));
  reset_state(st, s);
  cudaEvent_t e0, e1;
  cudaEventCreate(&e0), cudaEventCreate(&e1);
  double total = 0;
  bool finite = true;
  for (int p0 = 0; p0 < N; p0 += chunk) {
    cudaEventRecord(e0, s);
    pre.run(st, d_ids + p0, chunk, p0, d_logits, nullptr);
    cudaEventRecord(e1, s);
    INGINE_TEST_CUDA(cudaEventSynchronize(e1));
    float ms;
    cudaEventElapsedTime(&ms, e0, e1);
    total += ms;
    std::vector<float> lg(UNPADDED_VOCAB);
    INGINE_TEST_CUDA(cudaMemcpy(lg.data(), d_logits, lg.size() * 4, cudaMemcpyDeviceToHost));
    size_t bad = 0;
    for (float x : lg) bad += !std::isfinite(x);
    finite &= bad == 0;
    std::printf("chunk ending at %5d: %7.1f ms  %6.0f tok/s  (logits[0..2] %.3f %.3f %.3f, %zu non-finite)\n",
                p0 + chunk, ms, chunk / ms * 1e3, lg[0], lg[1], lg[2], bad);
  }
  std::printf("average %.0f tok/s over %d tokens; %s\n", N / total * 1e3, N, finite ? "logits finite" : "NON-FINITE");
  return finite ? 0 : 1;
}
