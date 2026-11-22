// One decoder layer through the weights loader (weights.load_layer + the decoder's derived tensors) against
// the checkpoint: layers 5 (full, NVFP4), 6 (loop, NVFP4) and 20 (sliding, MXFP8), with FP8 dense rows.
// Needs the model directory and about 1.1 GB of GPU memory (one layer at a time); skips (77) otherwise.
#include "common/test_util.h"

#include "core/safetensors.h"
#include "kernels/dense/dense.h"
#include "model/runtime.h"

#include <cuda_runtime.h>

#include <cmath>
#include <cstring>
#include <filesystem>
#include <string>

using namespace ingine;
using namespace ingine::model;
using ingine::test::bf16_to_float;

namespace {

std::vector<float> host_values(const SafetensorsModel& m, const std::string& name) {
  const auto ref = m.at(name);
  const auto raw = m.read(name);
  std::vector<float> out(std::size_t(ref->numel()));
  for (std::size_t i = 0; i < out.size(); ++i) {
    if (ref->dtype == "BF16") {
      __nv_bfloat16 b;
      std::memcpy(&b, raw.data() + 2 * i, 2);
      out[i] = bf16_to_float(b);
    } else {
      std::memcpy(&out[i], raw.data() + 4 * i, 4);
    }
  }
  return out;
}

template <class T>
std::vector<T> download(const void* p, std::size_t n) {
  std::vector<T> h(n);
  INGINE_TEST_CUDA(cudaMemcpy(h.data(), p, n * sizeof(T), cudaMemcpyDeviceToHost));
  return h;
}

bool check_layer(const SafetensorsModel& m, int l, cudaStream_t stream) {
  std::vector<DeviceBuffer> storage;
  const Layer layer = load_layer(INGINE_MODEL_DIR, l, true, stream, storage);
  const std::string p = "model.llm.layers." + std::to_string(l) + ".";
  bool ok = true;
  auto expect = [&](bool c, const std::string& what) {
    if (!c) std::printf("FAIL layer %d: %s\n", l, what.c_str());
    ok &= c;
  };
  expect(layer.full == is_full_layer(l) && layer.loop == is_loop_layer(l), "flags");
  expect(layer.experts.fmt == (l < 20 ? ExpertFormat::NVFP4 : ExpertFormat::MXFP8), "expert format");
  const int R = proj_rows(layer.full), H = HIDDEN;
  expect(layer.wproj.fp8() && layer.wproj.rows() == R && layer.wproj.cols() == H, "wproj shape");
  expect(layer.wo.fp8() && layer.wo.rows() == H && layer.wo.cols() == 3072, "wo shape");
  expect(layer.wk.w.data == static_cast<char*>(layer.wproj.w.data) + std::size_t(SEG_BEGIN[SEG_K]) * H &&
             layer.wk.scale.data == static_cast<float*>(layer.wproj.scale.data) + SEG_BEGIN[SEG_K],
         "wk is a row view of wproj");

  // FP8 rows: dequantized wproj reproduces the BF16 checkpoint within E4M3 precision
  DeviceBuffer deq(std::size_t(R) * H * 2);
  kernels::dense::dequant_rows(layer.wproj.w.data, static_cast<const float*>(layer.wproj.scale.data), R, H,
                               static_cast<__nv_bfloat16*>(deq.p), stream);
  INGINE_TEST_CUDA(cudaStreamSynchronize(stream));
  const auto got = download<__nv_bfloat16>(deq.p, std::size_t(R) * H);
  const auto scale = download<float>(layer.wproj.scale.data, R);
  const char* names[NUM_SEGS] = {"attn.wq_du.weight",        "attn.gate_attn.weight",     "attn.wr_du.weight",
                                 "attn.attn_sink_proj.weight", "attn.wk_dv.weight",       "attn.wv_dv.weight",
                                 "attn.k_sconv.weight_proj", "attn.v_sconv.weight_proj"};
  double worst = 0;
  for (int s = 0; s < num_segs(layer.full); ++s) {
    const auto w = host_values(m, p + names[s]);
    for (int r = 0; r < SEG_ROWS[s]; ++r) {
      const int row = SEG_BEGIN[s] + r;
      for (int k = 0; k < H; ++k) {
        const double ref = w[std::size_t(r) * H + k];
        const double d = std::fabs(bf16_to_float(got[std::size_t(row) * H + k]) - ref);
        // E4M3: 3 mantissa bits (relative 2^-4), subnormal step 2^-9 * scale, plus the BF16 rounding
        const double tol = std::fabs(ref) * (1.0 / 16 + 1.0 / 256) + scale[row] * std::ldexp(1.0, -9);
        worst = std::max(worst, d / tol);
      }
    }
  }
  expect(worst <= 1.0, "fp8 rows reproduce the checkpoint (worst err/tol " + std::to_string(worst) + ")");

  // exact copies
  auto same_f32 = [&](const Tensor& t, const std::string& name) {
    const auto want = host_values(m, name);
    const auto have = download<float>(t.data, want.size());
    expect(t.numel() == std::int64_t(want.size()) && have == want, name + " (fp32)");
  };
  same_f32(layer.router, p + "mlp.gate.weight");
  same_f32(layer.router_bias, p + "mlp.gate.bias");
  same_f32(layer.mlp_query, "model.llm.attn_res.mlp_queries." + std::to_string(l));
  same_f32(layer.attn_query, "model.llm.attn_res.attn_queries." + std::to_string(l - 1));
  expect(layer.attn_query.ne[1] == (layer.full ? 3 : 1), "attn_query rows");
  same_f32(layer.q_scale, p + "attn.q_l2_clip.scale");
  same_f32(layer.rel_proj, p + "attn.rel_logits_proj.proj");
  same_f32(layer.attn_conv, p + "attn_sconv.weight");
  if (layer.full) {
    same_f32(layer.k_static, p + "attn.k_sconv.static_w");
    const auto q = host_values(m, p + "q_input_norm.weight"), v = host_values(m, p + "v_input_norm.weight");
    const auto n3 = download<float>(layer.attn_norm3.data, std::size_t(3) * H);
    expect(std::equal(q.begin(), q.end(), n3.begin()) && std::equal(v.begin(), v.end(), n3.begin() + 2 * H),
           "attn_norm3 = (q, k, v) norms");
  } else {
    same_f32(layer.k_conv, p + "attn.k_sconv.weight");
  }
  {
    const auto want = host_values(m, p + "mlp.gate.weight");
    const auto have = download<__nv_bfloat16>(layer.router_bf16.data, want.size());
    bool eq = true;
    for (std::size_t i = 0; i < want.size(); ++i) eq &= bf16_to_float(have[i]) == want[i];
    expect(eq, "router_bf16 holds the checkpoint values");
  }
  if (layer.loop) {
    same_f32(layer.k_gate_forget, p + "controllers.k.forget_weight");
    same_f32(layer.mlp_gate_gain, p + "controllers.mlp.gain_weight");
    const auto want = host_values(m, p + "controllers.v.down_weight");
    const auto have = download<__nv_bfloat16>(layer.v_gate_down.data, want.size());
    bool eq = true;
    for (std::size_t i = 0; i < want.size(); ++i) eq &= bf16_to_float(have[i]) == want[i];
    expect(eq && layer.v_gate_down.data == static_cast<char*>(layer.qkv_gate_down.data) + std::size_t(2) * 64 * 2 * H * 2,
           "v_gate_down is gate 2 of the qkv stack");
  }
  // experts: raw bytes, NVFP4 factors = amax / denominator
  {
    const auto ref = m.at(p + "mlp.experts.w2_weight_scale");
    const auto raw = m.read(p + "mlp.experts.w2_weight_scale");
    const auto have = download<std::uint8_t>(layer.experts.w2_scale.data, raw.size());
    expect(std::memcmp(have.data(), raw.data(), raw.size()) == 0 && ref->bytes == raw.size(), "w2 scales");
    if (layer.experts.fmt == ExpertFormat::NVFP4) {
      const auto amax = host_values(m, p + "mlp.experts.w13_weight_amax");
      const auto den = host_values(m, p + "mlp.experts.w13_weight_outer_scale_denominator");
      const auto f = download<float>(layer.experts.w13_factor.data, 128);
      bool eq = true;
      for (int e = 0; e < 128; ++e) eq &= f[e] == amax[e] / den[0];
      expect(eq, "w13 factors");
    }
  }
  std::printf("%s layer %d (%s, %s experts)\n", ok ? "PASS" : "FAIL", l, layer.full ? "full" : "sliding",
              layer.experts.fmt == ExpertFormat::NVFP4 ? "nvfp4" : "mxfp8");
  return ok;
}

}  // namespace

int main() {
  if (ingine::test::skip_without_gpu()) return ingine::test::kSkip;
  if (!std::filesystem::exists(std::filesystem::path(INGINE_MODEL_DIR) / "model.safetensors.index.json")) {
    std::printf("skip: no model at %s\n", INGINE_MODEL_DIR);
    return ingine::test::kSkip;
  }
  std::size_t free = 0, total = 0;
  INGINE_TEST_CUDA(cudaMemGetInfo(&free, &total));
  if (free < (std::size_t(3) << 29)) {
    std::printf("skip: %.2f GB of GPU memory free, need 1.5\n", double(free) / (1 << 30));
    return ingine::test::kSkip;
  }
  cudaStream_t stream;
  INGINE_TEST_CUDA(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
  const SafetensorsModel m(INGINE_MODEL_DIR);
  bool ok = true;
  for (int l : {5, 6, 20}) ok &= check_layer(m, l, stream);
  INGINE_TEST_CUDA(cudaStreamDestroy(stream));
  return ok ? 0 : 1;
}
