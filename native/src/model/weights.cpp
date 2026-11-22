// Weights in the decoder's device layouts, read from the Hugging Face checkpoint: weights.load / load_layer,
// plus fast.FastDecoder._concat_projections (one [R, H] projection matrix per layer, the per-projection
// matrices are row views of it), _stack_layer_params (stacked norms and gate weights) and _fp8_dense
// (wproj, wo and the unembedding as FP8 rows + FP32 row scales, the BF16 copies freed), and
// model.Ablation.load / none.
//
// Large tensors are uploaded straight from the file with the core DeviceUploader; small per-channel
// parameters are read on the host, widened to FP32 (exact from BF16) and copied.
#include "model/runtime.h"

#include "core/device.h"
#include "core/safetensors.h"
#include "kernels/dense/dense.h"
#include "kernels/moe/moe.h"

#include <nlohmann/json.hpp>

#include <cmath>
#include <cstring>
#include <fstream>
#include <stdexcept>
#include <string>

namespace ingine::model {
namespace {

namespace fs = std::filesystem;

float bf16_bits_to_float(std::uint16_t b) {
  const std::uint32_t w = std::uint32_t(b) << 16;
  float f;
  std::memcpy(&f, &w, 4);
  return f;
}

std::uint16_t float_to_bf16_bits(float f) {  // round to nearest even (torch .to(bfloat16))
  std::uint32_t w;
  std::memcpy(&w, &f, 4);
  if ((w & 0x7fffffffu) > 0x7f800000u) return std::uint16_t((w >> 16) | 0x40);
  return std::uint16_t((w + 0x7fffu + ((w >> 16) & 1u)) >> 16);
}

// Checkpoint values widened to FP32 on the host (BF16 -> FP32 exact, F32 as is).
std::vector<float> host_f32(const SafetensorsModel& m, const std::string& name) {
  const SafetensorsRef ref = m.at(name);
  const std::vector<std::byte> raw = m.read(name);
  const std::size_t n = std::size_t(ref->numel());
  std::vector<float> out(n);
  if (ref->dtype == "F32") {
    std::memcpy(out.data(), raw.data(), n * 4);
  } else if (ref->dtype == "BF16") {
    for (std::size_t i = 0; i < n; ++i) {
      std::uint16_t b;
      std::memcpy(&b, raw.data() + 2 * i, 2);
      out[i] = bf16_bits_to_float(b);
    }
  } else {
    throw std::runtime_error("weights: " + name + " has unsupported dtype " + ref->dtype);
  }
  return out;
}

class Loader {
 public:
  Loader(const SafetensorsModel& m, DeviceUploader& up, cudaStream_t stream, std::vector<DeviceBuffer>& storage)
      : m_(m), up_(up), stream_(stream), storage_(storage) {}

  void* alloc(std::size_t bytes) {
    storage_.emplace_back(bytes);
    return storage_.back().p;
  }

  Tensor upload_f32(const std::vector<float>& host, std::initializer_list<std::int32_t> shape) {
    void* d = alloc(host.size() * 4);
    CUDA_CHECK(cudaMemcpy(d, host.data(), host.size() * 4, cudaMemcpyHostToDevice));
    return rm(d, DType::FP32, shape);
  }

  Tensor upload_bf16(const std::vector<float>& host, std::initializer_list<std::int32_t> shape) {
    std::vector<std::uint16_t> b(host.size());
    for (std::size_t i = 0; i < host.size(); ++i) b[i] = float_to_bf16_bits(host[i]);
    void* d = alloc(b.size() * 2);
    CUDA_CHECK(cudaMemcpy(d, b.data(), b.size() * 2, cudaMemcpyHostToDevice));
    return rm(d, DType::BF16, shape);
  }

  // f32(name): checkpoint tensor as FP32 (weights.load_layer's `f32`), viewed with `shape`.
  Tensor f32(const std::string& name, std::initializer_list<std::int32_t> shape) {
    return upload_f32(host_f32(m_, name), shape);
  }

  // Queues the raw bytes of `name` (rows [0, rows) if rows >= 0) for upload to `dst`.
  void queue(const std::string& name, void* dst, std::int64_t rows = -1) {
    const SafetensorsRef ref = m_.at(name);
    UploadRequest r = upload_request(ref, dst);
    if (rows >= 0) r.bytes = ref->bytes / std::uint64_t(ref->shape[0]) * std::uint64_t(rows);
    pending_.push_back(r);
  }

  // Raw checkpoint tensor (any dtype) in new device memory, uploaded at flush().
  Tensor raw(const std::string& name) {
    const SafetensorsRef ref = m_.at(name);
    void* d = alloc(ref->bytes);
    queue(name, d);
    return ref->view(d);
  }

  void flush() {
    if (!pending_.empty()) up_.upload(pending_, stream_);
    pending_.clear();
  }

  const SafetensorsModel& model() const { return m_; }
  cudaStream_t stream() const { return stream_; }

 private:
  const SafetensorsModel& m_;
  DeviceUploader& up_;
  cudaStream_t stream_;
  std::vector<DeviceBuffer>& storage_;
  std::vector<UploadRequest> pending_;
};

std::array<ExpertFormat, LAYERS> expert_formats(const fs::path& model_dir) {
  std::ifstream in(model_dir / "hf_quant_config.json");
  if (!in) throw std::runtime_error("weights: cannot open " + (model_dir / "hf_quant_config.json").string());
  const nlohmann::json q = nlohmann::json::parse(in);
  std::array<ExpertFormat, LAYERS> fmt{};
  std::array<bool, LAYERS> seen{};
  const std::string suffix = "mlp.experts.w13_weight";
  for (const auto& group : q.at("model").at("parameter_groups")) {
    const std::string dt = group.at("compute_dtype").get<std::string>();
    for (const auto& t : group.at("tensors")) {
      const std::string name = t.get<std::string>();
      if (name.size() < suffix.size() || name.compare(name.size() - suffix.size(), suffix.size(), suffix) != 0)
        continue;
      // model.llm.layers.<l>.mlp.experts.w13_weight
      const std::size_t a = std::string("model.llm.layers.").size();
      const int l = std::stoi(name.substr(a, name.find('.', a) - a));
      if (l < 0 || l >= LAYERS) continue;
      if (dt == "nvfp4") fmt[l] = ExpertFormat::NVFP4;
      else if (dt == "mxfp8") fmt[l] = ExpertFormat::MXFP8;
      else throw std::runtime_error("weights: unsupported expert format " + dt);
      seen[l] = true;
    }
  }
  for (int l = 0; l < LAYERS; ++l)
    if (!seen[l]) throw std::runtime_error("weights: no expert format for layer " + std::to_string(l));
  return fmt;
}

// FP8 rows + scales of a BF16 matrix (dense fp8_rows), into new storage.
Dense to_fp8(Loader& ld, const Tensor& bf16, int rows, int cols) {
  void* q = ld.alloc(std::size_t(rows) * cols);
  float* s = static_cast<float*>(ld.alloc(std::size_t(rows) * 4));
  kernels::dense::fp8_rows(static_cast<const __nv_bfloat16*>(bf16.data), rows, cols, q, s, ld.stream());
  return Dense{rm(q, DType::FP8_E4M3FN, {rows, cols}), rm(s, DType::FP32, {rows})};
}

Dense row_view(const Dense& d, int begin, int end) {
  const int cols = d.cols();
  const std::size_t esize = d.fp8() ? 1 : 2;
  Dense v;
  v.w = rm(static_cast<char*>(d.w.data) + std::size_t(begin) * cols * esize, d.w.dtype, {end - begin, cols});
  if (d.fp8()) v.scale = rm(static_cast<float*>(d.scale.data) + begin, DType::FP32, {end - begin});
  return v;
}

Layer load_layer_into(Loader& ld, int l, ExpertFormat fmt, bool dense_fp8, bool full_nvfp4) {
  const SafetensorsModel& m = ld.model();
  const std::string p = "model.llm.layers." + std::to_string(l) + ".";
  Layer layer;
  layer.idx = l;
  layer.full = is_full_layer(l);
  layer.loop = is_loop_layer(l);
  const int H = HIDDEN;

  if (layer.full) {
    layer.q_norm = ld.f32(p + "q_input_norm.weight", {H});
    layer.k_norm = ld.f32(p + "k_input_norm.weight", {H});
    layer.v_norm = ld.f32(p + "v_input_norm.weight", {H});
  } else {
    layer.attn_norm = ld.f32(p + "attn_norm.weight", {H});
  }
  layer.mlp_norm = ld.f32(p + "mlp_norm.weight", {H});

  // projections: one [R, H] BF16 matrix in SEG order (fast._concat_projections)
  static const char* seg_names[NUM_SEGS] = {"attn.wq_du.weight",        "attn.gate_attn.weight",
                                            "attn.wr_du.weight",        "attn.attn_sink_proj.weight",
                                            "attn.wk_dv.weight",        "attn.wv_dv.weight",
                                            "attn.k_sconv.weight_proj", "attn.v_sconv.weight_proj"};
  const int R = proj_rows(layer.full);
  DeviceBuffer proj_tmp, wo_tmp;
  Tensor wproj_bf16, wo_bf16;
  auto queue_bf16_rows = [&](const std::string& name, void* dst, int rows, int cols) {
    const SafetensorsRef ref = m.at(name);
    if (ref->dtype != "BF16" || ref->numel() != std::int64_t(rows) * cols)
      throw std::runtime_error("weights: " + name + " must be BF16 [" + std::to_string(rows) + ", " +
                               std::to_string(cols) + "]");
    ld.queue(name, dst);
  };
  {
    void* d = dense_fp8 ? (proj_tmp = DeviceBuffer(std::size_t(R) * H * 2)).p : ld.alloc(std::size_t(R) * H * 2);
    for (int s = 0; s < num_segs(layer.full); ++s)
      queue_bf16_rows(p + seg_names[s], static_cast<char*>(d) + std::size_t(SEG_BEGIN[s]) * H * 2, SEG_ROWS[s], H);
    wproj_bf16 = rm(d, DType::BF16, {R, H});
    void* o = dense_fp8 ? (wo_tmp = DeviceBuffer(std::size_t(H) * 3072 * 2)).p : ld.alloc(std::size_t(H) * 3072 * 2);
    queue_bf16_rows(p + "attn.wo_ud.weight", o, H, HEADS * HEAD_DIM);
    wo_bf16 = rm(o, DType::BF16, {H, HEADS * HEAD_DIM});
  }

  layer.q_scale = ld.f32(p + "attn.q_l2_clip.scale", {HEADS, HEAD_DIM});
  layer.r_scale = ld.f32(p + "attn.rel_l2_clip.scale", {HEADS, D_REL});
  layer.rel_proj = ld.f32(p + "attn.rel_logits_proj.proj", {D_REL, REL_EXTENT});
  if (layer.full) {
    layer.k_static = ld.f32(p + "attn.k_sconv.static_w", {CONV_K, KV_DIM});
    layer.v_static = ld.f32(p + "attn.v_sconv.static_w", {CONV_K, KV_DIM});
  } else {
    layer.k_conv = ld.f32(p + "attn.k_sconv.weight", {KV_DIM, CONV_K});  // [C, 1, K] squeezed
    layer.v_conv = ld.f32(p + "attn.v_sconv.weight", {KV_DIM, CONV_K});
  }
  layer.attn_conv = ld.f32(p + "attn_sconv.weight", {H, CONV_K});
  layer.mlp_conv = ld.f32(p + "mlp_sconv.weight", {H, CONV_K});
  if (!full_nvfp4) layer.router = ld.f32(p + "mlp.gate.weight", {EXPERTS, H});
  layer.router_bias = ld.f32(p + "mlp.gate.bias", {EXPERTS});
  layer.router_bf16 = ld.upload_bf16(host_f32(m, p + "mlp.gate.weight"), {EXPERTS, H});  // exact
  if (l > 0) {
    const std::string qn = "model.llm.attn_res.attn_queries." + std::to_string(l - 1);
    const std::int32_t nq = std::int32_t(m.at(qn)->numel() / H);
    layer.attn_query = ld.f32(qn, {nq, H});
  }
  layer.mlp_query = ld.f32("model.llm.attn_res.mlp_queries." + std::to_string(l), {1, H});

  // stacked norm weights (fast._stack_layer_params)
  {
    std::vector<float> n3(std::size_t(3) * H);
    if (layer.full) {
      const char* names[3] = {"q_input_norm.weight", "k_input_norm.weight", "v_input_norm.weight"};
      for (int g = 0; g < 3; ++g) {
        const std::vector<float> v = host_f32(m, p + names[g]);
        std::copy(v.begin(), v.end(), n3.begin() + std::ptrdiff_t(g) * H);
      }
    } else {
      const std::vector<float> v = host_f32(m, p + "attn_norm.weight");
      for (int g = 0; g < 3; ++g) std::copy(v.begin(), v.end(), n3.begin() + std::ptrdiff_t(g) * H);
    }
    layer.attn_norm3 = ld.upload_f32(n3, {3, H});
    layer.attn_norm_mix = layer.full ? layer.attn_norm3 : rm(layer.attn_norm3.data, DType::FP32, {1, H});
    layer.mlp_norm1 = rm(layer.mlp_norm.data, DType::FP32, {1, H});
  }

  if (layer.loop) {
    // checkpoint controllers: attn (q), k, v, mlp; stacks [q, k, v] and [mlp]
    const char* ckpt[4] = {"attn", "k", "v", "mlp"};
    std::vector<float> down[4], forget[4], gain[4];
    for (int g = 0; g < 4; ++g) {
      const std::string c = p + "controllers." + ckpt[g] + ".";
      down[g] = host_f32(m, c + "down_weight");
      forget[g] = host_f32(m, c + "forget_weight");
      gain[g] = host_f32(m, c + "gain_weight");
    }
    auto stack = [](std::vector<float>* parts, int n) {
      std::vector<float> out;
      for (int i = 0; i < n; ++i) out.insert(out.end(), parts[i].begin(), parts[i].end());
      return out;
    };
    const int D2 = 2 * H;
    layer.qkv_gate_down = ld.upload_bf16(stack(down, 3), {3, GATE_RANK, D2});  // exact (BF16 checkpoint)
    layer.qkv_gate_forget16 = ld.upload_bf16(stack(forget, 3), {3, H, GATE_RANK});
    layer.qkv_gate_gain16 = ld.upload_bf16(stack(gain, 3), {3, H, GATE_RANK});
    layer.mlp1_gate_down = ld.upload_bf16(down[3], {1, GATE_RANK, D2});
    layer.mlp1_gate_forget16 = ld.upload_bf16(forget[3], {1, H, GATE_RANK});
    layer.mlp1_gate_gain16 = ld.upload_bf16(gain[3], {1, H, GATE_RANK});
    auto at = [](const Tensor& stack, int g, DType dt, std::int32_t a, std::int32_t b) {
      const std::size_t es = dt == DType::FP32 ? 4 : 2;
      return rm(static_cast<char*>(stack.data) + std::size_t(g) * a * b * es, dt, {a, b});
    };
    Tensor* downs[3] = {&layer.q_gate_down, &layer.k_gate_down, &layer.v_gate_down};
    for (int g = 0; g < 3; ++g) *downs[g] = at(layer.qkv_gate_down, g, DType::BF16, GATE_RANK, D2);
    layer.mlp_gate_down = at(layer.mlp1_gate_down, 0, DType::BF16, GATE_RANK, D2);
    if (!full_nvfp4) {
      layer.qkv_gate_forget = ld.upload_f32(stack(forget, 3), {3, H, GATE_RANK});
      layer.qkv_gate_gain = ld.upload_f32(stack(gain, 3), {3, H, GATE_RANK});
      layer.mlp1_gate_forget = ld.upload_f32(forget[3], {1, H, GATE_RANK});
      layer.mlp1_gate_gain = ld.upload_f32(gain[3], {1, H, GATE_RANK});
      Tensor* forgets[3] = {&layer.q_gate_forget, &layer.k_gate_forget, &layer.v_gate_forget};
      Tensor* gains[3] = {&layer.q_gate_gain, &layer.k_gate_gain, &layer.v_gate_gain};
      for (int g = 0; g < 3; ++g) {
        *forgets[g] = at(layer.qkv_gate_forget, g, DType::FP32, H, GATE_RANK);
        *gains[g] = at(layer.qkv_gate_gain, g, DType::FP32, H, GATE_RANK);
      }
      layer.mlp_gate_forget = at(layer.mlp1_gate_forget, 0, DType::FP32, H, GATE_RANK);
      layer.mlp_gate_gain = at(layer.mlp1_gate_gain, 0, DType::FP32, H, GATE_RANK);
    }
  }

  // routed experts in checkpoint encoding (or transcoded on-device from MXFP8 to NVFP4 when full_nvfp4 is active)
  const std::string e = p + "mlp.experts.";
  Experts& ex = layer.experts;
  const bool nv = fmt == ExpertFormat::NVFP4;
  const std::string want_codes = nv ? "U8" : "F8_E4M3", want_scales = nv ? "F8_E4M3" : "U8";
  if (m.at(e + "w13_weight")->dtype != want_codes || m.at(e + "w13_weight_scale")->dtype != want_scales ||
      m.at(e + "w2_weight")->dtype != want_codes || m.at(e + "w2_weight_scale")->dtype != want_scales)
    throw std::runtime_error("weights: layer " + std::to_string(l) + " experts do not match their format");

  if (!nv && full_nvfp4) {
    ex.fmt = ExpertFormat::NVFP4;
    auto transcode = [&](const std::string& prefix, int rows, int cols, Tensor& out_w, Tensor& out_s, Tensor& out_f) {
      const SafetensorsRef rc = m.at(prefix + "_weight");
      const SafetensorsRef rs = m.at(prefix + "_weight_scale");
      DeviceBuffer tmp_c(rc->bytes), tmp_s(rs->bytes);
      ld.queue(prefix + "_weight", tmp_c.p);
      ld.queue(prefix + "_weight_scale", tmp_s.p);
      ld.flush();
      void* nw = ld.alloc(std::size_t(EXPERTS) * rows * (cols / 2));
      void* ns = ld.alloc(std::size_t(EXPERTS) * rows * (cols / 16));
      float* nf = static_cast<float*>(ld.alloc(std::size_t(EXPERTS) * sizeof(float)));
      kernels::moe::mxfp8_to_nvfp4(tmp_c.p, tmp_s.p, rows, cols, nw, ns, nf, ld.stream());
      CUDA_CHECK(cudaStreamSynchronize(ld.stream()));  // free MXFP8 temporaries before next allocation
      out_w = rm(nw, DType::U8, {EXPERTS, rows, cols / 2});
      out_s = rm(ns, DType::FP8_E4M3FN, {EXPERTS, rows, cols / 16});
      out_f = rm(nf, DType::FP32, {EXPERTS});
    };
    transcode(e + "w13", 2 * INTER, H, ex.w13, ex.w13_scale, ex.w13_factor);
    transcode(e + "w2", H, INTER, ex.w2, ex.w2_scale, ex.w2_factor);
  } else {
    ex.fmt = fmt;
    ex.w13 = ld.raw(e + "w13_weight");
    ex.w13_scale = ld.raw(e + "w13_weight_scale");
    ex.w2 = ld.raw(e + "w2_weight");
    ex.w2_scale = ld.raw(e + "w2_weight_scale");
    if (nv) {
      for (const char* w : {"w13", "w2"}) {
        const std::vector<float> amax = host_f32(m, e + w + "_weight_amax");
        const std::vector<float> den = host_f32(m, e + w + "_weight_outer_scale_denominator");
        std::vector<float> f(EXPERTS);
        for (int i = 0; i < EXPERTS; ++i) f[i] = amax[i] / den[den.size() == 1 ? 0 : i];  // FP32 division
        (std::string(w) == "w13" ? ex.w13_factor : ex.w2_factor) = ld.upload_f32(f, {EXPERTS});
      }
    }
    ld.flush();
  }
  if (dense_fp8) {
    layer.wproj = to_fp8(ld, wproj_bf16, R, H);
    layer.wo = to_fp8(ld, wo_bf16, H, HEADS * HEAD_DIM);
    CUDA_CHECK(cudaStreamSynchronize(ld.stream()));  // before the BF16 temporaries are freed
  } else {
    layer.wproj = Dense{wproj_bf16, {}};
    layer.wo = Dense{wo_bf16, {}};
  }
  Dense* views[NUM_SEGS] = {&layer.wq, &layer.wgate, &layer.wr, &layer.wsink,
                            &layer.wk, &layer.wv,    &layer.wk_dyn, &layer.wv_dyn};
  for (int s = 0; s < num_segs(layer.full); ++s) *views[s] = row_view(layer.wproj, SEG_BEGIN[s], SEG_BEGIN[s + 1]);
  return layer;
}

}  // namespace

Weights load_weights(const fs::path& model_dir, bool dense_fp8, cudaStream_t stream, bool full_nvfp4) {
  const SafetensorsModel m(model_dir);
  // Use a lean 64 MB staging pool (4 x 16 MB) so systems with 16 GB DDR4 RAM never hit memory pressure.
  DeviceUploader up(UploadOptions{std::size_t(16) << 20, 4, 4});
  Weights w;
  w.dense_fp8 = dense_fp8;
  w.expert_formats = expert_formats(model_dir);
  w.storage.reserve(4096);
  Loader ld(m, up, stream, w.storage);
  const int H = HIDDEN;
  // Load embedding, unembedding, and MXFP8-transcoded layers first while plenty of VRAM is free for temporary
  // BF16/MXFP8 buffers, eliminating the ~900 MiB temporary peak at the end of model loading on 16 GB GPUs.
  {
    const std::vector<float> q = host_f32(m, "model.llm.middle_cycle_cross_pass_queries");
    const std::vector<float> kv = host_f32(m, "model.llm.middle_cycle_cross_pass_kv_queries");
    const int n = int(q.size() / H / 2);
    std::vector<float> mq(std::size_t(n) * 4 * H);
    for (int i = 0; i < n; ++i) {
      const float* rows[4] = {&q[std::size_t(2 * i) * H], &kv[std::size_t(2 * i) * H], &kv[std::size_t(2 * i + 1) * H],
                              &q[std::size_t(2 * i + 1) * H]};
      for (int j = 0; j < 4; ++j) std::copy(rows[j], rows[j] + H, mq.begin() + (std::ptrdiff_t(i) * 4 + j) * H);
    }
    w.message_queries = ld.upload_f32(mq, {n, 4, H});
  }
  if (full_nvfp4 && dense_fp8) {
    const std::string name = "model.llm.embed.weight";
    if (m.at(name)->dtype != "BF16") throw std::runtime_error("weights: embed must be BF16");
    const std::size_t bytes = std::size_t(VOCAB) * H * 2;
    DeviceBuffer embed_tmp(bytes);
    ld.queue(name, embed_tmp.p, VOCAB);
    ld.flush();
    const Tensor bf = rm(embed_tmp.p, DType::BF16, {VOCAB, H});
    const Dense d = to_fp8(ld, bf, VOCAB, H);
    CUDA_CHECK(cudaStreamSynchronize(stream));
    w.embed = d.w;
    w.embed_scale = d.scale;
  } else {
    w.embed = ld.raw("model.llm.embed.weight");
    if (w.embed.dtype != DType::BF16) throw std::runtime_error("weights: embed must be BF16");
  }
  w.embed_norm = ld.f32("model.llm.embed_norm.weight", {H});
  w.final_query = ld.f32("model.llm.attn_res.output_query", {1, H});
  w.final_norm = ld.f32("model.llm.norm.weight", {H});
  {
    DeviceBuffer unembed_tmp;
    const std::string name = "model.llm.unembed.weight";
    if (m.at(name)->dtype != "BF16") throw std::runtime_error("weights: unembed must be BF16");
    const std::size_t bytes = std::size_t(UNPADDED_VOCAB) * H * 2;
    void* d = dense_fp8 ? (unembed_tmp = DeviceBuffer(bytes)).p : ld.alloc(bytes);
    ld.queue(name, d, UNPADDED_VOCAB);  // unembed[:UNPADDED_VOCAB]
    ld.flush();
    const Tensor bf = rm(d, DType::BF16, {UNPADDED_VOCAB, H});
    if (dense_fp8) {
      w.unembed = to_fp8(ld, bf, UNPADDED_VOCAB, H);
      CUDA_CHECK(cudaStreamSynchronize(stream));
    } else {
      w.unembed = Dense{bf, {}};
    }
  }
  w.layers.resize(LAYERS);
  for (int pass = 0; pass < 2; ++pass) {
    for (int l = 0; l < LAYERS; ++l) {
      const bool is_mxfp8 = w.expert_formats[l] != ExpertFormat::NVFP4;
      if ((pass == 0) != is_mxfp8) continue;
      w.layers[l] = load_layer_into(ld, l, w.expert_formats[l], dense_fp8, full_nvfp4);
      if (full_nvfp4) w.expert_formats[l] = ExpertFormat::NVFP4;
    }
  }
  CUDA_CHECK(cudaStreamSynchronize(stream));
  return w;
}

Layer load_layer(const fs::path& model_dir, int l, bool dense_fp8, cudaStream_t stream,
                 std::vector<DeviceBuffer>& storage, bool full_nvfp4) {
  if (l < 0 || l >= LAYERS) throw std::invalid_argument("load_layer: layer out of range");
  const SafetensorsModel m(model_dir);
  DeviceUploader up(UploadOptions{std::size_t(16) << 20, 4, 4});
  Loader ld(m, up, stream, storage);
  Layer layer = load_layer_into(ld, l, expert_formats(model_dir)[l], dense_fp8, full_nvfp4);
  CUDA_CHECK(cudaStreamSynchronize(stream));
  return layer;
}

Ablation load_ablation(const fs::path& path) {
  Ablation a;
  const int H = HIDDEN;
  std::vector<float> attn_dirs(std::size_t(NUM_VISITS) * H, 0.f), mlp_dirs(attn_dirs.size(), 0.f);
  std::vector<float> attn_w(NUM_VISITS, 0.f), mlp_w(NUM_VISITS, 0.f);
  if (!path.empty()) {
    const SafetensorsModel m(path);
    auto read = [&](const char* name, std::size_t n) {
      std::vector<float> v = host_f32(m, name);
      if (v.size() != n) throw std::runtime_error(std::string("ablation: ") + name + " has the wrong size");
      return v;
    };
    // F.normalize(dirs, dim=-1): x / max(||x||, 1e-12) in FP32
    auto normalize = [&](std::vector<float> v) {
      for (int r = 0; r < NUM_VISITS; ++r) {
        double ss = 0;
        for (int c = 0; c < H; ++c) ss += double(v[std::size_t(r) * H + c]) * v[std::size_t(r) * H + c];
        const float norm = std::max(float(std::sqrt(ss)), 1e-12f);
        for (int c = 0; c < H; ++c) v[std::size_t(r) * H + c] = v[std::size_t(r) * H + c] / norm;
      }
      return v;
    };
    attn_dirs = normalize(read("attn_dirs", attn_dirs.size()));
    mlp_dirs = normalize(read("mlp_dirs", mlp_dirs.size()));
    attn_w = read("attn_w", NUM_VISITS);
    mlp_w = read("mlp_w", NUM_VISITS);
    a.active = true;
  }
  auto up = [&](const std::vector<float>& h, std::initializer_list<std::int32_t> shape) {
    a.storage.emplace_back(h.size() * 4);
    CUDA_CHECK(cudaMemcpy(a.storage.back().p, h.data(), h.size() * 4, cudaMemcpyHostToDevice));
    return rm(a.storage.back().p, DType::FP32, shape);
  };
  a.attn_dirs = up(attn_dirs, {NUM_VISITS, H});
  a.attn_w = up(attn_w, {NUM_VISITS});
  a.mlp_dirs = up(mlp_dirs, {NUM_VISITS, H});
  a.mlp_w = up(mlp_w, {NUM_VISITS});
  return a;
}

}  // namespace ingine::model
