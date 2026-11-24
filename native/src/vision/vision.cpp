#include "vision/vision.h"

#define STB_IMAGE_IMPLEMENTATION
#include "stb/stb_image.h"

#include <nlohmann/json.hpp>

#include <algorithm>
#include <cmath>
#include <cstring>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace ingine::vision {
namespace {

constexpr int kPatchSize = 20;
constexpr int kMergeSize = 2;
constexpr int kAlign = kPatchSize * kMergeSize;  // 40
constexpr int kHidden = 1024;
constexpr int kHeads = 16;
constexpr int kHeadDim = 64;
constexpr int kMlpDim = 4096;
constexpr int kLayers = 23;
constexpr int kPosGrid = 32;
constexpr int kProjDim = 4096;
constexpr int kOutDim = 2304;
constexpr float kLnEps = 1e-5f;
constexpr float kRopeTheta = 10000.0f;

float bf16_to_f32(std::uint16_t u) {
  std::uint32_t bits = static_cast<std::uint32_t>(u) << 16;
  float f = 0.0f;
  std::memcpy(&f, &bits, sizeof(f));
  return f;
}

int align_up(int v, int a) { return ((v + a - 1) / a) * a; }

void layer_norm(const float* x, const float* w, const float* b, int rows, int dim, float* out) {
#pragma omp parallel for schedule(static) if (rows > 4)
  for (int r = 0; r < rows; ++r) {
    const float* xr = x + std::size_t(r) * dim;
    float* yr = out + std::size_t(r) * dim;
    double sum = 0.0;
    for (int d = 0; d < dim; ++d) sum += xr[d];
    const float mean = static_cast<float>(sum / dim);
    double var = 0.0;
    for (int d = 0; d < dim; ++d) {
      const float diff = xr[d] - mean;
      var += double(diff) * double(diff);
    }
    const float inv = 1.0f / std::sqrt(static_cast<float>(var / dim) + kLnEps);
    for (int d = 0; d < dim; ++d) {
      yr[d] = (xr[d] - mean) * inv * w[d] + b[d];
    }
  }
}

// C[M, N] = A[M, K] @ B[N, K]^T (+ bias[N] if non-null)
void matmul_nt(const float* A, const float* B, const float* bias, int M, int N, int K, float* C) {
#pragma omp parallel for collapse(2) schedule(static) if (M * N >= 1024)
  for (int m = 0; m < M; ++m) {
    for (int n = 0; n < N; ++n) {
      const float* a = A + std::size_t(m) * K;
      const float* b = B + std::size_t(n) * K;
      float acc = bias ? bias[n] : 0.0f;
#pragma omp simd reduction(+ : acc)
      for (int k = 0; k < K; ++k) {
        acc += a[k] * b[k];
      }
      C[std::size_t(m) * N + n] = acc;
    }
  }
}

void gelu_erf_inplace(float* x, std::size_t n) {
  constexpr float kInvSqrt2 = 0.7071067811865475f;
#pragma omp parallel for schedule(static) if (n > 4096)
  for (std::size_t i = 0; i < n; ++i) {
    const float v = x[i];
    x[i] = 0.5f * v * (1.0f + std::erff(v * kInvSqrt2));
  }
}

void apply_rope_2d(float* qk, int n_pos, const int* pos_w, const int* pos_h) {
  constexpr int kHalf = kHeadDim / 2;  // 32
  constexpr int kPairs = kHalf / 2;    // 16
  float inv_freq[kPairs];
  for (int i = 0; i < kPairs; ++i) {
    inv_freq[i] = std::pow(kRopeTheta, -2.0f * float(i) / float(kHalf));
  }
#pragma omp parallel for schedule(static) if (n_pos > 16)
  for (int p = 0; p < n_pos; ++p) {
    const float pw = float(pos_w[p]);
    const float ph = float(pos_h[p]);
    for (int h = 0; h < kHeads; ++h) {
      float* v = qk + (std::size_t(p) * kHeads + h) * kHeadDim;
      for (int i = 0; i < kPairs; ++i) {
        const float aw = pw * inv_freq[i];
        const float cw = std::cosf(aw), sw = std::sinf(aw);
        const float x0 = v[2 * i], x1 = v[2 * i + 1];
        v[2 * i] = x0 * cw - x1 * sw;
        v[2 * i + 1] = x0 * sw + x1 * cw;

        const float ah = ph * inv_freq[i];
        const float ch = std::cosf(ah), sh = std::sinf(ah);
        const float y0 = v[kHalf + 2 * i], y1 = v[kHalf + 2 * i + 1];
        v[kHalf + 2 * i] = y0 * ch - y1 * sh;
        v[kHalf + 2 * i + 1] = y0 * sh + y1 * ch;
      }
    }
  }
}

// Bilinear resize of RGB u8 [src_w, src_h, 3] -> [dst_w, dst_h, 3]
std::vector<std::uint8_t> resize_rgb_bilinear(const std::uint8_t* src, int src_w, int src_h, int dst_w, int dst_h) {
  std::vector<std::uint8_t> dst(std::size_t(dst_w) * dst_h * 3);
  const float sx = float(src_w) / float(dst_w);
  const float sy = float(src_h) / float(dst_h);
  for (int y = 0; y < dst_h; ++y) {
    const float fy = std::clamp((y + 0.5f) * sy - 0.5f, 0.0f, float(src_h - 1));
    const int y0 = int(fy);
    const int y1 = std::min(y0 + 1, src_h - 1);
    const float wy = fy - y0;
    for (int x = 0; x < dst_w; ++x) {
      const float fx = std::clamp((x + 0.5f) * sx - 0.5f, 0.0f, float(src_w - 1));
      const int x0 = int(fx);
      const int x1 = std::min(x0 + 1, src_w - 1);
      const float wx = fx - x0;
      const std::uint8_t* p00 = src + (std::size_t(y0) * src_w + x0) * 3;
      const std::uint8_t* p01 = src + (std::size_t(y0) * src_w + x1) * 3;
      const std::uint8_t* p10 = src + (std::size_t(y1) * src_w + x0) * 3;
      const std::uint8_t* p11 = src + (std::size_t(y1) * src_w + x1) * 3;
      std::uint8_t* out = dst.data() + (std::size_t(y) * dst_w + x) * 3;
      for (int c = 0; c < 3; ++c) {
        const float v = (1.0f - wy) * ((1.0f - wx) * p00[c] + wx * p01[c]) +
                        wy * ((1.0f - wx) * p10[c] + wx * p11[c]);
        out[c] = static_cast<std::uint8_t>(std::clamp(std::lround(v), 0L, 255L));
      }
    }
  }
  return dst;
}

}  // namespace

struct VisionEncoder::Weights {
  struct Layer {
    std::vector<float> ln1_w, ln1_b, ln2_w, ln2_b;
    std::vector<float> ls1, ls2;
    std::vector<float> q_w, q_b, k_w, k_b, v_w, v_b, o_w, o_b;
    std::vector<float> fc1_w, fc1_b, fc2_w, fc2_b;
  };
  std::vector<float> class_emb, cls_pos_emb;
  std::vector<float> patch_w;   // [1024, 1200]
  std::vector<float> pos_emb;   // [1024, 1024] = [32, 32, 1024]
  std::vector<float> ln_pre_w, ln_pre_b;
  std::vector<Layer> layers{kLayers};
  std::vector<float> proj_in_w;   // [4096, 4096]
  std::vector<float> proj_out_w;  // [2304, 4096]
};

VisionEncoder::VisionEncoder(std::filesystem::path model_dir) : model_dir_(std::move(model_dir)) {}
VisionEncoder::~VisionEncoder() = default;

void VisionEncoder::ensure_loaded() const {
  std::call_once(load_once_, [&]() {
    const auto path = model_dir_ / "model-00001-of-00001.safetensors";
    std::ifstream f(path, std::ios::binary);
    if (!f) throw std::runtime_error("VisionEncoder: cannot open " + path.string());
    std::uint64_t hdr_len = 0;
    f.read(reinterpret_cast<char*>(&hdr_len), 8);
    if (!f || hdr_len == 0 || hdr_len > (64ULL << 20)) {
      throw std::runtime_error("VisionEncoder: invalid safetensors header in " + path.string());
    }
    std::string hdr_str(hdr_len, '\0');
    f.read(hdr_str.data(), std::streamsize(hdr_len));
    const nlohmann::json hdr = nlohmann::json::parse(hdr_str);
    const std::uint64_t data_base = 8 + hdr_len;

    auto w = std::make_unique<Weights>();
    auto load_tensor = [&](const std::string& name) -> std::vector<float> {
      const auto it = hdr.find(name);
      if (it == hdr.end()) throw std::runtime_error("VisionEncoder: missing tensor " + name);
      const std::string dtype = it->at("dtype").get<std::string>();
      const std::uint64_t off0 = it->at("data_offsets").at(0).get<std::uint64_t>();
      const std::uint64_t off1 = it->at("data_offsets").at(1).get<std::uint64_t>();
      const std::size_t nbytes = std::size_t(off1 - off0);
      f.seekg(std::streamoff(data_base + off0), std::ios::beg);
      if (dtype == "BF16") {
        std::vector<std::uint16_t> raw(nbytes / 2);
        f.read(reinterpret_cast<char*>(raw.data()), std::streamsize(nbytes));
        std::vector<float> out(raw.size());
        for (std::size_t i = 0; i < raw.size(); ++i) out[i] = bf16_to_f32(raw[i]);
        return out;
      }
      if (dtype == "F32") {
        std::vector<float> out(nbytes / 4);
        f.read(reinterpret_cast<char*>(out.data()), std::streamsize(nbytes));
        return out;
      }
      throw std::runtime_error("VisionEncoder: unsupported dtype " + dtype + " for " + name);
    };

    w->class_emb = load_tensor("model.visual.backbone.embeddings.class_embedding");
    w->cls_pos_emb = load_tensor("model.visual.backbone.embeddings.cls_position_embedding");
    w->patch_w = load_tensor("model.visual.backbone.embeddings.patch_embedding.weight");
    w->pos_emb = load_tensor("model.visual.backbone.embeddings.position_embedding");
    w->ln_pre_w = load_tensor("model.visual.backbone.ln_pre.weight");
    w->ln_pre_b = load_tensor("model.visual.backbone.ln_pre.bias");
    for (int l = 0; l < kLayers; ++l) {
      const std::string p = "model.visual.backbone.layers." + std::to_string(l) + ".";
      auto& L = w->layers[l];
      L.ln1_w = load_tensor(p + "ln_1.weight");
      L.ln1_b = load_tensor(p + "ln_1.bias");
      L.ln2_w = load_tensor(p + "ln_2.weight");
      L.ln2_b = load_tensor(p + "ln_2.bias");
      L.ls1 = load_tensor(p + "ls_1.gamma");
      L.ls2 = load_tensor(p + "ls_2.gamma");
      L.q_w = load_tensor(p + "self_attn.q_proj.weight");
      L.q_b = load_tensor(p + "self_attn.q_proj.bias");
      L.k_w = load_tensor(p + "self_attn.k_proj.weight");
      L.k_b = load_tensor(p + "self_attn.k_proj.bias");
      L.v_w = load_tensor(p + "self_attn.v_proj.weight");
      L.v_b = load_tensor(p + "self_attn.v_proj.bias");
      L.o_w = load_tensor(p + "self_attn.out_proj.weight");
      L.o_b = load_tensor(p + "self_attn.out_proj.bias");
      L.fc1_w = load_tensor(p + "mlp.c_fc.weight");
      L.fc1_b = load_tensor(p + "mlp.c_fc.bias");
      L.fc2_w = load_tensor(p + "mlp.c_proj.weight");
      L.fc2_b = load_tensor(p + "mlp.c_proj.bias");
    }
    w->proj_in_w = load_tensor("model.visual.proj_in.weight");
    w->proj_out_w = load_tensor("model.visual.proj_out.weight");
    weights_ = std::move(w);
  });
}

ImageEmbeddings VisionEncoder::encode_image_bytes(std::span<const std::uint8_t> bytes, int max_tokens) const {
  ensure_loaded();
  int raw_w = 0, raw_h = 0, comp = 0;
  stbi_uc* decoded = stbi_load_from_memory(bytes.data(), int(bytes.size()), &raw_w, &raw_h, &comp, 3);
  if (!decoded || raw_w <= 0 || raw_h <= 0) {
    if (decoded) stbi_image_free(decoded);
    throw std::runtime_error("VisionEncoder: failed to decode image (supported: JPEG, PNG, BMP, GIF)");
  }

  std::vector<std::uint8_t> img_buf;
  const std::uint8_t* img_ptr = decoded;
  int w = raw_w, h = raw_h;
  const std::int64_t max_pixels = std::int64_t(std::max(1, max_tokens)) * kAlign * kAlign;
  if (std::int64_t(align_up(w, kAlign)) * align_up(h, kAlign) > max_pixels) {
    const double scale = std::sqrt(double(max_pixels) / (double(w) * double(h)));
    int nw = std::max(kAlign, int(std::floor(w * scale / kAlign)) * kAlign);
    int nh = std::max(kAlign, int(std::floor(h * scale / kAlign)) * kAlign);
    while (std::int64_t(nw) * nh > max_pixels && (nw > kAlign || nh > kAlign)) {
      if (nw >= nh && nw > kAlign) nw -= kAlign;
      else if (nh > kAlign) nh -= kAlign;
    }
    img_buf = resize_rgb_bilinear(decoded, raw_w, raw_h, nw, nh);
    img_ptr = img_buf.data();
    w = nw;
    h = nh;
  }

  const int pw = align_up(w, kAlign);
  const int ph = align_up(h, kAlign);
  std::vector<float> norm_img(std::size_t(pw) * ph * 3, -1.0f);
  for (int y = 0; y < h; ++y) {
    for (int x = 0; x < w; ++x) {
      const std::uint8_t* src = img_ptr + (std::size_t(y) * w + x) * 3;
      float* dst = norm_img.data() + (std::size_t(y) * pw + x) * 3;
      dst[0] = (float(src[0]) / 255.0f - 0.5f) / 0.5f;
      dst[1] = (float(src[1]) / 255.0f - 0.5f) / 0.5f;
      dst[2] = (float(src[2]) / 255.0f - 0.5f) / 0.5f;
    }
  }
  stbi_image_free(decoded);

  const int nx = pw / kPatchSize;
  const int ny = ph / kPatchSize;
  const int n_patches = nx * ny;
  const int n_pos = n_patches + 1;
  constexpr int kPatchDim = 3 * kPatchSize * kPatchSize;  // 1200

  std::vector<float> patches(std::size_t(n_patches) * kPatchDim);
  for (int py = 0; py < ny; ++py) {
    for (int px = 0; px < nx; ++px) {
      float* dst = patches.data() + (std::size_t(py) * nx + px) * kPatchDim;
      for (int c = 0; c < 3; ++c) {
        for (int dy = 0; dy < kPatchSize; ++dy) {
          for (int dx = 0; dx < kPatchSize; ++dx) {
            const int iy = py * kPatchSize + dy;
            const int ix = px * kPatchSize + dx;
            dst[(c * kPatchSize + dy) * kPatchSize + dx] = norm_img[(std::size_t(iy) * pw + ix) * 3 + c];
          }
        }
      }
    }
  }

  const Weights& W = *weights_;
  std::vector<float> patch_emb(std::size_t(n_patches) * kHidden);
  matmul_nt(patches.data(), W.patch_w.data(), nullptr, n_patches, kHidden, kPatchDim, patch_emb.data());

  // Bilinearly resize [32, 32, 1024] position embeddings to [ny, nx, 1024]
  const float sy = float(kPosGrid) / float(ny);
  const float sx = float(kPosGrid) / float(nx);
  for (int py = 0; py < ny; ++py) {
    const float fy = std::clamp((py + 0.5f) * sy - 0.5f, 0.0f, float(kPosGrid - 1));
    const int y0 = int(fy), y1 = std::min(y0 + 1, kPosGrid - 1);
    const float wy = fy - y0;
    for (int px = 0; px < nx; ++px) {
      const float fx = std::clamp((px + 0.5f) * sx - 0.5f, 0.0f, float(kPosGrid - 1));
      const int x0 = int(fx), x1 = std::min(x0 + 1, kPosGrid - 1);
      const float wx = fx - x0;
      const float* p00 = W.pos_emb.data() + (std::size_t(y0) * kPosGrid + x0) * kHidden;
      const float* p01 = W.pos_emb.data() + (std::size_t(y0) * kPosGrid + x1) * kHidden;
      const float* p10 = W.pos_emb.data() + (std::size_t(y1) * kPosGrid + x0) * kHidden;
      const float* p11 = W.pos_emb.data() + (std::size_t(y1) * kPosGrid + x1) * kHidden;
      float* dst = patch_emb.data() + (std::size_t(py) * nx + px) * kHidden;
      for (int d = 0; d < kHidden; ++d) {
        dst[d] += (1.0f - wy) * ((1.0f - wx) * p00[d] + wx * p01[d]) +
                  wy * ((1.0f - wx) * p10[d] + wx * p11[d]);
      }
    }
  }

  std::vector<float> x_raw(std::size_t(n_pos) * kHidden);
  for (int d = 0; d < kHidden; ++d) x_raw[d] = W.class_emb[d] + W.cls_pos_emb[d];
  std::memcpy(x_raw.data() + kHidden, patch_emb.data(), patch_emb.size() * sizeof(float));

  std::vector<float> x(std::size_t(n_pos) * kHidden);
  layer_norm(x_raw.data(), W.ln_pre_w.data(), W.ln_pre_b.data(), n_pos, kHidden, x.data());

  std::vector<int> pos_w(n_pos, 0), pos_h(n_pos, 0);
  for (int py = 0; py < ny; ++py) {
    for (int px = 0; px < nx; ++px) {
      const int idx = 1 + py * nx + px;
      pos_w[idx] = px + 1;
      pos_h[idx] = py + 1;
    }
  }

  std::vector<float> ln_buf(std::size_t(n_pos) * kHidden);
  std::vector<float> q(std::size_t(n_pos) * kHidden);
  std::vector<float> k(std::size_t(n_pos) * kHidden);
  std::vector<float> v(std::size_t(n_pos) * kHidden);
  std::vector<float> attn(std::size_t(n_pos) * kHidden);
  std::vector<float> attn_proj(std::size_t(n_pos) * kHidden);
  std::vector<float> mlp_mid(std::size_t(n_pos) * kMlpDim);
  std::vector<float> mlp_out(std::size_t(n_pos) * kHidden);
  const float attn_scale = 1.0f / std::sqrt(float(kHeadDim));

  for (int l = 0; l < kLayers; ++l) {
    const auto& L = W.layers[l];
    layer_norm(x.data(), L.ln1_w.data(), L.ln1_b.data(), n_pos, kHidden, ln_buf.data());
    matmul_nt(ln_buf.data(), L.q_w.data(), L.q_b.data(), n_pos, kHidden, kHidden, q.data());
    matmul_nt(ln_buf.data(), L.k_w.data(), L.k_b.data(), n_pos, kHidden, kHidden, k.data());
    matmul_nt(ln_buf.data(), L.v_w.data(), L.v_b.data(), n_pos, kHidden, kHidden, v.data());
    apply_rope_2d(q.data(), n_pos, pos_w.data(), pos_h.data());
    apply_rope_2d(k.data(), n_pos, pos_w.data(), pos_h.data());

    // Multi-head bidirectional attention [n_pos, kHeads, kHeadDim]
#pragma omp parallel for collapse(2) schedule(static)
    for (int h_idx = 0; h_idx < kHeads; ++h_idx) {
      for (int qi = 0; qi < n_pos; ++qi) {
        std::vector<float> scores(n_pos);
        const float* q_vec = q.data() + (std::size_t(qi) * kHeads + h_idx) * kHeadDim;
        float max_s = -1e30f;
        for (int ki = 0; ki < n_pos; ++ki) {
          const float* k_vec = k.data() + (std::size_t(ki) * kHeads + h_idx) * kHeadDim;
          float dot = 0.0f;
          for (int d = 0; d < kHeadDim; ++d) dot += q_vec[d] * k_vec[d];
          const float s = dot * attn_scale;
          scores[ki] = s;
          if (s > max_s) max_s = s;
        }
        float sum_exp = 0.0f;
        for (int ki = 0; ki < n_pos; ++ki) {
          scores[ki] = std::exp(scores[ki] - max_s);
          sum_exp += scores[ki];
        }
        const float inv_sum = 1.0f / sum_exp;
        float* out_vec = attn.data() + (std::size_t(qi) * kHeads + h_idx) * kHeadDim;
        std::memset(out_vec, 0, kHeadDim * sizeof(float));
        for (int ki = 0; ki < n_pos; ++ki) {
          const float p = scores[ki] * inv_sum;
          const float* v_vec = v.data() + (std::size_t(ki) * kHeads + h_idx) * kHeadDim;
          for (int d = 0; d < kHeadDim; ++d) out_vec[d] += p * v_vec[d];
        }
      }
    }

    matmul_nt(attn.data(), L.o_w.data(), L.o_b.data(), n_pos, kHidden, kHidden, attn_proj.data());
    for (int i = 0; i < n_pos; ++i) {
      float* xr = x.data() + std::size_t(i) * kHidden;
      const float* ar = attn_proj.data() + std::size_t(i) * kHidden;
      for (int d = 0; d < kHidden; ++d) xr[d] += ar[d] * L.ls1[d];
    }

    layer_norm(x.data(), L.ln2_w.data(), L.ln2_b.data(), n_pos, kHidden, ln_buf.data());
    matmul_nt(ln_buf.data(), L.fc1_w.data(), L.fc1_b.data(), n_pos, kMlpDim, kHidden, mlp_mid.data());
    gelu_erf_inplace(mlp_mid.data(), std::size_t(n_pos) * kMlpDim);
    matmul_nt(mlp_mid.data(), L.fc2_w.data(), L.fc2_b.data(), n_pos, kHidden, kMlpDim, mlp_out.data());
    for (int i = 0; i < n_pos; ++i) {
      float* xr = x.data() + std::size_t(i) * kHidden;
      const float* mr = mlp_out.data() + std::size_t(i) * kHidden;
      for (int d = 0; d < kHidden; ++d) xr[d] += mr[d] * L.ls2[d];
    }
  }

  // Drop CLS token and apply 2x2 spatial merge: [ny, nx, 1024] -> [ny/2 * nx/2, 4096]
  const int my = ny / kMergeSize;
  const int mx = nx / kMergeSize;
  const int n_merged = my * mx;
  std::vector<float> merged(std::size_t(n_merged) * kProjDim);
  for (int hy = 0; hy < my; ++hy) {
    for (int wx = 0; wx < mx; ++wx) {
      float* dst = merged.data() + (std::size_t(hy) * mx + wx) * kProjDim;
      for (int dy = 0; dy < kMergeSize; ++dy) {
        for (int dx = 0; dx < kMergeSize; ++dx) {
          const int patch_idx = 1 + (hy * kMergeSize + dy) * nx + (wx * kMergeSize + dx);
          const float* src = x.data() + std::size_t(patch_idx) * kHidden;
          std::memcpy(dst + (dy * kMergeSize + dx) * kHidden, src, kHidden * sizeof(float));
        }
      }
    }
  }

  std::vector<float> proj_mid(std::size_t(n_merged) * kProjDim);
  matmul_nt(merged.data(), W.proj_in_w.data(), nullptr, n_merged, kProjDim, kProjDim, proj_mid.data());
  gelu_erf_inplace(proj_mid.data(), proj_mid.size());

  ImageEmbeddings out;
  out.n_tokens = n_merged;
  out.hidden_size = kOutDim;
  out.data.resize(std::size_t(n_merged) * kOutDim);
  matmul_nt(proj_mid.data(), W.proj_out_w.data(), nullptr, n_merged, kOutDim, kProjDim, out.data.data());
  return out;
}

}  // namespace ingine::vision
