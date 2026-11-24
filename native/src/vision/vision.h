#pragma once

#include <cstdint>
#include <filesystem>
#include <memory>
#include <mutex>
#include <span>
#include <vector>

namespace ingine::vision {

struct ImageEmbeddings {
  int n_tokens = 0;         // number of merged visual tokens
  int hidden_size = 2304;   // text_hidden_size
  std::vector<float> data;  // [n_tokens * hidden_size] row-major FP32
};

// CPU/RAM-only Inkling-2-Mini vision encoder (inkling2_vision).
// Reads `model.visual.*` BF16 tensors from `model-00001-of-00001.safetensors` lazily on first use,
// using ~1.1 GiB of system RAM and 0 bytes of GPU VRAM.
class VisionEncoder {
 public:
  explicit VisionEncoder(std::filesystem::path model_dir);
  ~VisionEncoder();

  VisionEncoder(const VisionEncoder&) = delete;
  VisionEncoder& operator=(const VisionEncoder&) = delete;

  // Decodes JPEG/PNG/BMP/GIF image bytes and runs the 23-layer ViT + 2x2 merge + projector on CPU.
  ImageEmbeddings encode_image_bytes(std::span<const std::uint8_t> bytes, int max_tokens = 256) const;

 private:
  struct Weights;
  void ensure_loaded() const;

  std::filesystem::path model_dir_;
  mutable std::once_flag load_once_;
  mutable std::unique_ptr<Weights> weights_;
};

}  // namespace ingine::vision
