// BF16 GEMM through cuBLASLt for the drafter's BF16 matmuls (x @ W.T in dspark.py: o_proj, down_proj and the
// context k/v projections): c bf16 [n, N] = bf16(a [n, K] . w [N, K]^T), FP32 accumulation, as torch's BF16
// matmul. Plans (descriptors + heuristic algorithm) are cached per shape; run() is CUDA-graph capturable.
#pragma once

#include <cublasLt.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <cstddef>
#include <map>
#include <tuple>

namespace ingine::kernels::draft {

class Bf16Gemm {
 public:
  explicit Bf16Gemm(std::size_t workspace_bytes = std::size_t(4) << 20);
  ~Bf16Gemm();
  Bf16Gemm(const Bf16Gemm&) = delete;
  Bf16Gemm& operator=(const Bf16Gemm&) = delete;

  void run(const __nv_bfloat16* a, int n, const __nv_bfloat16* w, int N, int K, __nv_bfloat16* c,
           cudaStream_t stream);

 private:
  struct Plan {
    cublasLtMatmulDesc_t op = nullptr;
    cublasLtMatrixLayout_t a = nullptr, b = nullptr, c = nullptr;
    cublasLtMatmulAlgo_t algo{};
  };
  Plan& plan(int n, int N, int K);

  cublasLtHandle_t handle_ = nullptr;
  void* workspace_ = nullptr;
  std::size_t workspace_bytes_ = 0;
  std::map<std::tuple<int, int, int>, Plan> plans_;
};

}  // namespace ingine::kernels::draft
