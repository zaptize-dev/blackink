#include "kernels/draft/gemm.h"

#include "core/device.h"

#include <stdexcept>
#include <string>

namespace ingine::kernels::draft {
namespace {

void lt_check(cublasStatus_t s, const char* what) {
  if (s != CUBLAS_STATUS_SUCCESS)
    throw std::runtime_error(std::string("cuBLASLt ") + what + " failed: " + std::to_string(int(s)));
}

}  // namespace

Bf16Gemm::Bf16Gemm(std::size_t workspace_bytes) : workspace_bytes_(workspace_bytes) {
  lt_check(cublasLtCreate(&handle_), "create");
  CUDA_CHECK(cudaMalloc(&workspace_, workspace_bytes_));
}

Bf16Gemm::~Bf16Gemm() {
  for (auto& [key, p] : plans_) {
    cublasLtMatrixLayoutDestroy(p.a);
    cublasLtMatrixLayoutDestroy(p.b);
    cublasLtMatrixLayoutDestroy(p.c);
    cublasLtMatmulDescDestroy(p.op);
  }
  if (workspace_ != nullptr) cudaFree(workspace_);
  if (handle_ != nullptr) cublasLtDestroy(handle_);
}

// Row-major c [n, N] = a [n, K] w[N, K]^T is column-major c^T [N, n] = op(w) a^T with w seen as a
// column-major [K, N] matrix (transposed) and a as column-major [K, n].
Bf16Gemm::Plan& Bf16Gemm::plan(int n, int N, int K) {
  const auto key = std::make_tuple(n, N, K);
  if (auto it = plans_.find(key); it != plans_.end()) return it->second;
  Plan p;
  lt_check(cublasLtMatmulDescCreate(&p.op, CUBLAS_COMPUTE_32F, CUDA_R_32F), "matmul desc");
  const cublasOperation_t ta = CUBLAS_OP_T, tb = CUBLAS_OP_N;
  lt_check(cublasLtMatmulDescSetAttribute(p.op, CUBLASLT_MATMUL_DESC_TRANSA, &ta, sizeof(ta)), "transa");
  lt_check(cublasLtMatmulDescSetAttribute(p.op, CUBLASLT_MATMUL_DESC_TRANSB, &tb, sizeof(tb)), "transb");
  lt_check(cublasLtMatrixLayoutCreate(&p.a, CUDA_R_16BF, K, N, K), "layout w");
  lt_check(cublasLtMatrixLayoutCreate(&p.b, CUDA_R_16BF, K, n, K), "layout a");
  lt_check(cublasLtMatrixLayoutCreate(&p.c, CUDA_R_16BF, N, n, N), "layout c");
  cublasLtMatmulPreference_t pref = nullptr;
  lt_check(cublasLtMatmulPreferenceCreate(&pref), "preference");
  lt_check(cublasLtMatmulPreferenceSetAttribute(pref, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &workspace_bytes_,
                                                sizeof(workspace_bytes_)),
           "workspace preference");
  cublasLtMatmulHeuristicResult_t result{};
  int found = 0;
  const cublasStatus_t s = cublasLtMatmulAlgoGetHeuristic(handle_, p.op, p.a, p.b, p.c, p.c, pref, 1, &result, &found);
  cublasLtMatmulPreferenceDestroy(pref);
  if (s != CUBLAS_STATUS_SUCCESS || found == 0) {
    cublasLtMatrixLayoutDestroy(p.a);
    cublasLtMatrixLayoutDestroy(p.b);
    cublasLtMatrixLayoutDestroy(p.c);
    cublasLtMatmulDescDestroy(p.op);
    throw std::runtime_error("cuBLASLt: no BF16 algorithm for [" + std::to_string(n) + ", " + std::to_string(K) +
                             "] x [" + std::to_string(K) + ", " + std::to_string(N) + "]");
  }
  p.algo = result.algo;
  return plans_.emplace(key, p).first->second;
}

void Bf16Gemm::run(const __nv_bfloat16* a, int n, const __nv_bfloat16* w, int N, int K, __nv_bfloat16* c,
                   cudaStream_t stream) {
  if (n <= 0) return;
  Plan& p = plan(n, N, K);
  const float alpha = 1.f, beta = 0.f;
  lt_check(cublasLtMatmul(handle_, p.op, &alpha, w, p.a, a, p.b, &beta, c, p.c, c, p.c, &p.algo, workspace_,
                          workspace_bytes_, stream),
           "matmul");
}

}  // namespace ingine::kernels::draft
