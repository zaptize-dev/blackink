// Adapted from fngine (github.com/satellitedown/fngine, Apache-2.0) for ingine.
// cuBLASLt BF16 GEMM for the prefill projections (from fngine src/kernels/linear/linear_bf16.cu): FP32
// compute, split-K partial sums reduced in FP32 only (torch's allow_bf16_reduced_precision_reduction = False),
// one heuristic algorithm cached per shape.
#include "kernels/prefill/prefill.h"

#include <cublasLt.h>

#include <map>
#include <stdexcept>
#include <string>
#include <tuple>

namespace ingine::kernels::prefill {

namespace {

void cublas_check(cublasStatus_t status, const char* what) {
  if (status != CUBLAS_STATUS_SUCCESS)
    throw std::runtime_error(std::string("cuBLASLt ") + what + " failed: status " +
                             std::to_string(static_cast<int>(status)));
}

constexpr std::size_t kWorkspaceBytes = 32u << 20;

}  // namespace

struct Gemm::Impl {
  cublasLtHandle_t handle = nullptr;
  void* workspace = nullptr;
  // (M, N, K, lda, ldb, ldc, fp32 output) -> algorithm
  std::map<std::tuple<int, int, int, int, int, int, bool>, cublasLtMatmulAlgo_t> algos;
};

Gemm::Gemm() : impl_(new Impl) {
  try {
    cublas_check(cublasLtCreate(&impl_->handle), "create");
    if (cudaMalloc(&impl_->workspace, kWorkspaceBytes) != cudaSuccess)
      throw std::runtime_error("prefill Gemm: workspace allocation failed");
  } catch (...) {
    if (impl_->handle) cublasLtDestroy(impl_->handle);
    delete impl_;
    throw;
  }
}

Gemm::~Gemm() {
  if (impl_->workspace) cudaFree(impl_->workspace);
  if (impl_->handle) cublasLtDestroy(impl_->handle);
  delete impl_;
}

// Row-major C[M, N] = A[M, K] . B[N, K]^T is column-major C^T[N, M] = B^T(op T on the K x N matrix B) . A
// (the K x M matrix A).
void Gemm::run(const __nv_bfloat16* A, int lda, const __nv_bfloat16* B, int ldb, void* C, int ldc, bool c_fp32,
               int M, int N, int K, cudaStream_t stream) {
  if (M <= 0 || N <= 0 || K <= 0) throw std::invalid_argument("prefill Gemm: empty shape");
  cublasLtMatmulDesc_t op = nullptr;
  cublasLtMatrixLayout_t la = nullptr, lb = nullptr, lc = nullptr;
  auto cleanup = [&] {
    if (la) cublasLtMatrixLayoutDestroy(la);
    if (lb) cublasLtMatrixLayoutDestroy(lb);
    if (lc) cublasLtMatrixLayoutDestroy(lc);
    if (op) cublasLtMatmulDescDestroy(op);
  };
  try {
    cublas_check(cublasLtMatmulDescCreate(&op, CUBLAS_COMPUTE_32F, CUDA_R_32F), "desc create");
    const cublasOperation_t ta = CUBLAS_OP_T, tb = CUBLAS_OP_N;
    cublas_check(cublasLtMatmulDescSetAttribute(op, CUBLASLT_MATMUL_DESC_TRANSA, &ta, sizeof(ta)), "transa");
    cublas_check(cublasLtMatmulDescSetAttribute(op, CUBLASLT_MATMUL_DESC_TRANSB, &tb, sizeof(tb)), "transb");
    cublas_check(cublasLtMatrixLayoutCreate(&la, CUDA_R_16BF, K, N, ldb), "layout a");
    cublas_check(cublasLtMatrixLayoutCreate(&lb, CUDA_R_16BF, K, M, lda), "layout b");
    cublas_check(cublasLtMatrixLayoutCreate(&lc, c_fp32 ? CUDA_R_32F : CUDA_R_16BF, N, M, ldc), "layout c");

    const auto key = std::make_tuple(M, N, K, lda, ldb, ldc, c_fp32);
    auto it = impl_->algos.find(key);
    if (it == impl_->algos.end()) {
      cublasLtMatmulPreference_t pref = nullptr;
      cublas_check(cublasLtMatmulPreferenceCreate(&pref), "preference create");
      const std::size_t ws = kWorkspaceBytes;
      const std::uint32_t reduction = CUBLASLT_REDUCTION_SCHEME_COMPUTE_TYPE;
      cublasLtMatmulHeuristicResult_t result{};
      int found = 0;
      cublasStatus_t st =
          cublasLtMatmulPreferenceSetAttribute(pref, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &ws, sizeof(ws));
      if (st == CUBLAS_STATUS_SUCCESS)
        st = cublasLtMatmulPreferenceSetAttribute(pref, CUBLASLT_MATMUL_PREF_REDUCTION_SCHEME_MASK, &reduction,
                                                  sizeof(reduction));
      if (st == CUBLAS_STATUS_SUCCESS)
        st = cublasLtMatmulAlgoGetHeuristic(impl_->handle, op, la, lb, lc, lc, pref, 1, &result, &found);
      cublasLtMatmulPreferenceDestroy(pref);
      cublas_check(st, "heuristic");
      if (found == 0) throw std::runtime_error("prefill Gemm: no cuBLASLt algorithm");
      it = impl_->algos.emplace(key, result.algo).first;
    }
    const float alpha = 1.f, beta = 0.f;
    cublas_check(cublasLtMatmul(impl_->handle, op, &alpha, B, la, A, lb, &beta, C, lc, C, lc, &it->second,
                                impl_->workspace, kWorkspaceBytes, stream),
                 "matmul");
  } catch (...) {
    cleanup();
    throw;
  }
  cleanup();
}

}  // namespace ingine::kernels::prefill
