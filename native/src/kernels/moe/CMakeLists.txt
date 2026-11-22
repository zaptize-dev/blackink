add_library(ingine_kernels_moe STATIC
  moe.cu
  dequant.cu
)
ingine_cuda_target(ingine_kernels_moe)
ingine_internal_includes(ingine_kernels_moe)
target_include_directories(ingine_kernels_moe PUBLIC ${PROJECT_SOURCE_DIR}/src)
target_link_libraries(ingine_kernels_moe PUBLIC ingine_core ingine_kernels_common CUDA::cudart)
