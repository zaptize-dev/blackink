add_library(ingine_kernels_attn STATIC
  attn.cu
)
ingine_cuda_target(ingine_kernels_attn)
ingine_internal_includes(ingine_kernels_attn)
target_include_directories(ingine_kernels_attn PUBLIC ${PROJECT_SOURCE_DIR}/src)
target_link_libraries(ingine_kernels_attn PUBLIC ingine_core ingine_kernels_common CUDA::cudart)
