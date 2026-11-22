# Adapted from fngine (github.com/satellitedown/fngine, Apache-2.0) for ingine.
add_library(ingine_kernels_head STATIC
  head.cu
)
ingine_cuda_target(ingine_kernels_head)
ingine_internal_includes(ingine_kernels_head)
target_include_directories(ingine_kernels_head PUBLIC ${PROJECT_SOURCE_DIR}/src)
target_link_libraries(ingine_kernels_head PUBLIC ingine_core CUDA::cudart)
