# Adapted from fngine (github.com/satellitedown/fngine, Apache-2.0) for ingine.
add_library(ingine_core STATIC
  arena.cu
  decode_graph.cpp
  device.cu
  dtype.cpp
  host_memory.cpp
  safetensors.cpp
  tensor.cpp
)
ingine_cuda_target(ingine_core)
ingine_internal_includes(ingine_core)
target_include_directories(ingine_core PUBLIC ${PROJECT_SOURCE_DIR}/src ${PROJECT_SOURCE_DIR}/include)
target_link_libraries(ingine_core PUBLIC
  CUDA::cudart CUDA::nvtx3 Threads::Threads spdlog::spdlog nlohmann_json::nlohmann_json)
