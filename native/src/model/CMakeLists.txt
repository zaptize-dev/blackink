add_library(ingine_model STATIC
  weights.cpp
  state.cpp
  decoder.cpp
  runtime.cpp
  generator.cpp
  model_kernels.cu
)
# prefill (package Prefill) and the speculative loop (package Spec)
foreach(src prefill.cpp drafter.cpp spec.cpp)
  if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/${src})
    target_sources(ingine_model PRIVATE ${src})
  endif()
endforeach()
ingine_cuda_target(ingine_model)
ingine_internal_includes(ingine_model)
target_include_directories(ingine_model PUBLIC ${PROJECT_SOURCE_DIR}/src ${PROJECT_SOURCE_DIR}/include)
target_link_libraries(ingine_model PUBLIC
  ingine_core
  ingine_kernels_dense
  ingine_kernels_moe
  ingine_kernels_attn
  ingine_kernels_head
  nlohmann_json::nlohmann_json
  CUDA::cudart)
foreach(area prefill draft)
  if(TARGET ingine_kernels_${area})
    target_link_libraries(ingine_model PUBLIC ingine_kernels_${area})
  endif()
endforeach()
