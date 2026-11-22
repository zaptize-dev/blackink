# Adapted from fngine (github.com/satellitedown/fngine, Apache-2.0) for ingine.
find_package(CUDAToolkit REQUIRED)
find_package(Threads REQUIRED)
foreach(_cuda_target CUDA::cudart CUDA::cublasLt)
  if(NOT TARGET ${_cuda_target})
    message(FATAL_ERROR "CUDAToolkit does not provide ${_cuda_target}")
  endif()
endforeach()

# Internal headers: public API in include/, implementation in src/, vendored headers in third_party/.
function(ingine_internal_includes target)
  target_include_directories(${target} PRIVATE
    ${PROJECT_SOURCE_DIR}/include
    ${PROJECT_SOURCE_DIR}/src
    ${PROJECT_SOURCE_DIR}/third_party)
endfunction()

# CUDA compile settings shared by every target with .cu sources: whole-program compilation (no
# separable compilation or device linking), line info for profilers, constexpr host functions in
# device code.
function(ingine_cuda_target target)
  set_target_properties(${target} PROPERTIES
    CUDA_SEPARABLE_COMPILATION OFF
    CUDA_RESOLVE_DEVICE_SYMBOLS OFF)
  target_compile_options(${target} PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:-lineinfo --expt-relaxed-constexpr>)
endfunction()

# Test oracles: no reassociation or contraction.
set(INGINE_ORACLE_FLAGS -fno-fast-math -ffp-contract=off)
