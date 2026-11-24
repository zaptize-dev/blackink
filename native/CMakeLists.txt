# Adapted from fngine (github.com/satellitedown/fngine, Apache-2.0) for ingine.
cmake_minimum_required(VERSION 3.28)

# ingine is compiled only for sm_120a (RTX 5090 class). Set and validate the architecture before
# CUDA compiler detection.
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  set(CMAKE_CUDA_ARCHITECTURES 120a CACHE STRING "CUDA architectures to build")
endif()
if(NOT CMAKE_CUDA_ARCHITECTURES STREQUAL "120a")
  message(FATAL_ERROR
    "ingine supports only CMAKE_CUDA_ARCHITECTURES=120a; got "
    "'${CMAKE_CUDA_ARCHITECTURES}'")
endif()

# Prefer the installer's private toolkit, then /opt/cuda, when nvcc is not chosen explicitly.
if(NOT DEFINED CMAKE_CUDA_COMPILER AND NOT DEFINED ENV{CUDACXX})
  foreach(_nvcc ${CMAKE_CURRENT_SOURCE_DIR}/../.cuda-toolkit/nvidia/cu13/bin/nvcc
                ${CMAKE_CURRENT_SOURCE_DIR}/../.cuda-toolkit/bin/nvcc /opt/cuda/bin/nvcc)
    if(EXISTS ${_nvcc})
      set(CMAKE_CUDA_COMPILER ${_nvcc} CACHE FILEPATH "CUDA compiler")
      break()
    endif()
  endforeach()
endif()

project(ingine LANGUAGES C CXX CUDA)

option(INGINE_BUILD_TESTS "Build ingine tests" OFF)
option(INGINE_BUILD_APPS "Build the ingine executables (serve, bench, validate)" ON)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CXX_SCAN_FOR_MODULES OFF) # no C++20 modules; skip the dependency scan
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

if(CMAKE_GENERATOR MATCHES "Ninja")
  set_property(GLOBAL PROPERTY JOB_POOLS ingine_link=1)
  set(CMAKE_JOB_POOL_LINK ingine_link)
endif()

include(cmake/Dependencies.cmake)

add_subdirectory(third_party)
add_subdirectory(src/core)

# Every other component directory owns its CMakeLists.txt and defines the target named here; a
# directory is added only once it has one, so the tree builds while components are in progress.
#   src/text              -> ingine_text
#   src/kernels/<area>    -> ingine_kernels_<area>   (aggregated by ingine_kernels)
#   src/model             -> ingine_model
#   src/engine            -> ingine_engine
#   src/serve             -> ingine_serve
#   apps/serve|bench|validate -> ingine-serve | ingine-bench | ingine-validate
function(ingine_add_component dir)
  if(EXISTS ${PROJECT_SOURCE_DIR}/${dir}/CMakeLists.txt)
    add_subdirectory(${dir})
  endif()
endfunction()

ingine_add_component(src/text)

set(INGINE_KERNEL_AREAS common dense moe attn head prefill draft)
foreach(area ${INGINE_KERNEL_AREAS})
  ingine_add_component(src/kernels/${area})
endforeach()
add_library(ingine_kernels INTERFACE)
foreach(area ${INGINE_KERNEL_AREAS})
  if(TARGET ingine_kernels_${area})
    target_link_libraries(ingine_kernels INTERFACE ingine_kernels_${area})
  endif()
endforeach()

ingine_add_component(src/vision)
ingine_add_component(src/model)
ingine_add_component(src/engine)
ingine_add_component(src/serve)

if(INGINE_BUILD_APPS)
  foreach(app serve bench validate)
    ingine_add_component(apps/${app})
  endforeach()
endif()

if(INGINE_BUILD_TESTS)
  enable_testing()
  add_subdirectory(tests)
endif()
