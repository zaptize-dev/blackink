# Adapted from fngine (github.com/satellitedown/fngine, Apache-2.0) for ingine.
# ingine tests (CTest). Each area directory tests/<area>/ owns a CMakeLists.txt calling
# ingine_add_test(); the directory is added once that file exists.
#
#   Exit code 77 = skipped (no GPU, missing data). Shared helpers: tests/common/test_util.h.
#   Compile definitions in every test: INGINE_SOURCE_DIR (native/), INGINE_GOLDEN_DIR
#   (native/tests/golden), INGINE_MODEL_DIR (default model directory, may be absent).
#   GPU tests (GPU flag) carry the label `gpu` and share one resource lock, so they never run
#   concurrently: `ctest -L gpu` / `ctest -LE gpu` select or exclude them.
#   Model-level tests (MODEL flag) carry the label `model` (they load real weights).

# Model directory for model-level tests: -DINGINE_TEST_MODEL_DIR=<dir>, else $INGINE_MODEL_DIR, else the
# installer's setting in ~/.config/ingine/env, else the installer's download location.
set(_model_dir "$ENV{INGINE_MODEL_DIR}")
if(NOT _model_dir AND EXISTS "$ENV{HOME}/.config/ingine/env")
  file(STRINGS "$ENV{HOME}/.config/ingine/env" _lines REGEX "^INGINE_MODEL_DIR=")
  foreach(_line IN LISTS _lines)
    string(REGEX REPLACE "^INGINE_MODEL_DIR=[\"']?([^\"']*)[\"']?$" "\\1" _model_dir "${_line}")
  endforeach()
endif()
if(NOT _model_dir)
  get_filename_component(_model_dir "${PROJECT_SOURCE_DIR}/../models/Inkling-2-Mini" ABSOLUTE)
endif()
set(INGINE_TEST_MODEL_DIR "${_model_dir}" CACHE PATH "Model directory for model-level tests")

# ingine_add_test(NAME <name> SOURCES <src>... [LIBS <lib>...] [ARGS <arg>...] [LABELS <l>...]
#                 [TIMEOUT <seconds>] [GPU] [MODEL] [ORACLE])
#   ORACLE: compile host code with -fno-fast-math -ffp-contract=off (FP64 oracles).
function(ingine_add_test)
  cmake_parse_arguments(T "GPU;MODEL;ORACLE" "NAME;TIMEOUT" "SOURCES;LIBS;ARGS;LABELS" ${ARGN})
  if(NOT T_NAME OR NOT T_SOURCES)
    message(FATAL_ERROR "ingine_add_test requires NAME and SOURCES")
  endif()
  add_executable(${T_NAME} ${T_SOURCES})
  ingine_internal_includes(${T_NAME})
  target_include_directories(${T_NAME} PRIVATE ${PROJECT_SOURCE_DIR}/tests)
  target_link_libraries(${T_NAME} PRIVATE ${T_LIBS})
  target_compile_definitions(${T_NAME} PRIVATE
    INGINE_SOURCE_DIR="${PROJECT_SOURCE_DIR}"
    INGINE_GOLDEN_DIR="${PROJECT_SOURCE_DIR}/tests/golden"
    INGINE_MODEL_DIR="${INGINE_TEST_MODEL_DIR}")
  if(T_ORACLE)
    target_compile_options(${T_NAME} PRIVATE $<$<COMPILE_LANGUAGE:CXX>:${INGINE_ORACLE_FLAGS}>)
  endif()
  foreach(_s ${T_SOURCES})
    if(_s MATCHES "\\.cu$")
      ingine_cuda_target(${T_NAME})
      break()
    endif()
  endforeach()
  add_test(NAME ${T_NAME} COMMAND ${T_NAME} ${T_ARGS})
  set(_labels ${T_LABELS})
  if(T_GPU)
    list(APPEND _labels gpu)
    set_tests_properties(${T_NAME} PROPERTIES RESOURCE_LOCK gpu)
  endif()
  if(T_MODEL)
    list(APPEND _labels model)
  endif()
  set_tests_properties(${T_NAME} PROPERTIES SKIP_RETURN_CODE 77)
  if(_labels)
    set_tests_properties(${T_NAME} PROPERTIES LABELS "${_labels}")
  endif()
  if(T_TIMEOUT)
    set_tests_properties(${T_NAME} PROPERTIES TIMEOUT ${T_TIMEOUT})
  endif()
endfunction()

file(GLOB _ingine_test_areas LIST_DIRECTORIES true ${CMAKE_CURRENT_SOURCE_DIR}/*)
list(SORT _ingine_test_areas)
foreach(_area ${_ingine_test_areas})
  if(IS_DIRECTORY ${_area} AND EXISTS ${_area}/CMakeLists.txt)
    add_subdirectory(${_area})
  endif()
endforeach()
