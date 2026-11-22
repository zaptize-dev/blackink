foreach(name attention moe rows)
  if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/test_${name}.cpp)
    ingine_add_test(NAME prefill_${name} SOURCES test_${name}.cpp LIBS ingine_kernels_prefill GPU ORACLE TIMEOUT 600)
  endif()
endforeach()

# Kernel timings at production sizes with real layer weights; exits 77 (skip) unless INGINE_BENCH=1.
if(TARGET ingine_model AND EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/bench_prefill.cu)
  ingine_add_test(NAME prefill_bench SOURCES bench_prefill.cu LIBS ingine_model GPU MODEL LABELS bench
                  TIMEOUT 900)
endif()

# Model-level (whole model on the GPU): Prefiller consistency vs. chunking and vs. the decode step, and throughput.
if(TARGET ingine_model)
  if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/test_prefill_model.cpp)
    foreach(kv bf16 fp8)
      ingine_add_test(NAME prefill_model_kv${kv} SOURCES test_prefill_model.cpp LIBS ingine_model nlohmann_json::nlohmann_json
                      ARGS ${kv} GPU MODEL TIMEOUT 1800)
    endforeach()
  endif()
  if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/bench_prefill_model.cpp)
    ingine_add_test(NAME prefill_model_bench SOURCES bench_prefill_model.cpp LIBS ingine_model GPU MODEL
                    LABELS bench TIMEOUT 3600)
  endif()
endif()

# Whole-Prefiller timing on synthetic weights (< 2 GB, no checkpoint); exits 77 unless INGINE_BENCH=1.
if(TARGET ingine_model AND EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/bench_prefill_synth.cu)
  ingine_add_test(NAME prefill_synth_bench SOURCES bench_prefill_synth.cu LIBS ingine_model GPU LABELS bench
                  TIMEOUT 900)
endif()

# Real-model timing at (chunk, context) points for nsys kernel breakdowns; exits 77 unless INGINE_BENCH=1.
if(TARGET ingine_model AND EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/profile_prefill_model.cpp)
  ingine_add_test(NAME prefill_profile SOURCES profile_prefill_model.cpp LIBS ingine_model nlohmann_json::nlohmann_json
                  GPU MODEL LABELS bench TIMEOUT 900)
endif()
