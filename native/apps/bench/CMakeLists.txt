# ingine-bench: end-to-end prefill/decode/speculative speed (port of engine/bench/bench_e2e.py). Needs the model.
if(TARGET ingine_model)
  add_executable(ingine-bench main.cpp forced.cpp)
  ingine_internal_includes(ingine-bench)
  target_compile_definitions(ingine-bench PRIVATE INGINE_REPO_DIR="${PROJECT_SOURCE_DIR}/..")
  target_link_libraries(ingine-bench PRIVATE ingine_model ingine_serve ingine_text ingine_utf8proc CUDA::cudart)
else()
  message(STATUS "ingine-bench: ingine_model not in this tree; skipped")
endif()
