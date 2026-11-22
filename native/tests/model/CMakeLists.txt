ingine_add_test(NAME model_data SOURCES test_model_data.cpp LIBS ingine_core)
# one real layer at a time (~1.1 GB of GPU memory)
ingine_add_test(NAME model_layer SOURCES test_layer.cpp LIBS ingine_model GPU MODEL TIMEOUT 600)
# the decode runtime (graphs, verify widths, rollback, pipelined steps, device-selected verify) and the Generator's
# stop/cancel/budget exactness on a synthetic model (< 2 GB of GPU memory)
ingine_add_test(NAME model_runtime SOURCES test_runtime.cpp LIBS ingine_model GPU TIMEOUT 600)
# prompt checkpoints with speculative decoding on the full model (~22 GB of GPU memory)
ingine_add_test(NAME model_reuse SOURCES test_reuse.cpp LIBS ingine_model GPU MODEL TIMEOUT 900)
# full-model parity against the Python engine's goldens (tools/goldens_model.py); ~20 GB of GPU memory
if(TARGET ingine-validate)
  set(_validate ingine-validate --model-dir ${INGINE_TEST_MODEL_DIR} --golden ${PROJECT_SOURCE_DIR}/tests/golden/model)
  add_test(NAME model_parity COMMAND ${_validate})
  # exact greedy continuations only where the goldens have no near ties (long and FP8-cache cases flip at ties)
  set(_greedy_cases --cases kv16_chat,kv16_t300 --min-top1 1)
  add_test(NAME model_parity_generate COMMAND ${_validate} --generate ${_greedy_cases})
  add_test(NAME model_parity_spec COMMAND ${_validate} --generate --spec ${_greedy_cases})
  # long greedy paths of the Generator teacher-forced through a plain width-1 runtime: plain decoding exact;
  # speculative decoding as a rate over the four FP8-cache paths (12,000 tokens): single positions move by
  # several logits between kernel widths (30,000 tokens over all golden cases: 13-14 above 0.5 logits, 3 above
  # 2, in old and new verify kernels alike), a real bug moves most tokens
  add_test(NAME model_path COMMAND ${_validate} --path 1500 --cases abl_kv8_t300)
  add_test(NAME model_path_spec COMMAND ${_validate} --path 3000 --spec --cases abl_kv8 --max-gap-rate 15)
  set_tests_properties(model_parity model_parity_generate model_parity_spec model_path model_path_spec PROPERTIES
    LABELS "gpu;model" RESOURCE_LOCK gpu TIMEOUT 3600)
endif()
