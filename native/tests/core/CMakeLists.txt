ingine_add_test(NAME core_safetensors SOURCES test_safetensors.cpp LIBS ingine_core)
ingine_add_test(NAME core_safetensors_model SOURCES test_safetensors_model.cpp LIBS ingine_core
  LABELS model)
ingine_add_test(NAME core_arena SOURCES test_arena.cpp LIBS ingine_core GPU)
ingine_add_test(NAME core_upload SOURCES test_upload.cpp LIBS ingine_core GPU)
