ingine_add_test(NAME spec_draft_kernels SOURCES test_draft_kernels.cpp LIBS ingine_kernels_draft CUDA::cudart
                GPU ORACLE TIMEOUT 300)
ingine_add_test(NAME spec_accept SOURCES test_accept.cpp LIBS ingine_kernels_draft CUDA::cudart GPU ORACLE
                TIMEOUT 300)
# Sampling fidelity at the full vocabulary: plain draws and the speculative accept + residual path against the
# exact tail mass of golden rows (one-CTA and multi-block kernels; ~200 MB of GPU memory).
ingine_add_test(NAME spec_sampling_fidelity SOURCES test_sampling_fidelity.cpp
                LIBS ingine_kernels_draft ingine_kernels_head nlohmann_json::nlohmann_json CUDA::cudart GPU TIMEOUT 900)
# The Drafter with the real mtp.safetensors (about 1.2 GB of GPU memory) and a synthetic 4096-token target.
# drafter.cpp is compiled in directly so the test needs only the dense and draft kernels.
if(TARGET ingine_kernels_dense)
  ingine_add_test(NAME spec_drafter
                  SOURCES test_drafter.cpp ${PROJECT_SOURCE_DIR}/src/model/drafter.cpp
                  LIBS ingine_kernels_draft ingine_kernels_dense ingine_core CUDA::cudart
                  GPU MODEL TIMEOUT 600)
endif()
# Teacher-forced replay through the Generator (full model).
if(TARGET ingine_model)
  ingine_add_test(NAME spec_forced SOURCES test_forced.cpp LIBS ingine_model CUDA::cudart GPU MODEL TIMEOUT 600)
endif()
