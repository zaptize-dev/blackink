# Decode kernel tests against FP64 host oracles (synthetic weights at the model's shapes, < 1 GB of GPU memory).
ingine_add_test(NAME kernels_dense SOURCES test_dense.cpp LIBS ingine_kernels_dense GPU ORACLE TIMEOUT 600)
ingine_add_test(NAME kernels_moe SOURCES test_moe.cpp LIBS ingine_kernels_moe GPU ORACLE TIMEOUT 600)
ingine_add_test(NAME kernels_attn SOURCES test_attn.cpp LIBS ingine_kernels_attn GPU ORACLE TIMEOUT 900)
ingine_add_test(NAME kernels_head SOURCES test_head.cpp LIBS ingine_kernels_head GPU ORACLE TIMEOUT 600)
# every output bit of the M-wide weight-streaming and sliding-attention kernels (~1 GB of GPU memory)
ingine_add_test(NAME kernels_bitlock SOURCES test_bitlock.cpp
  LIBS ingine_kernels_dense ingine_kernels_moe ingine_kernels_attn GPU TIMEOUT 600)
