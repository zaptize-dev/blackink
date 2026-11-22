# The shortlist ranking is embedded as a generated source (no runtime data file).
set(_shortlist_bin ${CMAKE_CURRENT_SOURCE_DIR}/draft_shortlist.bin)
set(_shortlist_cpp ${CMAKE_CURRENT_BINARY_DIR}/draft_shortlist_data.cpp)
set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS ${_shortlist_bin})
file(READ ${_shortlist_bin} _shortlist_hex HEX)
string(REGEX REPLACE "([0-9a-f][0-9a-f])" "0x\\1," _shortlist_bytes "${_shortlist_hex}")
file(WRITE ${_shortlist_cpp}.tmp
  "// Generated from draft_shortlist.bin by src/kernels/draft/CMakeLists.txt.\n"
  "#include \"kernels/draft/shortlist.h\"\n"
  "namespace ingine::kernels::draft {\nnamespace {\n"
  "alignas(4) const unsigned char kBytes[] = {${_shortlist_bytes}};\n}\n"
  "std::span<const std::uint32_t> draft_shortlist_ranks() {\n"
  "  return {reinterpret_cast<const std::uint32_t*>(kBytes), sizeof(kBytes) / 4};  // little-endian host\n}\n}\n")
file(COPY_FILE ${_shortlist_cpp}.tmp ${_shortlist_cpp} ONLY_IF_DIFFERENT)

add_library(ingine_kernels_draft STATIC
  accept.cu
  attention.cu
  block.cu
  fold.cu
  gemm.cpp
  markov.cu
  ${_shortlist_cpp}
)
ingine_cuda_target(ingine_kernels_draft)
ingine_internal_includes(ingine_kernels_draft)
target_link_libraries(ingine_kernels_draft PUBLIC ingine_core CUDA::cudart CUDA::cublasLt)
