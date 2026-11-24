find_package(OpenMP)

add_library(ingine_vision STATIC
  vision.cpp
)
ingine_internal_includes(ingine_vision)
target_include_directories(ingine_vision PUBLIC
  ${PROJECT_SOURCE_DIR}/src ${PROJECT_SOURCE_DIR}/third_party)
target_compile_options(ingine_vision PRIVATE -O3 -mavx2 -mfma)
target_link_libraries(ingine_vision
  PUBLIC nlohmann_json::nlohmann_json
  PRIVATE $<$<BOOL:${OpenMP_CXX_FOUND}>:OpenMP::OpenMP_CXX>)
