# ingine_engine: the request scheduler (one running job + bounded FIFO) over model::Generator, the canned-output
# FakeGenerator and the server's logger. CUDA-free.
add_library(ingine_engine STATIC
  fake_generator.cpp
  log.cpp
  scheduler.cpp
)
ingine_internal_includes(ingine_engine)
target_include_directories(ingine_engine PUBLIC
  ${PROJECT_SOURCE_DIR}/include ${PROJECT_SOURCE_DIR}/src ${PROJECT_SOURCE_DIR}/third_party)
target_link_libraries(ingine_engine
  PUBLIC ingine_text nlohmann_json::nlohmann_json spdlog::spdlog Threads::Threads)
