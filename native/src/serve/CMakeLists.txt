# ingine_serve: the OpenAI-compatible HTTP server (cpp-httplib, SSE) over the scheduler. CUDA-free.
add_library(ingine_serve STATIC
  chat_server.cpp
  cli.cpp
  http_transport.cpp
  openai.cpp
)
ingine_internal_includes(ingine_serve)
target_link_libraries(ingine_serve
  PUBLIC ingine_engine ingine_text httplib::httplib nlohmann_json::nlohmann_json spdlog::spdlog Threads::Threads)
