# ingine_text: tokenizer, Jinja adapter, ChatFormat, StreamParser (CUDA-free).
add_library(ingine_text STATIC
  unicode.cpp
  tokenizer.cpp
  pyjson.cpp
  jinja.cpp
  chat.cpp
)
ingine_internal_includes(ingine_text)
target_include_directories(ingine_text PUBLIC ${PROJECT_SOURCE_DIR}/src)
target_link_libraries(ingine_text PUBLIC nlohmann_json::nlohmann_json PRIVATE ingine_jinja ingine_utf8proc)
