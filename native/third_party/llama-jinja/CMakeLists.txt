# Adapted from fngine (github.com/satellitedown/fngine, Apache-2.0) for ingine.
add_library(ingine_jinja STATIC
  jinja/json.cpp
  jinja/lexer.cpp
  jinja/parser.cpp
  jinja/runtime.cpp
  jinja/string.cpp
  jinja/value.cpp
  jinja/unicode.cpp
  jinja/utf8_support.cpp
)
target_include_directories(ingine_jinja PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(ingine_jinja PUBLIC ingine_utf8proc nlohmann_json::nlohmann_json)
