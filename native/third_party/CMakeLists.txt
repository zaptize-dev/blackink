# Adapted from fngine (github.com/satellitedown/fngine, Apache-2.0) for ingine.
# Repository-pinned third-party libraries. No configure-time downloads, no system packages.

# nlohmann json (header only): #include <nlohmann/json.hpp>
add_library(ingine_nlohmann_json INTERFACE)
target_include_directories(ingine_nlohmann_json SYSTEM INTERFACE ${CMAKE_CURRENT_SOURCE_DIR})
add_library(nlohmann_json::nlohmann_json ALIAS ingine_nlohmann_json)

# cpp-httplib (header only): #include <httplib.h>
add_library(ingine_httplib INTERFACE)
target_include_directories(ingine_httplib SYSTEM INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}/cpp-httplib)
target_link_libraries(ingine_httplib INTERFACE Threads::Threads)
add_library(httplib::httplib ALIAS ingine_httplib)

# utf8proc (compiled, static): #include <utf8proc/utf8proc.h>
add_library(ingine_utf8proc STATIC utf8proc/utf8proc.c)
target_include_directories(ingine_utf8proc SYSTEM PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_compile_definitions(ingine_utf8proc PUBLIC UTF8PROC_STATIC)

# spdlog with bundled fmt: spdlog::spdlog
add_subdirectory(spdlog)

# llama.cpp's Jinja runtime: ingine_jinja (#include <jinja/runtime.h>)
add_subdirectory(llama-jinja)
