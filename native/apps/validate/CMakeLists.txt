add_executable(ingine-validate main.cpp)
ingine_internal_includes(ingine-validate)
target_compile_definitions(ingine-validate PRIVATE INGINE_REPO_DIR="${PROJECT_SOURCE_DIR}/..")
target_link_libraries(ingine-validate PRIVATE ingine_model nlohmann_json::nlohmann_json)
