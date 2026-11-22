// Shared helpers of the text golden tests (CPU only).
#pragma once

#include <nlohmann/json.hpp>

#include <cstdio>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <string_view>

namespace ingine::text::test {

inline constexpr int kSkip = 77;

inline nlohmann::ordered_json load_golden(const char* name) {
  const std::filesystem::path path = std::filesystem::path(INGINE_GOLDEN_DIR) / "text" / name;
  std::ifstream in(path, std::ios::binary);
  if (!in) {
    std::fprintf(stderr, "missing golden %s (run tools/goldens_text.py)\n", path.c_str());
    std::exit(1);
  }
  std::stringstream ss;
  ss << in.rdbuf();
  return nlohmann::ordered_json::parse(ss.str());
}

// Model directory with tokenizer.json and chat_template.jinja, or empty (test skipped).
inline std::filesystem::path model_dir() {
  std::filesystem::path dir = INGINE_MODEL_DIR;
  if (const char* env = std::getenv("INGINE_MODEL_DIR"); env != nullptr && *env) dir = env;
  if (!std::filesystem::exists(dir / "tokenizer.json") || !std::filesystem::exists(dir / "chat_template.jinja")) {
    std::printf("skip: no tokenizer.json/chat_template.jinja in %s\n", dir.c_str());
    return {};
  }
  return dir;
}

// Quoted, escaped rendering for failure messages.
inline std::string show(std::string_view s) { return nlohmann::json(std::string(s)).dump(-1, ' ', true, nlohmann::json::error_handler_t::replace); }

struct Checker {
  int checks = 0, failures = 0;
  bool expect(bool ok, const std::string& what) {
    ++checks;
    if (!ok) {
      ++failures;
      if (failures <= 40) std::cerr << "FAIL: " << what << "\n";
    }
    return ok;
  }
  int finish(const char* name) const {
    std::printf("%s: %d checks, %d failures\n", name, checks, failures);
    return failures == 0 ? 0 : 1;
  }
};

}  // namespace ingine::text::test
