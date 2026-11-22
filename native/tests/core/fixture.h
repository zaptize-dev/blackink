#pragma once

// Writes small safetensors files for the core tests (same layout as the reference writer: header
// padded with spaces to a multiple of 8 bytes, tensors stored back to back).

#include <nlohmann/json.hpp>

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <map>
#include <stdexcept>
#include <string>
#include <vector>

namespace ingine::test {

struct FixtureTensor {
    std::string name;
    std::string dtype;
    std::vector<std::int64_t> shape;
    std::vector<std::uint8_t> data;
};

// Tensors are stored in the order given; the JSON header lists them sorted by name (as
// serde_json/nlohmann do), so header order and data order differ.
inline void write_safetensors(const std::filesystem::path& path,
                              const std::vector<FixtureTensor>& tensors,
                              const std::map<std::string, std::string>& metadata = {}) {
    nlohmann::json header = nlohmann::json::object();
    if (!metadata.empty()) { header["__metadata__"] = metadata; }
    std::uint64_t offset = 0;
    for (const FixtureTensor& t : tensors) {
        header[t.name] = {{"dtype", t.dtype},
                          {"shape", t.shape},
                          {"data_offsets", {offset, offset + t.data.size()}}};
        offset += t.data.size();
    }
    std::string text = header.dump();
    while (text.size() % 8 != 0) { text += ' '; }
    std::ofstream out(path, std::ios::binary | std::ios::trunc);
    if (!out) { throw std::runtime_error("cannot write " + path.string()); }
    std::uint64_t n = text.size();
    unsigned char len[8];
    for (int i = 0; i < 8; ++i) { len[i] = static_cast<unsigned char>(n >> (8 * i)); }
    out.write(reinterpret_cast<const char*>(len), 8);
    out.write(text.data(), static_cast<std::streamsize>(text.size()));
    for (const FixtureTensor& t : tensors) {
        out.write(reinterpret_cast<const char*>(t.data.data()),
                  static_cast<std::streamsize>(t.data.size()));
    }
    if (!out) { throw std::runtime_error("short write to " + path.string()); }
}

// A fresh temporary directory, removed when the object goes away.
struct TempDir {
    std::filesystem::path path;
    TempDir() {
        std::string templ = (std::filesystem::temp_directory_path() / "ingine-test-XXXXXX").string();
        if (::mkdtemp(templ.data()) == nullptr) { throw std::runtime_error("mkdtemp failed"); }
        path = templ;
    }
    ~TempDir() {
        std::error_code ec;
        std::filesystem::remove_all(path, ec);
    }
    TempDir(const TempDir&)            = delete;
    TempDir& operator=(const TempDir&) = delete;
};

inline std::vector<std::uint8_t> pattern_bytes(std::size_t n, std::uint64_t seed) {
    std::vector<std::uint8_t> out(n);
    std::uint64_t s = seed * 0x9E3779B97F4A7C15ull + 1;
    for (std::size_t i = 0; i < n; ++i) {
        s ^= s << 13;
        s ^= s >> 7;
        s ^= s << 17;
        out[i] = static_cast<std::uint8_t>(s);
    }
    return out;
}

} // namespace ingine::test
