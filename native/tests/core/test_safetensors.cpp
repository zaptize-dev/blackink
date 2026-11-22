// Safetensors reader: header parsing, dtypes, metadata, lookup, reads, Tensor views, model
// directories (index.json and single file) and rejection of malformed files. CPU only.

#include "core/safetensors.h"
#include "core/fixture.h"

#include <cstdio>
#include <cstring>
#include <fstream>
#include <functional>
#include <string>
#include <vector>

namespace {

int failures = 0;

#define CHECK(cond)                                                                     \
    do {                                                                                \
        if (!(cond)) {                                                                  \
            std::fprintf(stderr, "%s:%d: check failed: %s\n", __FILE__, __LINE__, #cond); \
            ++failures;                                                                 \
        }                                                                               \
    } while (0)

bool throws(const std::function<void()>& fn) {
    try {
        fn();
    } catch (const std::exception&) { return true; }
    return false;
}

using ingine::DType;
using ingine::SafetensorsFile;
using ingine::SafetensorsModel;
using ingine::test::FixtureTensor;
using ingine::test::pattern_bytes;
using ingine::test::write_safetensors;

std::vector<FixtureTensor> fixture_tensors() {
    return {
        // Data order differs from the (sorted) header order.
        {"model.layers.0.w_fp8", "F8_E4M3", {2, 8}, pattern_bytes(16, 1)},
        {"model.layers.0.w_bf16", "BF16", {3, 4}, pattern_bytes(24, 2)},
        {"model.layers.0.scales", "U8", {5}, pattern_bytes(5, 3)},
        {"model.norm.scale", "F32", {}, pattern_bytes(4, 4)},
        {"model.ids", "I64", {2, 1, 1, 1}, pattern_bytes(16, 5)},
        {"model.empty", "F16", {0, 3}, {}},
        {"model.e8m0", "F8_E8M0", {3}, pattern_bytes(3, 6)},
    };
}

void test_file(const std::filesystem::path& dir) {
    const auto tensors = fixture_tensors();
    const auto path    = dir / "fixture.safetensors";
    write_safetensors(path, tensors, {{"format", "pt"}, {"note", "ingine fixture"}});

    SafetensorsFile f(path);
    CHECK(f.metadata().size() == 2);
    CHECK(f.metadata().at("format") == "pt");
    CHECK(f.metadata().at("note") == "ingine fixture");
    CHECK(f.tensors().size() == tensors.size());
    CHECK(f.data_offset() % 8 == 0);
    CHECK(f.file_size() == std::filesystem::file_size(path));

    // tensors() follows data order (file offsets ascending).
    for (std::size_t i = 1; i < f.tensors().size(); ++i) {
        CHECK(f.tensors()[i - 1].offset <= f.tensors()[i].offset);
    }
    CHECK(f.tensors().front().name == "model.layers.0.w_fp8");

    for (const FixtureTensor& t : tensors) {
        const ingine::SafetensorsTensor* info = f.find(t.name);
        CHECK(info != nullptr);
        if (info == nullptr) { continue; }
        CHECK(info->dtype == t.dtype);
        CHECK(info->shape == t.shape);
        CHECK(info->bytes == t.data.size());
        CHECK(f.contains(t.name));
        const std::vector<std::byte> got = f.read(t.name);
        CHECK(got.size() == t.data.size());
        CHECK(got.empty() || std::memcmp(got.data(), t.data.data(), got.size()) == 0);
    }
    CHECK(f.find("missing") == nullptr);
    CHECK(throws([&] { (void)f.at("missing"); }));
    CHECK(f.names().size() == tensors.size());

    // dtype mapping
    CHECK(f.at("model.layers.0.w_fp8").ingine_dtype() == DType::FP8_E4M3FN);
    CHECK(f.at("model.layers.0.w_bf16").ingine_dtype() == DType::BF16);
    CHECK(f.at("model.layers.0.scales").ingine_dtype() == DType::U8);
    CHECK(f.at("model.norm.scale").ingine_dtype() == DType::FP32);
    CHECK(f.at("model.ids").ingine_dtype() == DType::I64);
    CHECK(throws([&] { (void)f.at("model.e8m0").ingine_dtype(); }));
    CHECK(ingine::safetensors_dtype_size("F8_E8M0") == 1);
    CHECK(ingine::safetensors_dtype_size("BF16") == 2);
    CHECK(throws([] { (void)ingine::safetensors_dtype_size("F4"); }));
    CHECK(!ingine::safetensors_to_dtype("U32").has_value());

    // numel and Tensor views (ne[0] innermost)
    CHECK(f.at("model.layers.0.w_bf16").numel() == 12);
    CHECK(f.at("model.norm.scale").numel() == 1);
    CHECK(f.at("model.empty").numel() == 0);
    int dummy = 0;
    const ingine::Tensor v = f.at("model.layers.0.w_bf16").view(&dummy);
    CHECK(v.data == &dummy);
    CHECK(v.dtype == DType::BF16);
    CHECK(v.ne[0] == 4 && v.ne[1] == 3 && v.ne[2] == 1 && v.ne[3] == 1);
    CHECK(v.nb[0] == 2 && v.nb[1] == 8 && v.nb[2] == 24);
    CHECK(v.is_contiguous());
    CHECK(v.bytes() == 24);
    const ingine::Tensor s = f.at("model.norm.scale").view(&dummy);
    CHECK(s.ne[0] == 1 && s.numel() == 1 && s.dtype == DType::FP32);
    const ingine::Tensor fp8 = f.at("model.layers.0.w_fp8").view(&dummy);
    CHECK(fp8.ne[0] == 8 && fp8.ne[1] == 2 && fp8.nb[1] == 8);
    CHECK(throws([&] { (void)f.at("model.empty").view(&dummy); }));

    // raw range reads
    const ingine::SafetensorsTensor& w = f.at("model.layers.0.w_bf16");
    std::vector<std::uint8_t> part(10);
    f.read_range(w.offset + 3, part.size(), part.data());
    CHECK(std::memcmp(part.data(), tensors[1].data.data() + 3, part.size()) == 0);
    CHECK(throws([&] { f.read_range(f.file_size() - 2, 4, part.data()); }));

    // moves keep lookups working
    SafetensorsFile moved(std::move(f));
    CHECK(moved.find("model.ids") != nullptr);
    CHECK(moved.read("model.ids").size() == 16);
}

void test_model(const std::filesystem::path& dir) {
    // Split checkpoint with an index.
    const auto split = dir / "split";
    std::filesystem::create_directories(split);
    const auto tensors = fixture_tensors();
    std::vector<FixtureTensor> a(tensors.begin(), tensors.begin() + 3);
    std::vector<FixtureTensor> b(tensors.begin() + 3, tensors.end());
    // A tensor present in a file but not in the index is ignored.
    b.push_back({"not.indexed", "U8", {2}, pattern_bytes(2, 9)});
    write_safetensors(split / "model-00001-of-00002.safetensors", a);
    write_safetensors(split / "model-00002-of-00002.safetensors", b);
    nlohmann::json index = {{"metadata", {{"total_size", 0}}}, {"weight_map", nlohmann::json::object()}};
    for (const auto& t : a) { index["weight_map"][t.name] = "model-00001-of-00002.safetensors"; }
    for (std::size_t i = 3; i < tensors.size(); ++i) {
        index["weight_map"][tensors[i].name] = "model-00002-of-00002.safetensors";
    }
    std::ofstream(split / "model.safetensors.index.json") << index.dump(2);
    // Unrelated safetensors files in the same directory (e.g. mtp.safetensors) are not read.
    write_safetensors(split / "mtp.safetensors", {{"mtp.w", "BF16", {1}, pattern_bytes(2, 7)}});

    SafetensorsModel m(split);
    CHECK(m.files().size() == 2);
    CHECK(m.size() == tensors.size());
    CHECK(!m.contains("not.indexed"));
    CHECK(!m.contains("mtp.w"));
    for (const FixtureTensor& t : tensors) {
        const ingine::SafetensorsRef r = m.find(t.name);
        CHECK(static_cast<bool>(r));
        if (!r) { continue; }
        CHECK(r->dtype == t.dtype);
        const auto got = m.read(t.name);
        CHECK(got.empty() || std::memcmp(got.data(), t.data.data(), got.size()) == 0);
    }
    CHECK(m.at("model.ids").file->path().filename() == "model-00002-of-00002.safetensors");
    CHECK(m.at("model.layers.0.scales").file->path().filename() ==
          "model-00001-of-00002.safetensors");
    CHECK(throws([&] { (void)m.at("missing"); }));
    const auto names = m.names();
    CHECK(std::is_sorted(names.begin(), names.end()));

    SafetensorsModel moved(std::move(m));
    CHECK(moved.at("model.ids")->bytes == 16);

    // The index file path itself, and a single file path.
    SafetensorsModel by_index(split / "model.safetensors.index.json");
    CHECK(by_index.size() == tensors.size());
    SafetensorsModel single_file(split / "mtp.safetensors");
    CHECK(single_file.size() == 1 && single_file.contains("mtp.w"));

    // Directory with only model.safetensors.
    const auto single = dir / "single";
    std::filesystem::create_directories(single);
    write_safetensors(single / "model.safetensors", tensors);
    SafetensorsModel s(single);
    CHECK(s.files().size() == 1 && s.size() == tensors.size());

    // Index naming a tensor its file lacks.
    const auto broken = dir / "broken";
    std::filesystem::create_directories(broken);
    write_safetensors(broken / "a.safetensors", a);
    std::ofstream(broken / "model.safetensors.index.json")
        << nlohmann::json{{"weight_map", {{"model.ids", "a.safetensors"}}}}.dump();
    CHECK(throws([&] { SafetensorsModel bad(broken); }));
    CHECK(throws([&] { SafetensorsModel none(dir / "does-not-exist"); }));
}

void write_raw(const std::filesystem::path& path, std::uint64_t declared_len, const std::string& header,
               std::size_t data_bytes) {
    std::ofstream out(path, std::ios::binary | std::ios::trunc);
    unsigned char len[8];
    for (int i = 0; i < 8; ++i) { len[i] = static_cast<unsigned char>(declared_len >> (8 * i)); }
    out.write(reinterpret_cast<const char*>(len), 8);
    out << header;
    out << std::string(data_bytes, '\0');
}

void test_malformed(const std::filesystem::path& dir) {
    const auto p = dir / "bad.safetensors";
    const std::string ok = R"({"t":{"dtype":"U8","shape":[4],"data_offsets":[0,4]}})";

    write_raw(p, ok.size(), ok, 4);
    CHECK(!throws([&] { SafetensorsFile f(p); }));

    write_raw(p, ok.size() + 100, ok, 4); // header length beyond the file
    CHECK(throws([&] { SafetensorsFile f(p); }));

    write_raw(p, ok.size(), ok, 3); // data region too short
    CHECK(throws([&] { SafetensorsFile f(p); }));

    const std::string mismatch = R"({"t":{"dtype":"BF16","shape":[4],"data_offsets":[0,4]}})";
    write_raw(p, mismatch.size(), mismatch, 8); // 4 BF16 need 8 bytes
    CHECK(throws([&] { SafetensorsFile f(p); }));

    const std::string not_json = "{not json";
    write_raw(p, not_json.size(), not_json, 0);
    CHECK(throws([&] { SafetensorsFile f(p); }));

    const std::string bad_meta = R"({"__metadata__":{"k":1}})";
    write_raw(p, bad_meta.size(), bad_meta, 0);
    CHECK(throws([&] { SafetensorsFile f(p); }));

    const std::string reversed = R"({"t":{"dtype":"U8","shape":[0],"data_offsets":[4,0]}})";
    write_raw(p, reversed.size(), reversed, 4);
    CHECK(throws([&] { SafetensorsFile f(p); }));

    std::ofstream(p, std::ios::binary | std::ios::trunc) << "abc"; // shorter than 8 bytes
    CHECK(throws([&] { SafetensorsFile f(p); }));
    CHECK(throws([&] { SafetensorsFile f(dir / "missing.safetensors"); }));
}

} // namespace

int main() {
    ingine::test::TempDir tmp;
    test_file(tmp.path);
    test_model(tmp.path);
    test_malformed(tmp.path);
    if (failures != 0) {
        std::fprintf(stderr, "%d check(s) failed\n", failures);
        return 1;
    }
    std::printf("PASS core_safetensors\n");
    return 0;
}
