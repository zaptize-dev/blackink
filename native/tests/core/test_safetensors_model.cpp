// Opens the real checkpoint headers (no tensor data is read): model.safetensors.index.json and
// mtp.safetensors. Model directory: $INGINE_MODEL_DIR, else the configured default; skipped (77)
// when absent. CPU only.

#include "core/safetensors.h"

#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <map>
#include <string>

int main() {
    const char* env               = std::getenv("INGINE_MODEL_DIR");
    const std::filesystem::path dir = env != nullptr && *env != '\0' ? env : INGINE_MODEL_DIR;
    if (!std::filesystem::exists(dir / "model.safetensors.index.json")) {
        std::printf("skip: no model.safetensors.index.json in %s\n", dir.c_str());
        return 77;
    }
    int failures = 0;
    auto check   = [&](bool ok, const std::string& what) {
        if (!ok) {
            std::fprintf(stderr, "check failed: %s\n", what.c_str());
            ++failures;
        }
    };

    const ingine::SafetensorsModel model(dir);
    std::map<std::string, int> dtypes;
    std::uint64_t total = 0;
    for (const std::string& name : model.names()) {
        const ingine::SafetensorsRef r = model.at(name);
        ++dtypes[r->dtype];
        total += r->bytes;
        check(r->bytes > 0, name + " is empty");
        check(ingine::safetensors_to_dtype(r->dtype).has_value(), name + " has dtype " + r->dtype);
    }
    std::printf("%zu tensors in %zu file(s), %.2f GiB:", model.size(), model.files().size(),
                double(total) / double(1ull << 30));
    for (const auto& [dtype, n] : dtypes) { std::printf(" %s=%d", dtype.c_str(), n); }
    std::printf("\n");
    check(model.size() > 0, "model has tensors");
    check(dtypes.count("F8_E4M3") != 0, "FP8 tensors present");

    if (std::filesystem::exists(dir / "mtp.safetensors")) {
        const ingine::SafetensorsFile mtp(dir / "mtp.safetensors");
        std::printf("mtp.safetensors: %zu tensors\n", mtp.tensors().size());
        check(!mtp.tensors().empty(), "mtp has tensors");
    }
    if (failures != 0) { return 1; }
    std::printf("PASS core_safetensors_model\n");
    return 0;
}
