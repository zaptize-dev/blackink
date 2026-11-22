// DeviceUploader round trip: a ~9 MiB fixture checkpoint uploaded with small staging chunks (many
// chunks per tensor, more chunks than staging buffers) and compared byte for byte; raw ranges,
// reuse, empty requests and invalid requests.
//
// Optional bandwidth run: INGINE_UPLOAD_BENCH=<.safetensors file or model dir> streams every tensor
// through a 1 GiB device buffer (batches of <= 1 GiB) and prints GiB/s (cold or page-cache speed).

#include "common/test_util.h"
#include "core/arena.h"
#include "core/device.h"
#include "core/fixture.h"
#include "core/safetensors.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <vector>

namespace {

using ingine::DeviceBuffer;
using ingine::DeviceUploader;
using ingine::SafetensorsFile;
using ingine::UploadRequest;
using ingine::test::FixtureTensor;
using ingine::test::pattern_bytes;

bool throws(const std::function<void()>& fn) {
    try {
        fn();
    } catch (...) { return true; }
    return false;
}

void round_trip(const std::filesystem::path& dir, cudaStream_t stream) {
    std::vector<FixtureTensor> tensors = {
        {"a.bf16", "BF16", {1024, 1500}, pattern_bytes(1024 * 1500 * 2, 11)}, // ~2.9 MiB
        {"b.fp8", "F8_E4M3", {3000, 1001}, pattern_bytes(3000 * 1001, 12)},  // ~2.9 MiB
        {"c.u8", "U8", {7}, pattern_bytes(7, 13)},
        {"d.f32", "F32", {777, 1000}, pattern_bytes(777 * 1000 * 4, 14)},     // ~3 MiB
        {"e.scale", "F32", {}, pattern_bytes(4, 15)},
    };
    const auto path = dir / "upload.safetensors";
    ingine::test::write_safetensors(path, tensors, {{"format", "pt"}});
    const SafetensorsFile file(path);

    // Pack every tensor into one device buffer at odd (unaligned) offsets.
    std::vector<std::size_t> dst_off;
    std::size_t total = 0;
    for (const FixtureTensor& t : tensors) {
        dst_off.push_back(total);
        total += t.data.size() + 3;
    }
    DeviceBuffer dev(total);
    dev.fill(0);

    ingine::UploadOptions opts;
    opts.chunk_bytes     = 1 << 20;
    opts.staging_buffers = 3;
    opts.reader_threads  = 3;
    DeviceUploader up(opts);

    std::vector<UploadRequest> reqs;
    for (std::size_t i = 0; i < tensors.size(); ++i) {
        reqs.push_back(ingine::upload_request(file, file.at(tensors[i].name),
                                              static_cast<std::byte*>(dev.p) + dst_off[i]));
    }
    reqs.push_back(UploadRequest{&file, 0, 0, nullptr}); // empty requests are ignored

    for (int round = 0; round < 2; ++round) { // the uploader is reusable
        const ingine::UploadStats st = up.upload(reqs, stream);
        std::uint64_t expect = 0;
        for (const FixtureTensor& t : tensors) { expect += t.data.size(); }
        INGINE_CHECK(st.bytes == expect);
        std::vector<std::uint8_t> back(total);
        dev.copy_to_host(back.data(), back.size());
        for (std::size_t i = 0; i < tensors.size(); ++i) {
            INGINE_CHECK(std::memcmp(back.data() + dst_off[i], tensors[i].data.data(),
                                     tensors[i].data.size()) == 0);
            INGINE_CHECK(back[dst_off[i] + tensors[i].data.size()] == 0); // gap untouched
        }
        if (round == 0) { dev.fill(0); }
    }

    // A raw range spanning two tensors' data.
    const auto& a = file.at("a.bf16");
    std::vector<std::uint8_t> expect(a.bytes + 1000);
    file.read_range(a.offset, expect.size(), expect.data());
    DeviceBuffer dev2(expect.size());
    const UploadRequest range{&file, a.offset, expect.size(), dev2.p};
    up.upload({&range, 1}, stream);
    std::vector<std::uint8_t> back2(expect.size());
    dev2.copy_to_host(back2.data(), back2.size());
    INGINE_CHECK(back2 == expect);

    // SafetensorsRef overload via a model over the single file.
    const ingine::SafetensorsModel model(path);
    DeviceBuffer dev3(4);
    const UploadRequest scale = ingine::upload_request(model.at("e.scale"), dev3.p);
    up.upload({&scale, 1}, stream);
    std::uint8_t got[4];
    dev3.copy_to_host(got, 4);
    INGINE_CHECK(std::memcmp(got, tensors[4].data.data(), 4) == 0);

    // Invalid requests are rejected before anything is copied; the uploader stays usable.
    const UploadRequest past_end{&file, file.file_size() - 2, 4, dev2.p};
    INGINE_CHECK(throws([&] { up.upload({&past_end, 1}, stream); }));
    const UploadRequest no_dst{&file, 0, 4, nullptr};
    INGINE_CHECK(throws([&] { up.upload({&no_dst, 1}, stream); }));
    up.upload({&scale, 1}, stream);
    INGINE_CHECK(throws([] { DeviceUploader bad(ingine::UploadOptions{0, 1, 1}); }));
}

int bench(const char* target, cudaStream_t stream) {
    const ingine::SafetensorsModel model{std::filesystem::path(target)};
    constexpr std::size_t kWindow = std::size_t(1) << 30;
    DeviceBuffer dev(kWindow);
    // Tuning knobs: INGINE_UPLOAD_CHUNK_MIB, INGINE_UPLOAD_BUFFERS, INGINE_UPLOAD_READERS.
    ingine::UploadOptions opts;
    auto env_int = [](const char* name, long fallback) {
        const char* v = std::getenv(name);
        return v != nullptr && *v != '\0' ? std::strtol(v, nullptr, 10) : fallback;
    };
    opts.chunk_bytes     = std::size_t(env_int("INGINE_UPLOAD_CHUNK_MIB", long(opts.chunk_bytes >> 20))) << 20;
    opts.staging_buffers = int(env_int("INGINE_UPLOAD_BUFFERS", opts.staging_buffers));
    opts.reader_threads  = int(env_int("INGINE_UPLOAD_READERS", opts.reader_threads));
    DeviceUploader up(opts);
    ingine::UploadStats total;
    std::vector<UploadRequest> batch;
    std::size_t used = 0;
    auto flush       = [&] {
        const ingine::UploadStats s = up.upload(batch, stream);
        total.bytes += s.bytes;
        total.seconds += s.seconds;
        batch.clear();
        used = 0;
    };
    for (const std::string& name : model.names()) {
        const ingine::SafetensorsRef r = model.at(name);
        std::uint64_t off = 0;
        while (off < r->bytes) { // tensors larger than the window go in window-sized pieces
            if (used == kWindow) { flush(); }
            const std::uint64_t n = std::min<std::uint64_t>(r->bytes - off, kWindow - used);
            batch.push_back(UploadRequest{r.file, r->offset + off, n,
                                          static_cast<std::byte*>(dev.p) + used});
            used += n;
            off += n;
        }
    }
    if (!batch.empty()) { flush(); }
    std::printf("upload bench: %.2f GiB in %.3f s = %.2f GiB/s (chunk %zu MiB, %d buffers, %d readers)\n",
                double(total.bytes) / double(1ull << 30), total.seconds, total.gib_per_second(),
                up.options().chunk_bytes >> 20, up.options().staging_buffers,
                up.options().reader_threads);
    return 0;
}

} // namespace

int main() {
    if (ingine::test::skip_without_gpu()) { return ingine::test::kSkip; }
    ingine::DeviceContext ctx(0);
    ingine::test::TempDir tmp;
    round_trip(tmp.path, ctx.transfer_stream);
    if (const char* target = std::getenv("INGINE_UPLOAD_BENCH"); target != nullptr && *target) {
        return bench(target, ctx.transfer_stream);
    }
    std::printf("PASS core_upload\n");
    return 0;
}
