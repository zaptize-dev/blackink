// DeviceArena / DeviceBuffer / PinnedHostBuffer basics and Tensor views (a few MiB of GPU memory).

#include "common/test_util.h"
#include "core/arena.h"
#include "core/device.h"
#include "core/tensor.h"

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <functional>
#include <new>
#include <numeric>
#include <vector>

namespace {

using ingine::DeviceArena;
using ingine::DeviceBuffer;
using ingine::DType;
using ingine::Tensor;

bool throws(const std::function<void()>& fn) {
    try {
        fn();
    } catch (...) { return true; }
    return false;
}

std::uintptr_t addr(const void* p) { return reinterpret_cast<std::uintptr_t>(p); }

void test_tensor() {
    int dummy = 0;
    Tensor t(&dummy, DType::BF16, {8, 4, 2});
    INGINE_CHECK(t.numel() == 64 && t.bytes() == 128);
    INGINE_CHECK(t.nb[0] == 2 && t.nb[1] == 16 && t.nb[2] == 64 && t.nb[3] == 128);
    INGINE_CHECK(t.is_contiguous());
    const Tensor r = t.reshape({32, 2});
    INGINE_CHECK(r.ne[0] == 32 && r.ne[1] == 2 && r.ne[2] == 1);
    const Tensor s = t.slice(1, 1, 2);
    INGINE_CHECK(addr(s.data) == addr(&dummy) + 16 && s.ne[1] == 2 && !s.is_contiguous());
    INGINE_CHECK(throws([&] { (void)t.reshape({7}); }));
    INGINE_CHECK(ingine::dtype_size(DType::FP8_E4M3FN) == 1);
    INGINE_CHECK(std::strcmp(ingine::dtype_name(DType::FP8_E4M3FN), "fp8_e4m3fn") == 0);
}

void test_arena() {
    DeviceArena arena(std::size_t(1) << 20);
    INGINE_CHECK(arena.capacity() == (std::size_t(1) << 20));
    INGINE_CHECK(arena.used() == 0);

    const ingine::DeviceSpan a = arena.alloc_bytes(100);
    INGINE_CHECK(a.data == arena.base() && a.bytes == 100);
    const ingine::DeviceSpan b = arena.alloc_bytes(10, 512);
    INGINE_CHECK(addr(b.data) % 512 == 0 && addr(b.data) >= addr(a.data) + 100);
    INGINE_CHECK(throws([&] { (void)arena.alloc_bytes(8, 3); }));
    INGINE_CHECK(throws([&] { (void)arena.alloc_bytes(0); }));

    const std::size_t before_scope = arena.used();
    {
        auto scope        = arena.scope();
        const Tensor x    = arena.alloc(DType::FP32, {256, 4});
        INGINE_CHECK(x.bytes() == 4096 && addr(x.data) % 256 == 0);
        INGINE_CHECK(arena.used() > before_scope);
    }
    INGINE_CHECK(arena.used() == before_scope);
    INGINE_CHECK(arena.peak_used() >= before_scope + 4096);

    bool bad_alloc = false;
    try {
        (void)arena.alloc_bytes(std::size_t(2) << 20);
    } catch (const std::bad_alloc&) { bad_alloc = true; }
    INGINE_CHECK(bad_alloc);
    INGINE_CHECK(arena.used() == before_scope);

    // Data written through arena allocations round-trips.
    const Tensor y = arena.alloc(DType::I32, {1000});
    std::vector<std::int32_t> host(1000);
    std::iota(host.begin(), host.end(), -500);
    INGINE_TEST_CUDA(cudaMemcpy(y.data, host.data(), y.bytes(), cudaMemcpyHostToDevice));
    std::vector<std::int32_t> back(1000);
    INGINE_TEST_CUDA(cudaMemcpy(back.data(), y.data, y.bytes(), cudaMemcpyDeviceToHost));
    INGINE_CHECK(back == host);

    arena.reset();
    INGINE_CHECK(arena.used() == 0);
    arena.reset_peak();
    INGINE_CHECK(arena.peak_used() == 0);

    // Moves transfer ownership.
    DeviceArena moved(std::move(arena));
    INGINE_CHECK(moved.capacity() == (std::size_t(1) << 20));
    INGINE_CHECK(arena.base() == nullptr);
    INGINE_CHECK(throws([&] { (void)arena.alloc_bytes(16); }));
}

void test_buffers() {
    DeviceBuffer buf(4096);
    buf.fill(0x5a);
    std::vector<std::uint8_t> host(4096);
    buf.copy_to_host(host.data(), host.size());
    for (std::uint8_t v : host) { INGINE_CHECK(v == 0x5a); }
    std::vector<std::uint8_t> src(100);
    std::iota(src.begin(), src.end(), std::uint8_t(1));
    buf.copy_from_host(src.data(), src.size(), 1000);
    buf.copy_to_host(host.data(), host.size());
    INGINE_CHECK(std::memcmp(host.data() + 1000, src.data(), src.size()) == 0);
    INGINE_CHECK(host[999] == 0x5a && host[1100] == 0x5a);
    INGINE_CHECK(throws([&] { buf.copy_from_host(src.data(), src.size(), 4000); }));

    // A borrowed arena over the buffer does not free it.
    {
        DeviceArena borrowed(ingine::DeviceSpan{buf.p, buf.bytes});
        INGINE_CHECK(borrowed.alloc_bytes(64).data == buf.p);
    }
    buf.fill(0);

    ingine::PinnedHostBuffer pinned(1 << 16);
    INGINE_CHECK(pinned.data() != nullptr && pinned.size() == (1u << 16));
    cudaPointerAttributes attrs{};
    INGINE_TEST_CUDA(cudaPointerGetAttributes(&attrs, pinned.data()));
    INGINE_CHECK(attrs.type == cudaMemoryTypeHost);
}

} // namespace

int main() {
    test_tensor();
    if (ingine::test::skip_without_gpu()) { return ingine::test::kSkip; }
    ingine::DeviceContext ctx(0);
    test_arena();
    test_buffers();
    std::printf("PASS core_arena\n");
    return 0;
}
