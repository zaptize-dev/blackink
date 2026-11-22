// Adapted from fngine (github.com/satellitedown/fngine, Apache-2.0) for ingine.
// Modified for fngine from cinference:src/core/nvtx.h @ 383e5db. See NOTICE.
// ingine: categories and range names for the Inkling-2-Mini engine.
#pragma once

#include <nvtx3/nvToolsExt.h>

#include <array>
#include <cstddef>
#include <cstdint>

namespace ingine::nvtx {

enum class Category : std::uint32_t {
    Runtime = 1,
    Prefill,
    Decode,
    Spec,
    Draft,
    Attention,
    Moe,
    Head,
    Control,
    Graph,
    Load,
};

enum class Name : std::size_t {
    Generate,
    Request,
    Prefill,
    PrefillChunk,
    Decode,
    DecodeStep,
    SpecRound,
    Draft,
    DraftCommit,
    Verify,
    Accept,
    Rollback,
    Sample,
    EngineLoad,
    WeightRead,
    WeightUpload,
    CudaGraphCapture,
    CudaGraphInstantiate,
    CudaGraphUpdate,
    CudaGraphUpload,
    CudaGraphLaunch,
    Count,
};

[[nodiscard]] inline std::uint32_t color(Category category) noexcept {
    switch (category) {
    case Category::Runtime:
        return 0xff4c78a8u;
    case Category::Prefill:
        return 0xff59a14fu;
    case Category::Decode:
        return 0xfff28e2bu;
    case Category::Spec:
        return 0xffb279a2u;
    case Category::Draft:
        return 0xffaf7aa1u;
    case Category::Attention:
        return 0xff76b7b2u;
    case Category::Moe:
        return 0xffb07aa1u;
    case Category::Head:
        return 0xffedc948u;
    case Category::Control:
        return 0xff9c9c9cu;
    case Category::Graph:
        return 0xff79706eu;
    case Category::Load:
        return 0xffe15759u;
    }
    return 0xff9c9c9cu;
}

[[nodiscard]] inline nvtxDomainHandle_t domain() noexcept {
    static nvtxDomainHandle_t handle = [] {
        nvtxDomainHandle_t out = nvtxDomainCreateA("ingine");
        nvtxDomainNameCategoryA(out, static_cast<std::uint32_t>(Category::Runtime), "runtime");
        nvtxDomainNameCategoryA(out, static_cast<std::uint32_t>(Category::Prefill), "prefill");
        nvtxDomainNameCategoryA(out, static_cast<std::uint32_t>(Category::Decode), "decode");
        nvtxDomainNameCategoryA(out, static_cast<std::uint32_t>(Category::Spec), "spec");
        nvtxDomainNameCategoryA(out, static_cast<std::uint32_t>(Category::Draft), "draft");
        nvtxDomainNameCategoryA(out, static_cast<std::uint32_t>(Category::Attention), "attention");
        nvtxDomainNameCategoryA(out, static_cast<std::uint32_t>(Category::Moe), "moe");
        nvtxDomainNameCategoryA(out, static_cast<std::uint32_t>(Category::Head), "head");
        nvtxDomainNameCategoryA(out, static_cast<std::uint32_t>(Category::Control), "control");
        nvtxDomainNameCategoryA(out, static_cast<std::uint32_t>(Category::Graph), "cuda-graph");
        nvtxDomainNameCategoryA(out, static_cast<std::uint32_t>(Category::Load), "load");
        return out;
    }();
    return handle;
}

[[nodiscard]] inline nvtxStringHandle_t registered_message(Name name) noexcept {
    static constexpr auto names = std::to_array<const char*>({
        "generate",
        "request",
        "prefill",
        "prefill.chunk",
        "decode",
        "decode.step",
        "spec.round",
        "spec.draft",
        "spec.draft_commit",
        "spec.verify",
        "spec.accept",
        "spec.rollback",
        "sample",
        "engine.load",
        "weights.read",
        "weights.upload",
        "cuda_graph.capture",
        "cuda_graph.instantiate",
        "cuda_graph.update",
        "cuda_graph.upload",
        "cuda_graph.launch",
    });
    static_assert(names.size() == static_cast<std::size_t>(Name::Count));
    static const auto handles = [] {
        std::array<nvtxStringHandle_t, names.size()> out{};
        for (std::size_t i = 0; i < names.size(); ++i) {
            out[i] = nvtxDomainRegisterStringA(domain(), names[i]);
        }
        return out;
    }();
    return handles[static_cast<std::size_t>(name)];
}

[[nodiscard]] inline nvtxEventAttributes_t attributes(Name name, Category category,
                                                      std::uint64_t payload) noexcept {
    nvtxEventAttributes_t result{};
    result.version            = NVTX_VERSION;
    result.size               = NVTX_EVENT_ATTRIB_STRUCT_SIZE;
    result.category           = static_cast<std::uint32_t>(category);
    result.colorType          = NVTX_COLOR_ARGB;
    result.color              = color(category);
    result.payloadType        = NVTX_PAYLOAD_TYPE_UNSIGNED_INT64;
    result.payload.ullValue   = payload;
    result.messageType        = NVTX_MESSAGE_TYPE_REGISTERED;
    result.message.registered = registered_message(name);
    return result;
}

class ScopedRange {
public:
    explicit ScopedRange(Name name, Category category, std::uint64_t payload = 0) noexcept
        : domain_(domain()) {
        const nvtxEventAttributes_t event = attributes(name, category, payload);
        nvtxDomainRangePushEx(domain_, &event);
    }

    ScopedRange(const ScopedRange&)            = delete;
    ScopedRange& operator=(const ScopedRange&) = delete;
    ScopedRange(ScopedRange&&)                 = delete;
    ScopedRange& operator=(ScopedRange&&)      = delete;

    ~ScopedRange() noexcept { nvtxDomainRangePop(domain_); }

private:
    nvtxDomainHandle_t domain_;
};

// Process ranges can begin and end on different threads. Use this only for ownership lifetimes;
// nested execution phases belong on ScopedRange so their stack hierarchy remains explicit.
class ScopedAsyncRange {
public:
    explicit ScopedAsyncRange(Name name, Category category, std::uint64_t payload = 0) noexcept
        : domain_(domain()) {
        const nvtxEventAttributes_t event = attributes(name, category, payload);
        id_                               = nvtxDomainRangeStartEx(domain_, &event);
    }

    ScopedAsyncRange(const ScopedAsyncRange&)            = delete;
    ScopedAsyncRange& operator=(const ScopedAsyncRange&) = delete;
    ScopedAsyncRange(ScopedAsyncRange&&)                 = delete;
    ScopedAsyncRange& operator=(ScopedAsyncRange&&)      = delete;

    ~ScopedAsyncRange() noexcept { nvtxDomainRangeEnd(domain_, id_); }

private:
    nvtxDomainHandle_t domain_;
    nvtxRangeId_t id_{};
};

} // namespace ingine::nvtx
