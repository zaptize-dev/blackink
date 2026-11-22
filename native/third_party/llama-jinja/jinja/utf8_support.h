// Adapted from fngine (github.com/satellitedown/fngine, Apache-2.0) for ingine.
// UTF-8 helpers the Jinja runtime needs (from fngine src/text/unicode.{h,cpp}), kept here so the
// vendored runtime builds without the text component.
#pragma once

#include <cstdint>
#include <string>
#include <string_view>

namespace jinja::unicode::utf8_support {

struct CodepointSpan {
    std::int32_t value = 0;
    std::size_t offset = 0;
    std::size_t length = 0;
};

CodepointSpan utf8_codepoint_at(std::string_view text, std::size_t offset,
                                std::string_view context);
std::string codepoint_to_utf8(std::int32_t codepoint);
bool is_whitespace(std::int32_t codepoint) noexcept;

} // namespace jinja::unicode::utf8_support
