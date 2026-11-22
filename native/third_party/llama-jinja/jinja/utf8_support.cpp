// Adapted from fngine (github.com/satellitedown/fngine, Apache-2.0) for ingine.
#include "utf8_support.h"

#include <utf8proc/utf8proc.h>

#include <stdexcept>

namespace jinja::unicode::utf8_support {
namespace {

bool is_ascii_whitespace(std::int32_t codepoint) noexcept {
    return codepoint == ' ' || codepoint == '\t' || codepoint == '\n' || codepoint == '\r' ||
           codepoint == '\v' || codepoint == '\f';
}

} // namespace

CodepointSpan utf8_codepoint_at(std::string_view text, std::size_t offset,
                                std::string_view context) {
    if (offset >= text.size()) {
        throw std::out_of_range("UTF-8 codepoint offset exceeds " + std::string(context));
    }
    utf8proc_int32_t codepoint = 0;
    const utf8proc_ssize_t length =
        utf8proc_iterate(reinterpret_cast<const utf8proc_uint8_t*>(text.data() + offset),
                         static_cast<utf8proc_ssize_t>(text.size() - offset), &codepoint);
    if (length < 0) {
        throw std::invalid_argument("invalid UTF-8 in " + std::string(context) + ": " +
                                    utf8proc_errmsg(length));
    }
    if (length == 0) {
        throw std::invalid_argument("truncated UTF-8 sequence in " + std::string(context));
    }
    return CodepointSpan{codepoint, offset, static_cast<std::size_t>(length)};
}

std::string codepoint_to_utf8(std::int32_t codepoint) {
    utf8proc_uint8_t buffer[4];
    const utf8proc_ssize_t length = utf8proc_encode_char(codepoint, buffer);
    if (length <= 0) { throw std::invalid_argument("invalid Unicode codepoint"); }
    return {reinterpret_cast<const char*>(buffer), static_cast<std::size_t>(length)};
}

bool is_whitespace(std::int32_t codepoint) noexcept {
    if (is_ascii_whitespace(codepoint)) { return true; }
    switch (utf8proc_category(codepoint)) {
    case UTF8PROC_CATEGORY_ZS:
    case UTF8PROC_CATEGORY_ZL:
    case UTF8PROC_CATEGORY_ZP:
        return true;
    default:
        return false;
    }
}

} // namespace jinja::unicode::utf8_support
