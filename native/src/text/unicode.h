// Adapted from fngine (github.com/satellitedown/fngine, Apache-2.0) for ingine.
// UTF-8 and Unicode-property helpers of the text stack (utf8proc-backed).
#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <string_view>
#include <vector>

namespace ingine::text::unicode {

struct CodepointSpan {
  std::int32_t value = 0;
  std::size_t offset = 0;
  std::size_t length = 0;
};

// Strict decoding: throws std::invalid_argument on invalid or truncated UTF-8.
CodepointSpan utf8_codepoint_at(std::string_view text, std::size_t offset, std::string_view context);
std::vector<CodepointSpan> utf8_codepoints(std::string_view text, std::string_view context);
std::string codepoint_to_utf8(std::int32_t codepoint);
void append_utf8(std::string& out, std::int32_t codepoint);

// Rust's String::from_utf8_lossy (what the Hugging Face byte-level decoder returns): every maximal invalid
// subpart of a sequence becomes one U+FFFD.
std::string utf8_lossy(std::string_view bytes);
// Number of codepoints of valid UTF-8 (counts lead bytes).
std::size_t utf8_length(std::string_view text) noexcept;
// Byte offset of codepoint `index` (or text.size()).
std::size_t utf8_offset(std::string_view text, std::size_t index) noexcept;

// General-category tests.
bool is_letter(std::int32_t codepoint) noexcept;   // L*
bool is_mark(std::int32_t codepoint) noexcept;     // M*
bool is_number(std::int32_t codepoint) noexcept;   // N*
bool is_decimal(std::int32_t codepoint) noexcept;  // Nd
// Oniguruma's Unicode \s (White_Space): U+0009..U+000D, U+0085, Zs, Zl, Zp.
bool is_regex_space(std::int32_t codepoint) noexcept;
// Python str.isspace() / re's \s on str: adds U+001C..U+001F to the above.
bool is_py_space(std::int32_t codepoint) noexcept;
// Python str.isprintable() for one character.
bool is_py_printable(std::int32_t codepoint) noexcept;
// Decimal digit value of an Nd codepoint (Python int()/float() accept any Nd digit), or -1.
int decimal_value(std::int32_t codepoint) noexcept;

// Python str.strip()/rstrip() with no arguments (is_py_space).
std::string_view py_strip(std::string_view text);
std::string_view py_rstrip(std::string_view text);
// Python str.lower().
std::string py_lower(std::string_view text);

}  // namespace ingine::text::unicode
