// Adapted from fngine (github.com/satellitedown/fngine, Apache-2.0) for ingine.
#include "text/unicode.h"

#include <utf8proc/utf8proc.h>

#include <stdexcept>

namespace ingine::text::unicode {

CodepointSpan utf8_codepoint_at(std::string_view text, std::size_t offset, std::string_view context) {
  if (offset >= text.size()) {
    throw std::out_of_range("UTF-8 codepoint offset exceeds " + std::string(context));
  }
  utf8proc_int32_t codepoint = 0;
  const utf8proc_ssize_t length =
      utf8proc_iterate(reinterpret_cast<const utf8proc_uint8_t*>(text.data() + offset),
                       static_cast<utf8proc_ssize_t>(text.size() - offset), &codepoint);
  if (length < 0) {
    throw std::invalid_argument("invalid UTF-8 in " + std::string(context) + ": " + utf8proc_errmsg(length));
  }
  if (length == 0) throw std::invalid_argument("truncated UTF-8 sequence in " + std::string(context));
  return CodepointSpan{codepoint, offset, static_cast<std::size_t>(length)};
}

std::vector<CodepointSpan> utf8_codepoints(std::string_view text, std::string_view context) {
  std::vector<CodepointSpan> codepoints;
  codepoints.reserve(text.size());
  for (std::size_t offset = 0; offset < text.size();) {
    const CodepointSpan codepoint = utf8_codepoint_at(text, offset, context);
    codepoints.push_back(codepoint);
    offset += codepoint.length;
  }
  return codepoints;
}

void append_utf8(std::string& out, std::int32_t cp) {
  const auto u = static_cast<std::uint32_t>(cp);
  if (u < 0x80) {
    out.push_back(static_cast<char>(u));
  } else if (u < 0x800) {
    out.push_back(static_cast<char>(0xC0 | (u >> 6)));
    out.push_back(static_cast<char>(0x80 | (u & 0x3F)));
  } else if (u < 0x10000) {
    out.push_back(static_cast<char>(0xE0 | (u >> 12)));
    out.push_back(static_cast<char>(0x80 | ((u >> 6) & 0x3F)));
    out.push_back(static_cast<char>(0x80 | (u & 0x3F)));
  } else if (u < 0x110000) {
    out.push_back(static_cast<char>(0xF0 | (u >> 18)));
    out.push_back(static_cast<char>(0x80 | ((u >> 12) & 0x3F)));
    out.push_back(static_cast<char>(0x80 | ((u >> 6) & 0x3F)));
    out.push_back(static_cast<char>(0x80 | (u & 0x3F)));
  } else {
    throw std::invalid_argument("invalid Unicode codepoint");
  }
}

std::string codepoint_to_utf8(std::int32_t codepoint) {
  std::string out;
  append_utf8(out, codepoint);
  return out;
}

std::string utf8_lossy(std::string_view bytes) {
  static constexpr std::string_view kReplacement = "\xEF\xBF\xBD";
  std::string out;
  out.reserve(bytes.size());
  const auto* s = reinterpret_cast<const unsigned char*>(bytes.data());
  const std::size_t n = bytes.size();
  std::size_t i = 0;
  while (i < n) {
    const unsigned char b = s[i];
    if (b < 0x80) {
      out.push_back(static_cast<char>(b));
      ++i;
      continue;
    }
    // Width and the valid range of the first continuation byte (Unicode Table 3-7).
    int width = 0;
    unsigned char lo = 0x80, hi = 0xBF;
    if (b >= 0xC2 && b <= 0xDF) {
      width = 2;
    } else if (b == 0xE0) {
      width = 3, lo = 0xA0;
    } else if ((b >= 0xE1 && b <= 0xEC) || b == 0xEE || b == 0xEF) {
      width = 3;
    } else if (b == 0xED) {
      width = 3, hi = 0x9F;
    } else if (b == 0xF0) {
      width = 4, lo = 0x90;
    } else if (b >= 0xF1 && b <= 0xF3) {
      width = 4;
    } else if (b == 0xF4) {
      width = 4, hi = 0x8F;
    }
    if (width == 0) {
      out += kReplacement;
      ++i;
      continue;
    }
    std::size_t j = i + 1;
    bool ok = true;
    for (int k = 1; k < width; ++k, ++j) {
      const unsigned char lower = k == 1 ? lo : 0x80, upper = k == 1 ? hi : 0xBF;
      if (j >= n || s[j] < lower || s[j] > upper) {
        ok = false;
        break;
      }
    }
    if (ok) {
      out.append(bytes.data() + i, static_cast<std::size_t>(width));
    } else {
      out += kReplacement;
    }
    i = j;
  }
  return out;
}

std::size_t utf8_length(std::string_view text) noexcept {
  std::size_t count = 0;
  for (const char c : text) count += (static_cast<unsigned char>(c) & 0xC0) != 0x80;
  return count;
}

std::size_t utf8_offset(std::string_view text, std::size_t index) noexcept {
  std::size_t seen = 0;
  for (std::size_t i = 0; i < text.size(); ++i) {
    if ((static_cast<unsigned char>(text[i]) & 0xC0) != 0x80) {
      if (seen == index) return i;
      ++seen;
    }
  }
  return text.size();
}

bool is_letter(std::int32_t codepoint) noexcept {
  switch (utf8proc_category(codepoint)) {
    case UTF8PROC_CATEGORY_LU:
    case UTF8PROC_CATEGORY_LL:
    case UTF8PROC_CATEGORY_LT:
    case UTF8PROC_CATEGORY_LM:
    case UTF8PROC_CATEGORY_LO:
      return true;
    default:
      return false;
  }
}

bool is_mark(std::int32_t codepoint) noexcept {
  switch (utf8proc_category(codepoint)) {
    case UTF8PROC_CATEGORY_MN:
    case UTF8PROC_CATEGORY_MC:
    case UTF8PROC_CATEGORY_ME:
      return true;
    default:
      return false;
  }
}

bool is_number(std::int32_t codepoint) noexcept {
  switch (utf8proc_category(codepoint)) {
    case UTF8PROC_CATEGORY_ND:
    case UTF8PROC_CATEGORY_NL:
    case UTF8PROC_CATEGORY_NO:
      return true;
    default:
      return false;
  }
}

bool is_decimal(std::int32_t codepoint) noexcept {
  return utf8proc_category(codepoint) == UTF8PROC_CATEGORY_ND;
}

bool is_regex_space(std::int32_t codepoint) noexcept {
  if ((codepoint >= 0x09 && codepoint <= 0x0D) || codepoint == 0x20 || codepoint == 0x85) return true;
  if (codepoint < 0x80) return false;
  switch (utf8proc_category(codepoint)) {
    case UTF8PROC_CATEGORY_ZS:
    case UTF8PROC_CATEGORY_ZL:
    case UTF8PROC_CATEGORY_ZP:
      return true;
    default:
      return false;
  }
}

bool is_py_space(std::int32_t codepoint) noexcept {
  return (codepoint >= 0x1C && codepoint <= 0x1F) || is_regex_space(codepoint);
}

bool is_py_printable(std::int32_t codepoint) noexcept {
  if (codepoint == ' ') return true;
  switch (utf8proc_category(codepoint)) {
    case UTF8PROC_CATEGORY_CC:
    case UTF8PROC_CATEGORY_CF:
    case UTF8PROC_CATEGORY_CS:
    case UTF8PROC_CATEGORY_CO:
    case UTF8PROC_CATEGORY_CN:
    case UTF8PROC_CATEGORY_ZL:
    case UTF8PROC_CATEGORY_ZP:
    case UTF8PROC_CATEGORY_ZS:
      return false;
    default:
      return true;
  }
}

int decimal_value(std::int32_t codepoint) noexcept {
  if (codepoint >= '0' && codepoint <= '9') return codepoint - '0';
  if (!is_decimal(codepoint)) return -1;
  // Nd characters come in runs of complete 0..9 sequences.
  std::int32_t start = codepoint;
  while (start > 0 && is_decimal(start - 1)) --start;
  return (codepoint - start) % 10;
}

std::string_view py_strip(std::string_view text) {
  text = py_rstrip(text);
  std::size_t begin = 0;
  while (begin < text.size()) {
    const CodepointSpan cp = utf8_codepoint_at(text, begin, "string");
    if (!is_py_space(cp.value)) break;
    begin += cp.length;
  }
  return text.substr(begin);
}

std::string_view py_rstrip(std::string_view text) {
  std::size_t end = text.size();
  while (end > 0) {
    std::size_t begin = end - 1;
    while (begin > 0 && (static_cast<unsigned char>(text[begin]) & 0xC0) == 0x80) --begin;
    const CodepointSpan cp = utf8_codepoint_at(text, begin, "string");
    if (!is_py_space(cp.value)) break;
    end = begin;
  }
  return text.substr(0, end);
}

std::string py_lower(std::string_view text) {
  std::string out;
  out.reserve(text.size());
  for (std::size_t offset = 0; offset < text.size();) {
    const CodepointSpan cp = utf8_codepoint_at(text, offset, "string");
    offset += cp.length;
    if (cp.value == 0x130) {  // SpecialCasing: LATIN CAPITAL LETTER I WITH DOT ABOVE -> i + U+0307
      out += "i\xCC\x87";
      continue;
    }
    append_utf8(out, utf8proc_tolower(cp.value));
  }
  return out;
}

}  // namespace ingine::text::unicode
