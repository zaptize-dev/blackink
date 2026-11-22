#include "text/pyjson.h"

#include "text/unicode.h"

#include <cctype>
#include <charconv>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <limits>
#include <vector>

namespace ingine::text::py {
namespace {

namespace uni = unicode;

// Port of CPython 3.12 Modules/_json.c (scanstring_unicode, _parse_object_unicode, _parse_array_unicode,
// scan_once_unicode, _match_number_unicode) and json.decoder.JSONDecoder.decode, over codepoint indices so
// error positions are Python character offsets.
class Scanner {
 public:
  explicit Scanner(std::vector<char32_t> s) : s_(std::move(s)), len_(static_cast<std::ptrdiff_t>(s_.size())) {}

  Json decode() {
    if (len_ > 0 && s_[0] == 0xFEFF) fail("Unexpected UTF-8 BOM (decode using utf-8-sig)", 0);
    std::ptrdiff_t idx = skip_ws(0);
    std::ptrdiff_t end = 0;
    Json obj = scan_once(idx, end, 0);
    end = skip_ws(end);
    if (end != len_) fail("Extra data", end);
    return obj;
  }

 private:
  struct StopIteration {
    std::ptrdiff_t idx;
  };

  [[noreturn]] void fail(const char* msg, std::ptrdiff_t pos) const {
    std::ptrdiff_t line = 1, last_nl = -1;
    for (std::ptrdiff_t i = 0; i < pos && i < len_; ++i) {
      if (s_[static_cast<std::size_t>(i)] == '\n') ++line, last_nl = i;
    }
    const std::ptrdiff_t col = pos - last_nl;
    throw JsonDecodeError(std::string(msg) + ": line " + std::to_string(line) + " column " + std::to_string(col) +
                          " (char " + std::to_string(pos) + ")");
  }

  char32_t at(std::ptrdiff_t i) const { return s_[static_cast<std::size_t>(i)]; }
  static bool is_ws(char32_t c) { return c == ' ' || c == '\t' || c == '\n' || c == '\r'; }
  std::ptrdiff_t skip_ws(std::ptrdiff_t i) const {
    while (i < len_ && is_ws(at(i))) ++i;
    return i;
  }

  static int hex(char32_t c) {
    if (c >= '0' && c <= '9') return static_cast<int>(c - '0');
    if (c >= 'a' && c <= 'f') return static_cast<int>(c - 'a' + 10);
    if (c >= 'A' && c <= 'F') return static_cast<int>(c - 'A' + 10);
    return -1;
  }

  static void append(std::string& out, char32_t c) {
    if (c >= 0xD800 && c <= 0xDFFF) c = 0xFFFD;
    uni::append_utf8(out, static_cast<std::int32_t>(c));
  }

  // end: index just after the opening quote; begin = end - 1.
  std::string scanstring(std::ptrdiff_t end, std::ptrdiff_t& next_end) const {
    const std::ptrdiff_t begin = end - 1;
    std::string out;
    std::ptrdiff_t next = 0;
    while (true) {
      char32_t c = 0;
      {
        char32_t d = 0;
        for (next = end; next < len_; ++next) {
          d = at(next);
          if (d == '"' || d == '\\') break;
          if (d <= 0x1f) fail("Invalid control character at", next);
        }
        c = d;
      }
      if (c != '"' && c != '\\') fail("Unterminated string starting at", begin);
      for (std::ptrdiff_t i = end; i < next; ++i) append(out, at(i));
      ++next;
      if (c == '"') {
        end = next;
        break;
      }
      if (next == len_) fail("Unterminated string starting at", begin);
      c = at(next);
      if (c != 'u') {
        end = next + 1;
        switch (c) {
          case '"': break;
          case '\\': break;
          case '/': break;
          case 'b': c = '\b'; break;
          case 'f': c = '\f'; break;
          case 'n': c = '\n'; break;
          case 'r': c = '\r'; break;
          case 't': c = '\t'; break;
          default: c = 0;
        }
        if (c == 0) fail("Invalid \\escape", end - 2);
      } else {
        c = 0;
        ++next;
        end = next + 4;
        if (end >= len_) fail("Invalid \\uXXXX escape", next - 1);
        for (; next < end; ++next) {
          const int h = hex(at(next));
          if (h < 0) fail("Invalid \\uXXXX escape", end - 5);
          c = (c << 4) | static_cast<char32_t>(h);
        }
        if (c >= 0xD800 && c <= 0xDBFF && end + 6 < len_ && at(next++) == '\\' && at(next++) == 'u') {
          char32_t c2 = 0;
          end += 6;
          for (; next < end; ++next) {
            const int h = hex(at(next));
            if (h < 0) fail("Invalid \\uXXXX escape", end - 5);
            c2 = (c2 << 4) | static_cast<char32_t>(h);
          }
          if (c2 >= 0xDC00 && c2 <= 0xDFFF) {
            c = 0x10000 + (((c - 0xD800) << 10) | (c2 - 0xDC00));
          } else {
            end -= 6;
          }
        }
      }
      append(out, c);
    }
    next_end = end;
    return out;
  }

  Json parse_object(std::ptrdiff_t idx, std::ptrdiff_t& next_idx, int depth) {
    Json obj = Json::object();
    const std::ptrdiff_t end_idx = len_ - 1;
    while (idx <= end_idx && is_ws(at(idx))) ++idx;
    if (idx > end_idx || at(idx) != '}') {
      while (true) {
        if (idx > end_idx || at(idx) != '"') fail("Expecting property name enclosed in double quotes", idx);
        std::ptrdiff_t n = 0;
        std::string key = scanstring(idx + 1, n);
        idx = n;
        while (idx <= end_idx && is_ws(at(idx))) ++idx;
        if (idx > end_idx || at(idx) != ':') fail("Expecting ':' delimiter", idx);
        ++idx;
        while (idx <= end_idx && is_ws(at(idx))) ++idx;
        Json val = scan_once(idx, n, depth + 1);
        obj[key] = std::move(val);  // ordered_map: an existing key keeps its position, takes the new value
        idx = n;
        while (idx <= end_idx && is_ws(at(idx))) ++idx;
        if (idx <= end_idx && at(idx) == '}') break;
        if (idx > end_idx || at(idx) != ',') fail("Expecting ',' delimiter", idx);
        ++idx;
        while (idx <= end_idx && is_ws(at(idx))) ++idx;
      }
    }
    next_idx = idx + 1;
    return obj;
  }

  Json parse_array(std::ptrdiff_t idx, std::ptrdiff_t& next_idx, int depth) {
    Json arr = Json::array();
    const std::ptrdiff_t end_idx = len_ - 1;
    while (idx <= end_idx && is_ws(at(idx))) ++idx;
    if (idx > end_idx || at(idx) != ']') {
      while (true) {
        std::ptrdiff_t n = 0;
        arr.push_back(scan_once(idx, n, depth + 1));
        idx = n;
        while (idx <= end_idx && is_ws(at(idx))) ++idx;
        if (idx <= end_idx && at(idx) == ']') break;
        if (idx > end_idx || at(idx) != ',') fail("Expecting ',' delimiter", idx);
        ++idx;
        while (idx <= end_idx && is_ws(at(idx))) ++idx;
      }
    }
    next_idx = idx + 1;
    return arr;
  }

  bool matches(std::ptrdiff_t idx, std::string_view word) const {
    for (std::size_t i = 0; i < word.size(); ++i) {
      if (at(idx + static_cast<std::ptrdiff_t>(i)) != static_cast<char32_t>(word[i])) return false;
    }
    return true;
  }

  Json scan_once(std::ptrdiff_t idx, std::ptrdiff_t& next_idx, int depth) {
    if (depth > 900) throw std::runtime_error("maximum recursion depth exceeded while decoding a JSON document");
    if (idx >= len_) expecting_value(idx);
    switch (at(idx)) {
      case '"': return Json(scanstring(idx + 1, next_idx));
      case '{': return parse_object(idx + 1, next_idx, depth);
      case '[': return parse_array(idx + 1, next_idx, depth);
      case 'n':
        if (idx + 3 < len_ && matches(idx + 1, "ull")) return next_idx = idx + 4, Json(nullptr);
        break;
      case 't':
        if (idx + 3 < len_ && matches(idx + 1, "rue")) return next_idx = idx + 4, Json(true);
        break;
      case 'f':
        if (idx + 4 < len_ && matches(idx + 1, "alse")) return next_idx = idx + 5, Json(false);
        break;
      case 'N':
        if (idx + 2 < len_ && matches(idx + 1, "aN")) {
          return next_idx = idx + 3, Json(std::numeric_limits<double>::quiet_NaN());
        }
        break;
      case 'I':
        if (idx + 7 < len_ && matches(idx + 1, "nfinity")) {
          return next_idx = idx + 8, Json(std::numeric_limits<double>::infinity());
        }
        break;
      case '-':
        if (idx + 8 < len_ && matches(idx + 1, "Infinity")) {
          return next_idx = idx + 9, Json(-std::numeric_limits<double>::infinity());
        }
        break;
      default: break;
    }
    return match_number(idx, next_idx);
  }

  [[noreturn]] void expecting_value(std::ptrdiff_t idx) const { fail("Expecting value", idx); }

  static bool digit(char32_t c) { return c >= '0' && c <= '9'; }

  Json match_number(std::ptrdiff_t start, std::ptrdiff_t& next_idx) const {
    const std::ptrdiff_t end_idx = len_ - 1;
    std::ptrdiff_t idx = start;
    bool is_float = false;
    if (at(idx) == '-') {
      ++idx;
      if (idx > end_idx) expecting_value(start);
    }
    if (at(idx) >= '1' && at(idx) <= '9') {
      ++idx;
      while (idx <= end_idx && digit(at(idx))) ++idx;
    } else if (at(idx) == '0') {
      ++idx;
    } else {
      expecting_value(start);
    }
    if (idx < end_idx && at(idx) == '.' && digit(at(idx + 1))) {
      is_float = true;
      idx += 2;
      while (idx <= end_idx && digit(at(idx))) ++idx;
    }
    if (idx < end_idx && (at(idx) == 'e' || at(idx) == 'E')) {
      const std::ptrdiff_t e_start = idx;
      ++idx;
      if (idx < end_idx && (at(idx) == '-' || at(idx) == '+')) ++idx;
      while (idx <= end_idx && digit(at(idx))) ++idx;
      if (digit(at(idx - 1))) {
        is_float = true;
      } else {
        idx = e_start;
      }
    }
    std::string num;
    for (std::ptrdiff_t i = start; i < idx; ++i) num.push_back(static_cast<char>(at(i)));
    next_idx = idx;
    if (!is_float) {
      std::int64_t v = 0;
      if (auto [p, ec] = std::from_chars(num.data(), num.data() + num.size(), v); ec == std::errc{}) return Json(v);
      std::uint64_t u = 0;
      if (auto [p, ec] = std::from_chars(num.data(), num.data() + num.size(), u); ec == std::errc{}) return Json(u);
    }
    return Json(std::strtod(num.c_str(), nullptr));
  }

  std::vector<char32_t> s_;
  std::ptrdiff_t len_;
};

void dump_string(std::string& out, std::string_view s) {
  static constexpr char kHex[] = "0123456789abcdef";
  out.push_back('"');
  for (const char ch : s) {
    const auto c = static_cast<unsigned char>(ch);
    switch (c) {
      case '"': out += "\\\""; break;
      case '\\': out += "\\\\"; break;
      case '\n': out += "\\n"; break;
      case '\r': out += "\\r"; break;
      case '\t': out += "\\t"; break;
      case '\b': out += "\\b"; break;
      case '\f': out += "\\f"; break;
      default:
        if (c < 0x20) {
          out += "\\u00";
          out.push_back(kHex[c >> 4]);
          out.push_back(kHex[c & 15]);
        } else {
          out.push_back(static_cast<char>(c));
        }
    }
  }
  out.push_back('"');
}

void dump(std::string& out, const Json& v) {
  switch (v.type()) {
    case Json::value_t::null: out += "null"; break;
    case Json::value_t::boolean: out += v.get<bool>() ? "true" : "false"; break;
    case Json::value_t::number_integer: out += std::to_string(v.get<std::int64_t>()); break;
    case Json::value_t::number_unsigned: out += std::to_string(v.get<std::uint64_t>()); break;
    case Json::value_t::number_float: {
      const double d = v.get<double>();
      if (std::isnan(d)) {
        out += "NaN";
      } else if (std::isinf(d)) {
        out += d < 0 ? "-Infinity" : "Infinity";
      } else {
        out += float_repr(d);
      }
      break;
    }
    case Json::value_t::string: dump_string(out, v.get_ref<const std::string&>()); break;
    case Json::value_t::array: {
      out.push_back('[');
      bool first = true;
      for (const auto& item : v) {
        if (!first) out.push_back(',');
        first = false;
        dump(out, item);
      }
      out.push_back(']');
      break;
    }
    case Json::value_t::object: {
      out.push_back('{');
      bool first = true;
      for (const auto& [key, item] : v.items()) {
        if (!first) out.push_back(',');
        first = false;
        dump_string(out, key);
        out.push_back(':');
        dump(out, item);
      }
      out.push_back('}');
      break;
    }
    default: throw std::invalid_argument("json_dumps: unsupported value");
  }
}

void repr_to(std::string& out, const Json& v) {
  switch (v.type()) {
    case Json::value_t::null: out += "None"; break;
    case Json::value_t::boolean: out += v.get<bool>() ? "True" : "False"; break;
    case Json::value_t::number_integer: out += std::to_string(v.get<std::int64_t>()); break;
    case Json::value_t::number_unsigned: out += std::to_string(v.get<std::uint64_t>()); break;
    case Json::value_t::number_float: out += float_repr(v.get<double>()); break;
    case Json::value_t::string: out += str_repr(v.get_ref<const std::string&>()); break;
    case Json::value_t::array: {
      out.push_back('[');
      bool first = true;
      for (const auto& item : v) {
        if (!first) out += ", ";
        first = false;
        repr_to(out, item);
      }
      out.push_back(']');
      break;
    }
    case Json::value_t::object: {
      out.push_back('{');
      bool first = true;
      for (const auto& [key, item] : v.items()) {
        if (!first) out += ", ";
        first = false;
        out += str_repr(key);
        out += ": ";
        repr_to(out, item);
      }
      out.push_back('}');
      break;
    }
    default: out += "None";
  }
}

}  // namespace

Json json_loads(std::string_view text) {
  std::vector<char32_t> cps;
  cps.reserve(text.size());
  for (const auto& cp : uni::utf8_codepoints(text, "JSON text")) cps.push_back(static_cast<char32_t>(cp.value));
  return Scanner(std::move(cps)).decode();
}

std::string json_dumps(const Json& value) {
  std::string out;
  dump(out, value);
  return out;
}

std::string float_repr(double value) {
  if (std::isnan(value)) return "nan";
  if (std::isinf(value)) return value < 0 ? "-inf" : "inf";
  if (value == 0.0) return std::signbit(value) ? "-0.0" : "0.0";
  char buf[64];
  const auto res = std::to_chars(buf, buf + sizeof(buf), value, std::chars_format::scientific);
  const std::string_view sci(buf, static_cast<std::size_t>(res.ptr - buf));  // [-]d[.ddd]e±XX
  std::string out;
  std::size_t i = 0;
  if (sci[0] == '-') out.push_back('-'), i = 1;
  const std::size_t e = sci.find('e');
  std::string digits;
  for (std::size_t k = i; k < e; ++k) {
    if (sci[k] != '.') digits.push_back(sci[k]);
  }
  int exp10 = 0;
  std::from_chars(sci.data() + e + (sci[e + 1] == '+' ? 2 : 1), sci.data() + sci.size(), exp10);
  if (exp10 >= -4 && exp10 < 16) {
    if (exp10 >= 0) {
      const auto int_len = static_cast<std::size_t>(exp10) + 1;
      if (digits.size() <= int_len) {
        out += digits;
        out.append(int_len - digits.size(), '0');
        out += ".0";
      } else {
        out += digits.substr(0, int_len);
        out.push_back('.');
        out += digits.substr(int_len);
      }
    } else {
      out += "0.";
      out.append(static_cast<std::size_t>(-exp10 - 1), '0');
      out += digits;
    }
    return out;
  }
  out.push_back(digits[0]);
  if (digits.size() > 1) {
    out.push_back('.');
    out += digits.substr(1);
  }
  out.push_back('e');
  out.push_back(exp10 < 0 ? '-' : '+');
  const int a = exp10 < 0 ? -exp10 : exp10;
  if (a < 10) out.push_back('0');
  out += std::to_string(a);
  return out;
}

std::string str_repr(std::string_view text) {
  static constexpr char kHex[] = "0123456789abcdef";
  const bool has_single = text.find('\'') != std::string_view::npos;
  const bool has_double = text.find('"') != std::string_view::npos;
  const char quote = has_single && !has_double ? '"' : '\'';
  std::string out(1, quote);
  for (std::size_t off = 0; off < text.size();) {
    const auto cp = uni::utf8_codepoint_at(text, off, "string");
    const std::int32_t c = cp.value;
    if (c == quote || c == '\\') {
      out.push_back('\\');
      out.push_back(static_cast<char>(c));
    } else if (c == '\t') {
      out += "\\t";
    } else if (c == '\n') {
      out += "\\n";
    } else if (c == '\r') {
      out += "\\r";
    } else if (c < 0x20 || c == 0x7F) {
      out += "\\x";
      out.push_back(kHex[c >> 4]);
      out.push_back(kHex[c & 15]);
    } else if (c < 0x7F) {
      out.push_back(static_cast<char>(c));
    } else if (uni::is_py_printable(c)) {
      out.append(text.substr(off, cp.length));
    } else {
      int width = 0;
      if (c <= 0xFF) {
        out += "\\x", width = 2;
      } else if (c <= 0xFFFF) {
        out += "\\u", width = 4;
      } else {
        out += "\\U", width = 8;
      }
      for (int k = width - 1; k >= 0; --k) out.push_back(kHex[(c >> (4 * k)) & 15]);
    }
    off += cp.length;
  }
  out.push_back(quote);
  return out;
}

std::string repr(const Json& value) {
  std::string out;
  repr_to(out, value);
  return out;
}

std::optional<double> parse_float(std::string_view text) {
  // _PyUnicode_TransformDecimalAndSpaceToASCII: Nd digits -> ASCII, white space -> ' '; other non-ASCII fails.
  std::string ascii;
  for (const auto& cp : uni::utf8_codepoints(text, "float")) {
    if (cp.value < 0x80) {
      ascii.push_back(static_cast<char>(cp.value));
    } else if (uni::is_py_space(cp.value)) {
      ascii.push_back(' ');
    } else if (const int d = uni::decimal_value(cp.value); d >= 0) {
      ascii.push_back(static_cast<char>('0' + d));
    } else {
      return std::nullopt;
    }
  }
  const auto is_ws = [](char c) { return c == ' ' || (c >= '\t' && c <= '\r') || (c >= 0x1C && c <= 0x1F); };
  std::size_t b = 0, e = ascii.size();
  while (b < e && is_ws(ascii[b])) ++b;
  while (e > b && is_ws(ascii[e - 1])) --e;
  std::string s = ascii.substr(b, e - b);
  // Underscores only between digits (PEP 515).
  std::string t;
  const auto is_digit = [](char c) { return c >= '0' && c <= '9'; };
  for (std::size_t i = 0; i < s.size(); ++i) {
    if (s[i] == '_') {
      if (i == 0 || i + 1 >= s.size() || !is_digit(s[i - 1]) || !is_digit(s[i + 1])) return std::nullopt;
      continue;
    }
    t.push_back(s[i]);
  }
  std::size_t i = 0;
  bool negative = false;
  if (i < t.size() && (t[i] == '+' || t[i] == '-')) negative = t[i++] == '-';
  std::string rest = t.substr(i);
  for (char& c : rest) c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
  if (rest == "inf" || rest == "infinity") {
    return negative ? -std::numeric_limits<double>::infinity() : std::numeric_limits<double>::infinity();
  }
  if (rest == "nan") return negative ? -std::numeric_limits<double>::quiet_NaN() : std::numeric_limits<double>::quiet_NaN();
  std::size_t k = 0, mant = 0;
  while (k < rest.size() && is_digit(rest[k])) ++k, ++mant;
  if (k < rest.size() && rest[k] == '.') {
    ++k;
    while (k < rest.size() && is_digit(rest[k])) ++k, ++mant;
  }
  if (mant == 0) return std::nullopt;
  if (k < rest.size() && rest[k] == 'e') {
    ++k;
    if (k < rest.size() && (rest[k] == '+' || rest[k] == '-')) ++k;
    std::size_t exp_digits = 0;
    while (k < rest.size() && is_digit(rest[k])) ++k, ++exp_digits;
    if (exp_digits == 0) return std::nullopt;
  }
  if (k != rest.size()) return std::nullopt;
  return std::strtod(t.c_str(), nullptr);
}

bool truthy(const Json& value) {
  switch (value.type()) {
    case Json::value_t::null: return false;
    case Json::value_t::boolean: return value.get<bool>();
    case Json::value_t::number_integer: return value.get<std::int64_t>() != 0;
    case Json::value_t::number_unsigned: return value.get<std::uint64_t>() != 0;
    case Json::value_t::number_float: return value.get<double>() != 0.0;
    case Json::value_t::string: return !value.get_ref<const std::string&>().empty();
    case Json::value_t::array:
    case Json::value_t::object: return !value.empty();
    default: return true;
  }
}

}  // namespace ingine::text::py
