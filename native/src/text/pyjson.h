// Python semantics the chat layer reproduces exactly: json.loads (CPython's C scanner, incl. its error
// messages), json.dumps(ensure_ascii=False, separators=(",", ":")), repr() of JSON values and float(str).
#pragma once

#include <nlohmann/json.hpp>

#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>

namespace ingine::text::py {

using Json = nlohmann::ordered_json;

// json.JSONDecodeError; what() is str(e), e.g. "Expecting value: line 1 column 1 (char 0)".
struct JsonDecodeError : std::runtime_error {
  using std::runtime_error::runtime_error;
};

// json.loads(text). Integers beyond 64 bits become doubles; lone UTF-16 surrogate escapes become U+FFFD
// (Python would keep an unencodable surrogate). Duplicate keys keep their first position and last value.
Json json_loads(std::string_view text);
// json.dumps(value, ensure_ascii=False, separators=(",", ":")).
std::string json_dumps(const Json& value);
// repr(float) / str(float).
std::string float_repr(double value);
// repr(str).
std::string str_repr(std::string_view text);
// repr() of the Python object json.loads would produce.
std::string repr(const Json& value);
// float(text) for a str argument; nullopt where Python raises ValueError.
std::optional<double> parse_float(std::string_view text);
// Python truthiness of a JSON value.
bool truthy(const Json& value);

}  // namespace ingine::text::py
