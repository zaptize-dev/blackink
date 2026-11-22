#include "value.h"

#include <nlohmann/json.hpp>

#include <charconv>
#include <cstdlib>

namespace jinja {
namespace {

// ingine: finite doubles formatted like Python's repr()/json.dumps (shortest round-trip digits,
// fixed notation for decimal exponents in [-4, 16) with at least one fractional digit, else
// d.ddde+XX), so rendered chat templates match the Python reference.
std::string python_float_repr(double x) {
    char buf[64];
    const auto res = std::to_chars(buf, buf + sizeof(buf), x, std::chars_format::scientific);
    const std::string sci(buf, res.ptr);
    const std::size_t e_pos = sci.find('e');
    const int exp10         = std::atoi(sci.c_str() + e_pos + 1);
    if (exp10 < -4 || exp10 >= 16) { return sci; }
    const bool negative = sci[0] == '-';
    std::string digits;
    for (std::size_t i = negative ? 1 : 0; i < e_pos; ++i) {
        if (sci[i] != '.') { digits += sci[i]; }
    }
    std::string out = negative ? "-" : "";
    if (exp10 >= 0) {
        const std::size_t int_len = static_cast<std::size_t>(exp10) + 1;
        if (digits.size() <= int_len) {
            out += digits + std::string(int_len - digits.size(), '0') + ".0";
        } else {
            out += digits.substr(0, int_len) + "." + digits.substr(int_len);
        }
    } else {
        out += "0." + std::string(static_cast<std::size_t>(-exp10 - 1), '0') + digits;
    }
    return out;
}

} // namespace

string value_float_t::as_string() const {
    if (std::isnan(val_flt)) return std::string("nan");
    if (std::isinf(val_flt)) return std::string(val_flt < 0 ? "-inf" : "inf");
    return python_float_repr(val_flt);
}

namespace {

void write_json(std::ostringstream& out, const value& input, int level, int indent,
                std::string_view item_separator, std::string_view key_separator, bool ensure_ascii,
                bool sort_keys, std::vector<const value_t*>& ancestors) {
    using Json = nlohmann::ordered_json;
    if (is_val<value_none>(input))
        out << "null";
    else if (is_val<value_bool>(input))
        out << (input->as_bool() ? "true" : "false");
    else if (is_val<value_int>(input))
        out << input->as_int();
    else if (is_val<value_float>(input)) {
        const auto number = input->as_float();
        if (std::isnan(number))
            out << "NaN";
        else if (std::isinf(number))
            out << (number < 0 ? "-Infinity" : "Infinity");
        else
            out << python_float_repr(number);
    } else if (is_val<value_string>(input)) {
        out << Json(input->as_string().str()).dump(-1, ' ', ensure_ascii);
    } else if (is_val<value_array>(input) || is_val<value_object>(input)) {
        if (std::find(ancestors.begin(), ancestors.end(), input.get()) != ancestors.end()) {
            throw std::runtime_error("Circular value cannot be serialized to JSON");
        }
        ancestors.push_back(input.get());
        const bool object = is_val<value_object>(input);
        auto entries      = input->val_obj;
        if (object && sort_keys) {
            std::stable_sort(entries.begin(), entries.end(), [](const auto& a, const auto& b) {
                return value_compare(a.first, b.first, lt);
            });
        }
        const auto size = object ? entries.size() : input->val_arr.size();
        out << (object ? '{' : '[');
        for (size_t i = 0; i < size; ++i) {
            if (i) out << item_separator;
            if (indent >= 0) out << '\n' << std::string((level + 1) * indent, ' ');
            if (object) {
                const auto& key = entries[i].first;
                std::string name;
                if (is_val<value_string>(key))
                    name = key->as_string().str();
                else if (is_val<value_bool>(key))
                    name = key->as_bool() ? "true" : "false";
                else if (key->is_none())
                    name = "null";
                else if (key->is_numeric())
                    name = key->as_string().str();
                else
                    throw std::runtime_error("Invalid JSON object key: " + key->type());
                out << Json(name).dump(-1, ' ', ensure_ascii) << key_separator;
            }
            write_json(out, object ? entries[i].second : input->val_arr[i], level + 1, indent,
                       item_separator, key_separator, ensure_ascii, sort_keys, ancestors);
        }
        if (size && indent >= 0) out << '\n' << std::string(level * indent, ' ');
        out << (object ? '}' : ']');
        ancestors.pop_back();
    } else {
        throw std::runtime_error("Cannot serialize " + input->type() + " to JSON");
    }
}

} // namespace

std::string value_to_json(const value& input, int indent, std::string_view item_separator,
                          std::string_view key_separator, bool ensure_ascii, bool sort_keys) {
    std::ostringstream out;
    std::vector<const value_t*> ancestors;
    write_json(out, input, 0, indent, item_separator, key_separator, ensure_ascii, sort_keys,
               ancestors);
    return out.str();
}

} // namespace jinja
