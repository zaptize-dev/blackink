// Adapted from fngine (github.com/satellitedown/fngine, Apache-2.0) for ingine.
#include "text/jinja.h"

#include <jinja/lexer.h>
#include <jinja/parser.h>
#include <jinja/runtime.h>

#include <cstdint>
#include <limits>

namespace ingine::text {
namespace {

// Thrown by raise_exception. Deliberately not a std::exception: the interpreter decorates every
// std::exception with source positions, and this one must reach render() with the template's message intact.
struct Raised {
  std::string message;
};

jinja::value convert(const nlohmann::ordered_json& input) {
  using namespace jinja;
  if (input.is_null()) return mk_val<value_none>();
  if (input.is_boolean()) return mk_val<value_bool>(input.get<bool>());
  if (input.is_number_float()) return mk_val<value_float>(input.get<double>());
  if (input.is_number_unsigned()) {
    const auto number = input.get<std::uint64_t>();
    if (number > static_cast<std::uint64_t>(std::numeric_limits<std::int64_t>::max())) {
      return mk_val<value_float>(static_cast<double>(number));
    }
    return mk_val<value_int>(static_cast<std::int64_t>(number));
  }
  if (input.is_number_integer()) return mk_val<value_int>(input.get<std::int64_t>());
  if (input.is_string()) return mk_val<value_string>(jinja::string(input.get<std::string>()));
  if (input.is_array()) {
    auto array = mk_val<value_array>();
    for (const auto& item : input) array->push_back(convert(item));
    return array;
  }
  if (input.is_object()) {
    auto object = mk_val<value_object>();
    for (const auto& item : input.items()) {
      object->insert(mk_val<value_string>(jinja::string(item.key())), convert(item.value()));
    }
    return object;
  }
  throw std::invalid_argument("unsupported template JSON value");
}

}  // namespace

struct JinjaTemplate::Impl {
  std::string source;
  jinja::program program;
};

JinjaTemplate::JinjaTemplate(const std::string& source) {
  auto impl = std::make_shared<Impl>();
  auto tokens = jinja::lexer().tokenize(source);
  impl->source = std::move(tokens.source);
  tokens.source = impl->source;
  impl->program = jinja::parse_from_tokens(tokens);
  impl_ = std::move(impl);
}

std::string JinjaTemplate::render(const nlohmann::ordered_json& input) const {
  if (!input.is_object()) throw std::invalid_argument("template context must be an object");
  try {
    jinja::context context(impl_->source);
    for (const auto& item : input.items()) context.set_val(item.key(), convert(item.value()));
    context.set_val("raise_exception",
                    jinja::mk_val<jinja::value_func>("raise_exception", [](const jinja::func_args& args) -> jinja::value {
                      args.ensure_count(1, 1);
                      throw Raised{args.get_pos(0)->as_string().str()};
                    }));
    jinja::runtime runtime(context);
    const auto rendered = jinja::runtime::gather_string_parts(runtime.execute(impl_->program));
    std::string text;
    for (const auto& part : rendered->val_str.parts) text += part.val;
    return text;
  } catch (const Raised& raised) {
    throw TemplateError(raised.message);
  } catch (const std::exception& error) {
    throw TemplateError(error.what());
  }
}

}  // namespace ingine::text
