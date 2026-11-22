// Python-compatibility goldens: json.loads (values and error messages), json.dumps, repr(), float(str).
#include "text/pyjson.h"
#include "text_test.h"

using namespace ingine::text;
using Json = nlohmann::ordered_json;

int main() {
  const Json golden = test::load_golden("pyjson.json");
  test::Checker c;
  for (const Json& e : golden["loads"]) {
    const std::string doc = e["doc"].get<std::string>();
    try {
      const Json v = py::json_loads(doc);
      if (!c.expect(e.contains("dumps"), "loads " + test::show(doc) + ": parsed, expected " + e.value("error", ""))) continue;
      c.expect(py::json_dumps(v) == e["dumps"].get<std::string>(),
               "dumps " + test::show(doc) + " got " + py::json_dumps(v) + " want " + e["dumps"].get<std::string>());
      c.expect(py::repr(v) == e["repr"].get<std::string>(),
               "repr " + test::show(doc) + " got " + py::repr(v) + " want " + e["repr"].get<std::string>());
    } catch (const py::JsonDecodeError& err) {
      if (!c.expect(e.contains("error"), "loads " + test::show(doc) + ": error " + err.what())) continue;
      c.expect(err.what() == e["error"].get<std::string>(),
               "loads " + test::show(doc) + " error got " + err.what() + " want " + e["error"].get<std::string>());
    }
  }
  for (const Json& e : golden["float_repr"]) {
    const double v = e["value"].get<double>();
    c.expect(py::float_repr(v) == e["repr"].get<std::string>(),
             "float_repr got " + py::float_repr(v) + " want " + e["repr"].get<std::string>());
  }
  for (const Json& e : golden["str_repr"]) {
    const std::string s = e["value"].get<std::string>();
    c.expect(py::str_repr(s) == e["repr"].get<std::string>(),
             "str_repr " + test::show(s) + " got " + py::str_repr(s) + " want " + e["repr"].get<std::string>());
  }
  for (const Json& e : golden["parse_float"]) {
    const std::string s = e["text"].get<std::string>();
    const auto v = py::parse_float(s);
    const std::string got = v ? py::float_repr(*v) : "None";
    const std::string want = e["value"].is_null() ? "None" : e["value"].get<std::string>();
    c.expect(got == want, "float(" + test::show(s) + ") got " + got + " want " + want);
  }
  return c.finish("text_pyjson");
}
