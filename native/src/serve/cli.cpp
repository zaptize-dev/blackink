#include "serve/cli.h"

#include <algorithm>
#include <cctype>
#include <cerrno>
#include <cstdint>
#include <optional>
#include <string_view>
#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <stdexcept>

namespace ingine::serve {
namespace {

bool parse_int(const std::string& s, int* out) {
  if (s.empty()) return false;
  errno = 0;
  char* end = nullptr;
  const long v = std::strtol(s.c_str(), &end, 10);
  if (errno != 0 || *end != '\0' || v < INT32_MIN || v > INT32_MAX) return false;
  *out = int(v);
  return true;
}

bool parse_float(const std::string& s, double* out) {
  if (s.empty()) return false;
  errno = 0;
  char* end = nullptr;
  const double v = std::strtod(s.c_str(), &end);
  if (errno == ERANGE || *end != '\0') return false;
  *out = v;
  return true;
}

std::string upper_metavar(const std::string& name) {
  std::string m = name.substr(2);
  for (char& c : m) c = c == '-' ? '_' : char(std::toupper(static_cast<unsigned char>(c)));
  return m;
}

}  // namespace

ArgParser::ArgParser(std::string prog, std::string description)
    : prog_(std::move(prog)), description_(std::move(description)) {}

void ArgParser::flag(const std::string& name, std::string help, bool* out) {
  args_.push_back(Arg{name, "", std::move(help), Kind::Flag, out, {}});
}

void ArgParser::option(const std::string& name, std::string metavar, std::string help, std::string* out,
                       std::vector<std::string> choices, bool required) {
  args_.push_back(Arg{name, metavar.empty() ? upper_metavar(name) : std::move(metavar), std::move(help), Kind::String,
                      out, std::move(choices), required});
}

void ArgParser::option(const std::string& name, std::string metavar, std::string help, int* out) {
  args_.push_back(
      Arg{name, metavar.empty() ? upper_metavar(name) : std::move(metavar), std::move(help), Kind::Int, out, {}});
}

void ArgParser::option(const std::string& name, std::string metavar, std::string help, double* out) {
  args_.push_back(
      Arg{name, metavar.empty() ? upper_metavar(name) : std::move(metavar), std::move(help), Kind::Float, out, {}});
}

void ArgParser::exclusive(std::vector<std::string> names, bool required) {
  groups_.push_back(Group{std::move(names), required});
}

const ArgParser::Arg* ArgParser::lookup(const std::string& name) const {
  for (const Arg& a : args_) {
    if (a.name == name) return &a;
  }
  return nullptr;
}

ArgParser::Arg* ArgParser::find(const std::string& name) {
  for (Arg& a : args_) {
    if (a.name == name) return &a;
  }
  std::vector<Arg*> matches;
  for (Arg& a : args_) {
    if (a.name.starts_with(name)) matches.push_back(&a);
  }
  if (matches.size() > 1) {
    std::string list;
    for (const Arg* a : matches) list += (list.empty() ? "" : ", ") + a->name;
    fail("ambiguous option: " + name + " could match " + list);
  }
  return matches.empty() ? nullptr : matches[0];
}

void ArgParser::store(Arg& arg, const std::string& value) {
  switch (arg.kind) {
    case Kind::Flag: break;
    case Kind::String:
      if (!arg.choices.empty() && std::find(arg.choices.begin(), arg.choices.end(), value) == arg.choices.end()) {
        std::string list;
        for (const std::string& c : arg.choices) list += (list.empty() ? "'" : ", '") + c + "'";
        fail("argument " + arg.name + ": invalid choice: '" + value + "' (choose from " + list + ")");
      }
      *static_cast<std::string*>(arg.out) = value;
      break;
    case Kind::Int:
      if (!parse_int(value, static_cast<int*>(arg.out))) {
        fail("argument " + arg.name + ": invalid int value: '" + value + "'");
      }
      break;
    case Kind::Float:
      if (!parse_float(value, static_cast<double*>(arg.out))) {
        fail("argument " + arg.name + ": invalid float value: '" + value + "'");
      }
      break;
  }
}

void ArgParser::parse(int argc, char** argv) {
  std::vector<std::string> unrecognized;
  for (int i = 1; i < argc; ++i) {
    std::string token = argv[i];
    if (token == "-h" || token == "--help") {
      std::cout << help();
      std::exit(0);
    }
    if (!token.starts_with("--") || token == "--") {
      unrecognized.push_back(token);
      continue;
    }
    std::optional<std::string> inline_value;
    if (const size_t eq = token.find('='); eq != std::string::npos) {
      inline_value = token.substr(eq + 1);
      token.resize(eq);
    }
    Arg* arg = find(token);
    if (!arg) {
      unrecognized.push_back(argv[i]);
      continue;
    }
    if (arg->kind == Kind::Flag) {
      if (inline_value) fail("argument " + arg->name + ": ignored explicit argument '" + *inline_value + "'");
      *static_cast<bool*>(arg->out) = true;
    } else if (inline_value) {
      store(*arg, *inline_value);
    } else {
      if (i + 1 >= argc || (std::string_view(argv[i + 1]).starts_with("-") && std::string_view(argv[i + 1]).size() > 1 &&
                            !std::isdigit(static_cast<unsigned char>(argv[i + 1][1])) && argv[i + 1][1] != '.')) {
        fail("argument " + arg->name + ": expected one argument");
      }
      store(*arg, argv[++i]);
    }
    for (const Group& g : groups_) {
      if (std::find(g.names.begin(), g.names.end(), arg->name) == g.names.end()) continue;
      for (const std::string& other : g.names) {
        const Arg* o = lookup(other);
        if (other != arg->name && o && o->given) fail("argument " + arg->name + ": not allowed with argument " + other);
      }
    }
    arg->given = true;
  }
  std::string missing;
  for (const Arg& a : args_) {
    if (a.required && !a.given) missing += (missing.empty() ? "" : ", ") + a.name;
  }
  if (!missing.empty()) fail("the following arguments are required: " + missing);
  for (const Group& g : groups_) {
    if (!g.required) continue;
    if (std::none_of(g.names.begin(), g.names.end(), [&](const std::string& n) { return given(n); })) {
      std::string list;
      for (const std::string& n : g.names) list += (list.empty() ? "" : " ") + n;
      fail("one of the arguments " + list + " is required");
    }
  }
  if (!unrecognized.empty()) {
    std::string list;
    for (const std::string& u : unrecognized) list += (list.empty() ? "" : " ") + u;
    fail("unrecognized arguments: " + list);
  }
}

bool ArgParser::given(const std::string& name) const {
  const Arg* a = lookup(name);
  return a && a->given;
}

std::string ArgParser::usage() const {
  std::string out = "usage: " + prog_ + " [-h]";
  std::vector<std::string> grouped;
  for (const Group& g : groups_) grouped.insert(grouped.end(), g.names.begin(), g.names.end());
  for (const Arg& a : args_) {
    if (std::find(grouped.begin(), grouped.end(), a.name) != grouped.end()) continue;
    const std::string body = a.kind == Kind::Flag ? a.name : a.name + " " + a.metavar;
    out += a.required ? " " + body : " [" + body + "]";
  }
  for (const Group& g : groups_) {
    std::string body;
    for (const std::string& n : g.names) {
      const Arg* a = lookup(n);
      body += (body.empty() ? "" : " | ") + (a && a->kind != Kind::Flag ? n + " " + a->metavar : n);
    }
    out += g.required ? " (" + body + ")" : " [" + body + "]";
  }
  return out + "\n";
}

std::string ArgParser::help() const {
  std::string out = usage() + "\n" + description_ + "\n\noptions:\n  -h, --help            show this help message and exit\n";
  for (const Arg& a : args_) {
    std::string left = "  " + (a.kind == Kind::Flag ? a.name : a.name + " " + a.metavar);
    if (left.size() <= 22) {
      left.resize(24, ' ');
    } else {
      left += "\n" + std::string(24, ' ');
    }
    out += left + a.help + "\n";
  }
  return out;
}

void ArgParser::fail(const std::string& message) const {
  std::cerr << usage() << prog_ << ": error: " << message << "\n";
  std::exit(2);
}

}  // namespace ingine::serve
