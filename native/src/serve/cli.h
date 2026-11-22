// A small argparse work-alike for the ingine executables: `--name value`, `--name=value`, unique-prefix
// abbreviations, store_true flags, choices, required options and mutually exclusive groups, with argparse's
// error messages (usage + "<prog>: error: ...", exit status 2) and -h/--help.
#pragma once

#include <optional>
#include <string>
#include <vector>

namespace ingine::serve {

class ArgParser {
 public:
  ArgParser(std::string prog, std::string description);

  void flag(const std::string& name, std::string help, bool* out);
  void option(const std::string& name, std::string metavar, std::string help, std::string* out,
              std::vector<std::string> choices = {}, bool required = false);
  void option(const std::string& name, std::string metavar, std::string help, int* out);
  void option(const std::string& name, std::string metavar, std::string help, double* out);
  // Exactly one (required) or at most one of these previously declared arguments may be given.
  void exclusive(std::vector<std::string> names, bool required);

  // Exits the process on --help (status 0) and on errors (status 2).
  void parse(int argc, char** argv);
  bool given(const std::string& name) const;

  std::string usage() const;
  std::string help() const;

 private:
  enum class Kind { Flag, String, Int, Float };
  struct Arg {
    std::string name, metavar, help;
    Kind kind;
    void* out;
    std::vector<std::string> choices;
    bool required = false;
    bool given = false;
  };
  struct Group {
    std::vector<std::string> names;
    bool required;
  };

  [[noreturn]] void fail(const std::string& message) const;
  Arg* find(const std::string& name);
  const Arg* lookup(const std::string& name) const;
  void store(Arg& arg, const std::string& value);

  std::string prog_, description_;
  std::vector<Arg> args_;
  std::vector<Group> groups_;
};

}  // namespace ingine::serve
