# NInfer Jinja source base

This maintained implementation originates from [llama.cpp](https://github.com/ggml-org/llama.cpp),
commit [`7609846557c50f9d984719a9e1e8c5f3d02f807b`](https://github.com/ggml-org/llama.cpp/commit/7609846557c50f9d984719a9e1e8c5f3d02f807b),
`common/jinja/`, under the MIT license; see [LICENSE](LICENSE).

NInfer maintains this fork for chat-template rendering and adopts upstream fixes selectively.

Unicode case data is generated from Unicode 17.0.0 `SpecialCasing.txt` and
`DerivedCoreProperties.txt`, under [UNICODE-LICENSE](UNICODE-LICENSE):

```bash
python3 tools/generate_jinja_unicode.py /path/to/unicode-17.0.0
```

## ingine

ingine vendors this runtime from fngine. The only change: `jinja/unicode.cpp` takes its UTF-8 helpers from the
bundled `jinja/utf8_support.{h,cpp}` (copied from fngine `src/text/unicode.cpp`) instead of the engine's
`text/unicode.h`, so the `ingine_jinja` target builds on its own.
`jinja/json.cpp` formats finite floats like Python's `repr()` (`python_float_repr`), as the reference chat
rendering does, instead of nlohmann's dump (which switches to exponent notation at 1e15).

Render speed (ingine; output unchanged, checked byte for byte on the captured agent requests):

- `runtime.h` `context`: scopes are chained `scope_frame`s that read through to the enclosing scope (or, for a
  macro/caller call, to the defining scope) instead of copying every variable on entry. Writes stay local, and a
  parent is never written while a child scope is alive, so lookups see what the copies saw.
- `for_statement`: no per-item scope without an `if` filter, loop variables bound directly (no per-item
  `std::function`), and `loop` built only when the body reads it (`scope_frame::loop_object`, materialised into
  the scope when a macro defined in the body keeps it alive); outputs collected in a plain vector.
- `if_statement` / `exec_statements`: outputs collected in a plain vector before gathering (same order).
- `member_expression`: `obj.prop` uses a key value made at parse time, one hash lookup for string keys, and no
  placeholder `undefined` allocations; `try_builtin_func` no longer copies the builtins map per call.
- `string_literal` returns a value made at parse time (string values are never modified in place).
- `value_string_t`: truthiness from the byte size (no character count), equality without concatenating parts
  (`string::equals`); `value_bool_t` / `value_float_t` hash like their integer without allocating one.
