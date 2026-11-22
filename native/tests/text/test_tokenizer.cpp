// Tokenizer goldens: encodes with and without added-token recognition, decodes, added tokens.
#include "text/chat.h"
#include "text/tokenizer.h"
#include "text_test.h"

#include <chrono>

using namespace ingine::text;
using Json = nlohmann::ordered_json;

namespace {

std::string ids_str(const std::vector<int32_t>& ids) {
  std::string s = "[";
  for (std::size_t i = 0; i < ids.size(); ++i) s += (i ? "," : "") + std::to_string(ids[i]);
  return s + "]";
}

}  // namespace

int main() {
  const auto dir = test::model_dir();
  if (dir.empty()) return test::kSkip;
  const Json golden = test::load_golden("tokenizer.json");
  const auto t0 = std::chrono::steady_clock::now();
  const ChatFormat fmt = ChatFormat::load(dir);
  const Tokenizer& tok = fmt.tokenizer();
  const auto t1 = std::chrono::steady_clock::now();
  std::printf("load: %.0f ms\n", std::chrono::duration<double, std::milli>(t1 - t0).count());
  test::Checker c;

  for (const Json& e : golden["encodes"]) {
    const std::string text = e["text"].get<std::string>();
    const auto ids = tok.encode(text, true);
    const auto expected = e["ids"].get<std::vector<int32_t>>();
    c.expect(ids == expected, "encode(special) " + test::show(text) + " got " + ids_str(ids) + " want " + ids_str(expected));
    const auto plain = tok.encode(text, false);
    const auto expected_plain = e["plain_ids"].get<std::vector<int32_t>>();
    c.expect(plain == expected_plain,
             "encode(plain) " + test::show(text) + " got " + ids_str(plain) + " want " + ids_str(expected_plain));
    const std::string dec = tok.decode_lossy(expected);
    c.expect(dec == e["decode"].get<std::string>(), "decode of " + test::show(text) + " got " + test::show(dec));
    c.expect(tok.decode(expected) == text, "raw decode round trip " + test::show(text));
  }
  for (const Json& d : golden["decodes"]) {
    const auto ids = d["ids"].get<std::vector<int32_t>>();
    const std::string got = fmt.decode(ids);
    c.expect(got == d["decode"].get<std::string>(),
             "decode " + ids_str(ids) + " got " + test::show(got) + " want " + test::show(d["decode"].get<std::string>()));
  }
  for (const Json& a : golden["added_tokens"]) {
    const std::string content = a["content"].get<std::string>();
    const auto id = tok.token_id(content);
    c.expect(id && *id == a["id"].get<int32_t>(), "token_id " + content);
    c.expect(tok.is_special(a["id"].get<int32_t>()) == a["special"].get<bool>(), "special flag " + content);
    c.expect(tok.token_bytes(a["id"].get<int32_t>()) == content, "token_bytes " + content);
  }
  c.expect(tok.added_tokens().size() == golden["added_tokens"].size(), "added token count");
  c.expect(fmt.stop_ids() == golden["stop_ids"].get<std::vector<int32_t>>(), "stop ids");
  c.expect(!tok.token_id("<|nope|>").has_value(), "unknown added token");
  c.expect(tok.token_bytes(-1).empty() && tok.token_bytes(1 << 30).empty(), "token_bytes out of range");

  // Throughput sanity on a long prompt (not a check).
  std::string big;
  for (const Json& e : golden["encodes"]) big += e["text"].get<std::string>();
  while (big.size() < (1u << 20)) big += big;
  const auto t2 = std::chrono::steady_clock::now();
  const auto n = tok.encode(big, true).size();
  const auto t3 = std::chrono::steady_clock::now();
  std::printf("encode %zu bytes -> %zu tokens in %.0f ms\n", big.size(), n,
              std::chrono::duration<double, std::milli>(t3 - t2).count());
  return c.finish("text_tokenizer");
}
