// Adapted from fngine (github.com/satellitedown/fngine, Apache-2.0) for ingine.
#include "text/tokenizer.h"

#include "text/unicode.h"

#include <nlohmann/json.hpp>
#include <utf8proc/utf8proc.h>

#include <algorithm>
#include <fstream>
#include <limits>
#include <queue>
#include <sstream>
#include <stdexcept>

namespace ingine::text {
namespace {

using Json = nlohmann::json;
namespace uni = unicode;

constexpr std::string_view kSplitPattern =
    "[^\\r\\n\\p{L}\\p{N}]?[\\p{Lu}\\p{Lt}\\p{Lm}\\p{Lo}\\p{M}]*[\\p{Ll}\\p{Lm}\\p{Lo}\\p{M}]+(?i:'s|'t|'re|'ve|'m|'ll|'d)?"
    "|[^\\r\\n\\p{L}\\p{N}]?[\\p{Lu}\\p{Lt}\\p{Lm}\\p{Lo}\\p{M}]+[\\p{Ll}\\p{Lm}\\p{Lo}\\p{M}]*(?i:'s|'t|'re|'ve|'m|'ll|'d)?"
    "|\\p{N}{1,3}| ?[^\\s\\p{L}\\p{N}]+[\\r\\n/]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+";

void require(bool ok, const char* what) {
  if (!ok) throw std::invalid_argument(std::string("tokenizer.json: unsupported ") + what);
}

bool has_type(const Json& v, const char* name) {
  return v.is_object() && v.contains("type") && v["type"] == name;
}

// The pipeline this tokenizer implements; anything else is rejected rather than silently mis-tokenised.
void validate_pipeline(const Json& root, const Json& model) {
  require(!root.contains("normalizer") || root["normalizer"].is_null(), "normalizer (expected none)");
  const Json& pre = root.at("pre_tokenizer");
  require(has_type(pre, "Sequence") && pre.contains("pretokenizers") && pre["pretokenizers"].is_array() &&
              pre["pretokenizers"].size() == 2,
          "pre_tokenizer");
  const Json& split = pre["pretokenizers"][0];
  const Json& bytes = pre["pretokenizers"][1];
  require(has_type(split, "Split") && split.contains("pattern") && split["pattern"].is_object() &&
              split["pattern"].value("Regex", std::string{}) == kSplitPattern &&
              split.value("behavior", std::string{}) == "Isolated" && split.value("invert", Json(false)) == false,
          "pre_tokenizer.Split");
  require(has_type(bytes, "ByteLevel") && bytes.value("add_prefix_space", Json(true)) == false &&
              bytes.value("use_regex", Json(true)) == false,
          "pre_tokenizer.ByteLevel");
  require(has_type(root.at("decoder"), "ByteLevel"), "decoder");
  require(has_type(model, "BPE"), "model.type");
  require(model.value("ignore_merges", Json(false)) == true, "model.ignore_merges (expected true)");
  require(model.value("byte_fallback", Json(false)) == false, "model.byte_fallback");
  require(!model.contains("dropout") || model["dropout"].is_null(), "model.dropout");
  for (const char* field : {"continuing_subword_prefix", "end_of_word_suffix"}) {
    require(!model.contains(field) || model[field].is_null() || model[field] == "", field);
  }
}

// GPT-2 byte <-> unicode alphabet.
std::array<std::int32_t, 256> byte_to_codepoint() {
  std::array<std::int32_t, 256> map{};
  std::int32_t next = 256;
  for (int b = 0; b < 256; ++b) {
    const bool visible = (b >= 33 && b <= 126) || (b >= 161 && b <= 172) || (b >= 174 && b <= 255);
    map[static_cast<std::size_t>(b)] = visible ? b : next++;
  }
  return map;
}

std::string byte_level_to_bytes(std::string_view token, const std::unordered_map<std::int32_t, char>& decoder) {
  std::string bytes;
  bytes.reserve(token.size());
  for (const auto& cp : uni::utf8_codepoints(token, "tokenizer.json vocabulary")) {
    const auto it = decoder.find(cp.value);
    if (it == decoder.end()) {
      throw std::invalid_argument("tokenizer.json vocabulary token outside the byte-level alphabet");
    }
    bytes.push_back(it->second);
  }
  return bytes;
}

std::uint64_t pair_key(std::int32_t left, std::int32_t right) noexcept {
  return (static_cast<std::uint64_t>(static_cast<std::uint32_t>(left)) << 32U) | static_cast<std::uint32_t>(right);
}

// ---- split regex --------------------------------------------------------------------------------------------
// Character classes of the pattern, one bit each, precomputed per codepoint.
enum : std::uint16_t {
  kLu = 1, kLl = 2, kLt = 4, kLm = 8, kLo = 16, kM = 32, kN = 64, kSpace = 128, kCrLf = 256, kBlank = 512,
  kSlash = 1024,
};
constexpr std::uint16_t kLetter = kLu | kLl | kLt | kLm | kLo;

std::uint16_t classify(std::int32_t cp) {
  std::uint16_t m = 0;
  switch (utf8proc_category(cp)) {
    case UTF8PROC_CATEGORY_LU: m |= kLu; break;
    case UTF8PROC_CATEGORY_LL: m |= kLl; break;
    case UTF8PROC_CATEGORY_LT: m |= kLt; break;
    case UTF8PROC_CATEGORY_LM: m |= kLm; break;
    case UTF8PROC_CATEGORY_LO: m |= kLo; break;
    case UTF8PROC_CATEGORY_MN:
    case UTF8PROC_CATEGORY_MC:
    case UTF8PROC_CATEGORY_ME: m |= kM; break;
    case UTF8PROC_CATEGORY_ND:
    case UTF8PROC_CATEGORY_NL:
    case UTF8PROC_CATEGORY_NO: m |= kN; break;
    default: break;
  }
  if (uni::is_regex_space(cp)) m |= kSpace;
  if (cp == '\r' || cp == '\n') m |= kCrLf;
  if (cp == ' ') m |= kBlank;
  if (cp == '/') m |= kSlash;
  return m;
}

using ClassPred = bool (*)(std::uint16_t);
bool c_prefix(std::uint16_t m) { return !(m & (kCrLf | kLetter | kN)); }                 // [^\r\n\p{L}\p{N}]
bool c_upper(std::uint16_t m) { return (m & (kLu | kLt | kLm | kLo | kM)) != 0; }        // [\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]
bool c_lower(std::uint16_t m) { return (m & (kLl | kLm | kLo | kM)) != 0; }              // [\p{Ll}\p{Lm}\p{Lo}\p{M}]
bool c_number(std::uint16_t m) { return (m & kN) != 0; }                                 // \p{N}
bool c_blank(std::uint16_t m) { return (m & kBlank) != 0; }                              // ' '
bool c_symbol(std::uint16_t m) { return !(m & (kSpace | kLetter | kN)); }                // [^\s\p{L}\p{N}]
bool c_crlf_slash(std::uint16_t m) { return (m & (kCrLf | kSlash)) != 0; }               // [\r\n/]
bool c_crlf(std::uint16_t m) { return (m & kCrLf) != 0; }                                // [\r\n]
bool c_space(std::uint16_t m) { return (m & kSpace) != 0; }                              // \s

struct Element {
  ClassPred pred;
  std::size_t min, max;
};
constexpr std::size_t kMany = std::numeric_limits<std::size_t>::max();

struct Text {
  const std::vector<std::int32_t>& cps;
  const std::vector<std::uint16_t>& cls;
};

// Backtracking match of a sequence of greedy quantified classes (Oniguruma semantics: the first successful
// path wins), followed by an optional lookahead `(?!\S)`.
bool match_elements(const Text& t, std::span<const Element> elems, std::size_t pos, bool not_followed_by_nonspace,
                    std::size_t& end) {
  if (elems.empty()) {
    if (not_followed_by_nonspace && pos < t.cls.size() && !c_space(t.cls[pos])) return false;
    end = pos;
    return true;
  }
  const Element& e = elems.front();
  std::size_t n = 0;
  while (n < e.max && pos + n < t.cls.size() && e.pred(t.cls[pos + n])) ++n;
  if (n < e.min) return false;
  for (std::size_t k = n + 1; k-- > e.min;) {
    if (match_elements(t, elems.subspan(1), pos + k, not_followed_by_nonspace, end)) return true;
  }
  return false;
}

// (?i:'s|'t|'re|'ve|'m|'ll|'d)? at pos; returns the end.
std::size_t contraction(const Text& t, std::size_t pos) {
  const auto& c = t.cps;
  if (pos >= c.size() || c[pos] != '\'') return pos;
  auto is = [&](std::size_t i, char lower) {
    if (i >= c.size()) return false;
    const std::int32_t v = c[i];
    if (v == lower || v == lower - 32) return true;
    return lower == 's' && v == 0x17F;  // LATIN SMALL LETTER LONG S case-folds to 's'
  };
  if (is(pos + 1, 's') || is(pos + 1, 't') || is(pos + 1, 'm') || is(pos + 1, 'd')) return pos + 2;
  if (is(pos + 1, 'r') && is(pos + 2, 'e')) return pos + 3;
  if (is(pos + 1, 'v') && is(pos + 2, 'e')) return pos + 3;
  if (is(pos + 1, 'l') && is(pos + 2, 'l')) return pos + 3;
  return pos;
}

// End (codepoint index) of the pre-tokenizer piece starting at pos.
std::size_t piece_end(const Text& t, std::size_t pos) {
  static constexpr Element kA1[] = {{c_prefix, 0, 1}, {c_upper, 0, kMany}, {c_lower, 1, kMany}};
  static constexpr Element kA2[] = {{c_prefix, 0, 1}, {c_upper, 1, kMany}, {c_lower, 0, kMany}};
  static constexpr Element kA3[] = {{c_number, 1, 3}};
  static constexpr Element kA4[] = {{c_blank, 0, 1}, {c_symbol, 1, kMany}, {c_crlf_slash, 0, kMany}};
  static constexpr Element kA5[] = {{c_space, 0, kMany}, {c_crlf, 1, kMany}};
  static constexpr Element kA6[] = {{c_space, 1, kMany}};
  std::size_t end = pos;
  if (match_elements(t, kA1, pos, false, end)) return contraction(t, end);
  if (match_elements(t, kA2, pos, false, end)) return contraction(t, end);
  if (match_elements(t, kA3, pos, false, end)) return end;
  if (match_elements(t, kA4, pos, false, end)) return end;
  if (match_elements(t, kA5, pos, false, end)) return end;
  if (match_elements(t, kA6, pos, true, end)) return end;
  if (match_elements(t, kA6, pos, false, end)) return end;
  return pos + 1;  // unreachable: every character matches some alternative; an unmatched one is its own piece
}

struct BpeNode {
  std::int32_t symbol = -1;
  std::int32_t previous = -1;
  std::int32_t next = -1;
  std::uint32_t generation = 0;
  bool live = true;
};

struct BpeCandidate {
  std::int32_t rank = 0;
  std::int32_t left = -1;
  std::int32_t right = -1;
  std::int32_t result = -1;
  std::uint32_t left_generation = 0;
  std::uint32_t right_generation = 0;
};

struct LaterBpeCandidate {
  bool operator()(const BpeCandidate& lhs, const BpeCandidate& rhs) const noexcept {
    if (lhs.rank != rhs.rank) return lhs.rank > rhs.rank;
    return lhs.left > rhs.left;
  }
};

std::string read_file(const std::filesystem::path& path) {
  std::ifstream in(path, std::ios::binary);
  if (!in) throw std::runtime_error("cannot open " + path.string());
  std::ostringstream ss;
  ss << in.rdbuf();
  return ss.str();
}

}  // namespace

void BpeMergeTable::reset(std::size_t rule_count) {
  std::size_t capacity = 1;
  while (capacity < rule_count * 2U) capacity <<= 1U;
  slots_.assign(capacity, Entry{});
  mask_ = capacity - 1U;
}

bool BpeMergeTable::insert(std::uint64_t key, int32_t rank, int32_t result) {
  std::uint64_t index = mix(key) & mask_;
  while (slots_[index].result >= 0) {
    if (slots_[index].key == key) return false;
    index = (index + 1U) & mask_;
  }
  slots_[index] = Entry{key, rank, result};
  return true;
}

Tokenizer Tokenizer::load(const std::filesystem::path& model_dir) {
  return from_json(read_file(model_dir / "tokenizer.json"));
}

Tokenizer Tokenizer::from_json(std::string_view tokenizer_json) {
  const Json root = Json::parse(tokenizer_json);
  const Json& model = root.at("model");
  validate_pipeline(root, model);

  std::unordered_map<std::int32_t, char> byte_decoder;
  const auto b2c = byte_to_codepoint();
  for (int b = 0; b < 256; ++b) byte_decoder.emplace(b2c[static_cast<std::size_t>(b)], static_cast<char>(b));

  Tokenizer tok;
  const Json& vocab = model.at("vocab");
  std::unordered_map<std::string, std::int32_t> by_symbol;  // byte-level token string -> id (load only)
  by_symbol.reserve(vocab.size());
  std::int32_t max_id = -1;
  for (const auto& item : vocab.items()) {
    const auto id = item.value().get<std::int32_t>();
    if (id < 0) throw std::invalid_argument("tokenizer.json: negative vocabulary id");
    if (!by_symbol.emplace(item.key(), id).second) throw std::invalid_argument("tokenizer.json: duplicate token");
    max_id = std::max(max_id, id);
  }
  for (const Json& a : root.at("added_tokens")) max_id = std::max(max_id, a.at("id").get<std::int32_t>());
  const auto size = static_cast<std::size_t>(max_id + 1);
  tok.token_bytes_.assign(size, {});
  tok.special_.assign(size, false);
  tok.added_.assign(size, false);
  std::vector<bool> present(size, false);
  tok.vocab_by_bytes_.reserve(by_symbol.size());
  for (const auto& [symbol, id] : by_symbol) {
    if (present[static_cast<std::size_t>(id)]) throw std::invalid_argument("tokenizer.json: duplicate vocabulary id");
    present[static_cast<std::size_t>(id)] = true;
    std::string bytes = byte_level_to_bytes(symbol, byte_decoder);
    tok.vocab_by_bytes_.emplace(bytes, id);
    tok.token_bytes_[static_cast<std::size_t>(id)] = std::move(bytes);
  }

  for (const Json& a : root.at("added_tokens")) {
    AddedToken t;
    t.id = a.at("id").get<std::int32_t>();
    t.content = a.at("content").get<std::string>();
    t.special = a.at("special").get<bool>();
    if (t.content.empty() || a.value("single_word", false) || a.value("lstrip", false) ||
        a.value("rstrip", false) || a.value("normalized", false)) {
      throw std::invalid_argument("tokenizer.json: unsupported added token " + t.content);
    }
    const auto index = static_cast<std::size_t>(t.id);
    if (present[index]) throw std::invalid_argument("tokenizer.json: added token overlaps the vocabulary");
    present[index] = true;
    tok.token_bytes_[index] = t.content;
    tok.special_[index] = t.special;
    tok.added_[index] = true;
    tok.added_by_content_.emplace(t.content, t.id);
    tok.max_added_bytes_ = std::max(tok.max_added_bytes_, t.content.size());
    tok.added_tokens_.push_back(std::move(t));
  }
  for (std::size_t i = 0; i < tok.added_tokens_.size(); ++i) {
    tok.added_candidates_[static_cast<unsigned char>(tok.added_tokens_[i].content.front())].push_back(i);
  }
  for (auto& candidates : tok.added_candidates_) {  // longest first: leftmost-longest matching
    std::stable_sort(candidates.begin(), candidates.end(), [&](std::size_t a, std::size_t b) {
      return tok.added_tokens_[a].content.size() > tok.added_tokens_[b].content.size();
    });
  }

  const Json& merges = model.at("merges");
  tok.merges_.reset(merges.size());
  std::int32_t rank = 0;
  for (const Json& item : merges) {
    std::string left, right;
    if (item.is_array() && item.size() == 2) {
      left = item[0].get<std::string>();
      right = item[1].get<std::string>();
    } else {
      const std::string pair = item.get<std::string>();
      const std::size_t space = pair.find(' ');
      if (space == std::string::npos) throw std::invalid_argument("tokenizer.json: malformed merge");
      left = pair.substr(0, space);
      right = pair.substr(space + 1);
    }
    const auto l = by_symbol.find(left), r = by_symbol.find(right), m = by_symbol.find(left + right);
    if (l == by_symbol.end() || r == by_symbol.end() || m == by_symbol.end()) {
      throw std::invalid_argument("tokenizer.json: merge references a symbol outside the vocabulary");
    }
    if (!tok.merges_.insert(pair_key(l->second, r->second), rank++, m->second)) {
      throw std::invalid_argument("tokenizer.json: duplicate merge");
    }
  }
  for (int b = 0; b < 256; ++b) {
    const auto it = by_symbol.find(uni::codepoint_to_utf8(b2c[static_cast<std::size_t>(b)]));
    if (it == by_symbol.end()) throw std::invalid_argument("tokenizer.json: byte token missing");
    tok.byte_ids_[static_cast<std::size_t>(b)] = it->second;
  }
  return tok;
}

void Tokenizer::encode_piece(std::string_view piece, std::vector<int32_t>& ids) const {
  // ignore_merges: a piece that is itself a vocabulary token is emitted whole.
  if (const auto it = vocab_by_bytes_.find(std::string(piece)); it != vocab_by_bytes_.end()) {
    ids.push_back(it->second);
    return;
  }
  std::vector<BpeNode> nodes(piece.size());
  for (std::size_t i = 0; i < piece.size(); ++i) {
    nodes[i] = BpeNode{.symbol = byte_ids_[static_cast<unsigned char>(piece[i])],
                       .previous = static_cast<std::int32_t>(i) - 1,
                       .next = i + 1 == piece.size() ? -1 : static_cast<std::int32_t>(i + 1)};
  }
  std::priority_queue<BpeCandidate, std::vector<BpeCandidate>, LaterBpeCandidate> queue;
  const auto push_candidate = [&](std::int32_t left) {
    if (left < 0 || !nodes[static_cast<std::size_t>(left)].live) return;
    const std::int32_t right = nodes[static_cast<std::size_t>(left)].next;
    if (right < 0) return;
    const auto* rule = merges_.find(
        pair_key(nodes[static_cast<std::size_t>(left)].symbol, nodes[static_cast<std::size_t>(right)].symbol));
    if (rule == nullptr) return;
    queue.push(BpeCandidate{rule->rank, left, right, rule->result, nodes[static_cast<std::size_t>(left)].generation,
                            nodes[static_cast<std::size_t>(right)].generation});
  };
  for (std::size_t i = 0; i + 1 < nodes.size(); ++i) push_candidate(static_cast<std::int32_t>(i));
  while (!queue.empty()) {
    const BpeCandidate c = queue.top();
    queue.pop();
    BpeNode& left = nodes[static_cast<std::size_t>(c.left)];
    BpeNode& right = nodes[static_cast<std::size_t>(c.right)];
    if (!left.live || !right.live || left.next != c.right || left.generation != c.left_generation ||
        right.generation != c.right_generation) {
      continue;
    }
    left.symbol = c.result;
    ++left.generation;
    left.next = right.next;
    right.live = false;
    ++right.generation;
    if (left.next >= 0) nodes[static_cast<std::size_t>(left.next)].previous = c.left;
    push_candidate(left.previous);
    push_candidate(c.left);
  }
  for (std::int32_t n = nodes.empty() ? -1 : 0; n >= 0; n = nodes[static_cast<std::size_t>(n)].next) {
    ids.push_back(nodes[static_cast<std::size_t>(n)].symbol);
  }
}

void Tokenizer::encode_ordinary(std::string_view text, std::vector<int32_t>& ids) const {
  if (text.empty()) return;
  const auto spans = uni::utf8_codepoints(text, "Tokenizer::encode input");
  std::vector<std::int32_t> cps(spans.size());
  std::vector<std::uint16_t> cls(spans.size());
  for (std::size_t i = 0; i < spans.size(); ++i) {
    cps[i] = spans[i].value;
    cls[i] = classify(spans[i].value);
  }
  const Text t{cps, cls};
  for (std::size_t pos = 0; pos < cps.size();) {
    const std::size_t end = piece_end(t, pos);
    const std::size_t b = spans[pos].offset;
    const std::size_t e = end < spans.size() ? spans[end].offset : text.size();
    encode_piece(text.substr(b, e - b), ids);
    pos = end;
  }
}

std::vector<int32_t> Tokenizer::encode(std::string_view text, bool special) const {
  std::vector<int32_t> ids;
  ids.reserve(text.size() / 3 + 1);
  if (!special) {
    encode_ordinary(text, ids);
    return ids;
  }
  encode_special(text, ids, nullptr, 0);
  return ids;
}

void Tokenizer::encode_special(std::string_view text, std::vector<int32_t>& ids, std::vector<TokenBoundary>* boundaries,
                               std::size_t byte_base) const {
  std::size_t ordinary_begin = 0;
  for (std::size_t pos = 0; pos < text.size();) {
    const AddedToken* match = nullptr;
    for (const std::size_t index : added_candidates_[static_cast<unsigned char>(text[pos])]) {
      const AddedToken& token = added_tokens_[index];
      if (text.compare(pos, token.content.size(), token.content) == 0) {
        match = &token;
        break;
      }
    }
    if (match == nullptr) {
      ++pos;
      continue;
    }
    encode_ordinary(text.substr(ordinary_begin, pos - ordinary_begin), ids);
    ids.push_back(match->id);
    pos += match->content.size();
    ordinary_begin = pos;
    if (boundaries != nullptr) boundaries->push_back(TokenBoundary{byte_base + pos, ids.size()});
  }
  encode_ordinary(text.substr(ordinary_begin), ids);
}

std::string Tokenizer::decode(std::span<const int32_t> ids) const {
  std::string out;
  for (const int32_t id : ids) {
    if (id >= 0 && static_cast<std::size_t>(id) < token_bytes_.size()) out += token_bytes_[static_cast<std::size_t>(id)];
  }
  return out;
}

std::string Tokenizer::decode_lossy(std::span<const int32_t> ids) const { return uni::utf8_lossy(decode(ids)); }

std::string Tokenizer::token_bytes(int32_t id) const {
  if (id < 0 || static_cast<std::size_t>(id) >= token_bytes_.size()) return {};
  return token_bytes_[static_cast<std::size_t>(id)];
}

std::optional<int32_t> Tokenizer::token_id(std::string_view added_token) const {
  const auto it = added_by_content_.find(std::string(added_token));
  if (it == added_by_content_.end()) return std::nullopt;
  return it->second;
}

}  // namespace ingine::text
