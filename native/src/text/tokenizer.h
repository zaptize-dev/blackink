// Adapted from fngine (github.com/satellitedown/fngine, Apache-2.0) for ingine.
// Inkling-2 tokenizer (tokenizer.json): byte-level BPE with ignore_merges, the o200k-style split regex as a
// hand-written matcher, no normalizer, added tokens recognised as literal spans (leftmost-longest).
#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

namespace ingine::text {

struct AddedToken {
  int32_t id = -1;
  std::string content;
  bool special = false;
};

// Merge table of the BPE encoder, open addressed with linear probing (one 16-byte slot per rule, at least half
// the slots empty). An empty slot has result < 0.
class BpeMergeTable {
 public:
  void reset(std::size_t rule_count);
  bool insert(std::uint64_t key, int32_t rank, int32_t result);  // false: duplicate pair
  struct Entry {
    std::uint64_t key = 0;
    int32_t rank = 0;
    int32_t result = -1;
  };
  const Entry* find(std::uint64_t key) const noexcept {
    std::uint64_t index = mix(key) & mask_;
    while (true) {
      const Entry& entry = slots_[index];
      if (entry.result < 0) return nullptr;
      if (entry.key == key) return &entry;
      index = (index + 1U) & mask_;
    }
  }

 private:
  static std::uint64_t mix(std::uint64_t key) noexcept {  // splitmix64 finalizer
    key += 0x9e3779b97f4a7c15ULL;
    key = (key ^ (key >> 30U)) * 0xbf58476d1ce4e5b9ULL;
    key = (key ^ (key >> 27U)) * 0x94d049bb133111ebULL;
    return key ^ (key >> 31U);
  }
  std::vector<Entry> slots_ = std::vector<Entry>(1);
  std::uint64_t mask_ = 0;
};

// End of an added token in an encoded text: the byte offset just after it and the number of ids up to and
// including it. Added tokens split the BPE, so the ids before such a boundary depend only on the text before it.
struct TokenBoundary {
  std::size_t byte = 0;
  std::size_t tokens = 0;
};

class Tokenizer {
 public:
  static Tokenizer load(const std::filesystem::path& model_dir);  // reads model_dir/tokenizer.json
  static Tokenizer from_json(std::string_view tokenizer_json);

  // special: recognise added tokens in the text (tokenizer.encode(text, add_special_tokens=False));
  // otherwise the whole text is ordinary (encode_special_tokens=True). Throws std::invalid_argument on
  // invalid UTF-8.
  std::vector<int32_t> encode(std::string_view text, bool special) const;
  // encode(text, true) appended to ids; with boundaries, records every added token's TokenBoundary (byte
  // offsets shifted by byte_base, token counts absolute in ids).
  void encode_special(std::string_view text, std::vector<int32_t>& ids, std::vector<TokenBoundary>* boundaries,
                      std::size_t byte_base) const;
  // Longest added token in bytes: whether a byte starts an added token depends on at most this many bytes.
  std::size_t max_added_token_bytes() const noexcept { return max_added_bytes_; }
  // Concatenated token bytes, raw UTF-8 that may be partial or invalid; unknown ids contribute nothing.
  std::string decode(std::span<const int32_t> ids) const;
  // tokenizer.decode(ids): decode() with invalid sequences replaced by U+FFFD (String::from_utf8_lossy).
  std::string decode_lossy(std::span<const int32_t> ids) const;
  std::string token_bytes(int32_t id) const;
  std::optional<int32_t> token_id(std::string_view added_token) const;

  bool is_special(int32_t id) const noexcept {
    return id >= 0 && static_cast<std::size_t>(id) < special_.size() && special_[static_cast<std::size_t>(id)];
  }
  bool is_added(int32_t id) const noexcept {
    return id >= 0 && static_cast<std::size_t>(id) < added_.size() && added_[static_cast<std::size_t>(id)];
  }
  const std::vector<AddedToken>& added_tokens() const noexcept { return added_tokens_; }
  // Number of ids (vocabulary + added tokens), i.e. one past the largest id.
  std::size_t vocab_size() const noexcept { return token_bytes_.size(); }

 private:
  Tokenizer() = default;
  void encode_ordinary(std::string_view text, std::vector<int32_t>& ids) const;
  void encode_piece(std::string_view piece, std::vector<int32_t>& ids) const;

  std::vector<std::string> token_bytes_;  // raw bytes per id (added tokens: their content)
  std::vector<bool> special_;
  std::vector<bool> added_;
  std::unordered_map<std::string, int32_t> vocab_by_bytes_;  // ordinary vocabulary, raw bytes -> id
  BpeMergeTable merges_;
  std::array<int32_t, 256> byte_ids_{};
  std::size_t max_added_bytes_ = 0;
  std::vector<AddedToken> added_tokens_;
  std::array<std::vector<std::size_t>, 256> added_candidates_;  // by first byte, longest content first
  std::unordered_map<std::string, int32_t> added_by_content_;
};

}  // namespace ingine::text
