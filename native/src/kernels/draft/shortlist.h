// The drafter's shortlist ranking (draft_shortlist.bin, built by native/tools/build_draft_shortlist.py and
// embedded at build time): token ids by descending corpus frequency, 65,536 entries.
#pragma once

#include <cstdint>
#include <span>

namespace ingine::kernels::draft {

std::span<const std::uint32_t> draft_shortlist_ranks();

}  // namespace ingine::kernels::draft
