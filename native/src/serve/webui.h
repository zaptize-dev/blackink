// Embedded official llama.cpp SvelteKit WebUI (llama-ui) static assets (gzip-compressed).
#pragma once

#include <cstddef>
#include <cstdint>
#include <string_view>

namespace ingine::serve {

struct WebUiAsset {
  std::string_view path;
  std::string_view mime_type;
  std::string_view etag;
  const uint8_t* gzip_data;
  std::size_t gzip_size;
  bool immutable;
};

const WebUiAsset* find_webui_asset(std::string_view path);

}  // namespace ingine::serve
