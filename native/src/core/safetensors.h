#pragma once

// Safetensors checkpoints: header parsing, tensor lookup, host reads (pread) and a fast host-to-device
// upload path (parallel reads into pinned staging buffers overlapped with asynchronous copies).
//
// File format: an 8-byte little-endian header length N, N bytes of JSON
//   {"__metadata__": {str: str}, "<name>": {"dtype": "BF16", "shape": [..], "data_offsets": [b, e]}, ...}
// then the data region; data_offsets are relative to the start of the data region.

#include "core/dtype.h"
#include "core/tensor.h"

#include <cuda_runtime.h>

#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <deque>
#include <exception>
#include <filesystem>
#include <map>
#include <memory>
#include <mutex>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <thread>
#include <unordered_map>
#include <vector>

namespace ingine {

// Element size in bytes of a safetensors dtype string ("BF16", "F8_E4M3", "U8", ...); throws for
// dtypes without a whole-byte element size or unknown names.
std::size_t safetensors_dtype_size(std::string_view dtype);
// The ingine DType for a safetensors dtype string, if ingine has one (F8_E4M3 -> FP8_E4M3FN).
std::optional<DType> safetensors_to_dtype(std::string_view dtype);

struct SafetensorsTensor {
    std::string name;
    std::string dtype;               // safetensors dtype string as stored ("BF16", "F8_E4M3", ...)
    std::vector<std::int64_t> shape; // as stored: outermost dimension first (PyTorch order)
    std::uint64_t offset = 0;        // absolute byte offset of the first element in the file
    std::uint64_t bytes  = 0;

    [[nodiscard]] std::int64_t numel() const noexcept;
    // ingine DType; throws when the stored dtype has no ingine equivalent.
    [[nodiscard]] DType ingine_dtype() const;
    // Non-owning ingine::Tensor over `data` (device or host) with this tensor's dtype and shape:
    // ne[0] is the innermost (last stored) dimension. Throws for rank > 4 or dims beyond int32.
    [[nodiscard]] Tensor view(void* data) const;
};

class SafetensorsFile {
public:
    // Opens the file and parses its header. Throws std::runtime_error on I/O or format errors.
    explicit SafetensorsFile(const std::filesystem::path& path);
    ~SafetensorsFile();

    SafetensorsFile(const SafetensorsFile&)            = delete;
    SafetensorsFile& operator=(const SafetensorsFile&) = delete;
    SafetensorsFile(SafetensorsFile&& other) noexcept;
    SafetensorsFile& operator=(SafetensorsFile&& other) noexcept;

    [[nodiscard]] const std::filesystem::path& path() const noexcept { return path_; }
    [[nodiscard]] int fd() const noexcept { return fd_; }
    [[nodiscard]] std::uint64_t file_size() const noexcept { return file_size_; }
    [[nodiscard]] std::uint64_t data_offset() const noexcept { return data_offset_; }
    // "__metadata__" (string values only, as the format requires); empty when absent.
    [[nodiscard]] const std::map<std::string, std::string>& metadata() const noexcept {
        return metadata_;
    }
    // Every tensor, in data-offset order.
    [[nodiscard]] const std::vector<SafetensorsTensor>& tensors() const noexcept { return tensors_; }
    [[nodiscard]] std::vector<std::string> names() const;
    [[nodiscard]] const SafetensorsTensor* find(std::string_view name) const;
    [[nodiscard]] const SafetensorsTensor& at(std::string_view name) const; // throws when absent
    [[nodiscard]] bool contains(std::string_view name) const { return find(name) != nullptr; }

    // Reads `bytes` bytes at absolute file offset `offset` into host memory (pread loop).
    void read_range(std::uint64_t offset, std::uint64_t bytes, void* dst) const;
    void read(const SafetensorsTensor& tensor, void* dst) const;
    [[nodiscard]] std::vector<std::byte> read(std::string_view name) const;

private:
    struct NameHash {
        using is_transparent = void;
        std::size_t operator()(std::string_view s) const noexcept {
            return std::hash<std::string_view>{}(s);
        }
    };

    std::filesystem::path path_;
    int fd_                    = -1;
    std::uint64_t file_size_   = 0;
    std::uint64_t data_offset_ = 0;
    std::map<std::string, std::string> metadata_;
    std::vector<SafetensorsTensor> tensors_;
    std::unordered_map<std::string, std::size_t, NameHash, std::equal_to<>> index_;
};

// A tensor together with the file holding it.
struct SafetensorsRef {
    const SafetensorsFile* file     = nullptr;
    const SafetensorsTensor* tensor = nullptr;

    explicit operator bool() const noexcept { return tensor != nullptr; }
    const SafetensorsTensor* operator->() const noexcept { return tensor; }
    const SafetensorsTensor& operator*() const noexcept { return *tensor; }
};

// A checkpoint made of one or more safetensors files.
class SafetensorsModel {
public:
    // `path` is a .safetensors file, or a directory holding model.safetensors.index.json (every file
    // named in its weight_map; the index decides which file holds which tensor) or model.safetensors.
    explicit SafetensorsModel(const std::filesystem::path& path);

    SafetensorsModel(SafetensorsModel&&) noexcept            = default;
    SafetensorsModel& operator=(SafetensorsModel&&) noexcept = default;

    [[nodiscard]] const std::vector<std::unique_ptr<SafetensorsFile>>& files() const noexcept {
        return files_;
    }
    [[nodiscard]] std::vector<std::string> names() const; // sorted
    [[nodiscard]] std::size_t size() const noexcept { return index_.size(); }
    [[nodiscard]] SafetensorsRef find(std::string_view name) const;
    [[nodiscard]] SafetensorsRef at(std::string_view name) const; // throws when absent
    [[nodiscard]] bool contains(std::string_view name) const { return static_cast<bool>(find(name)); }
    [[nodiscard]] std::vector<std::byte> read(std::string_view name) const;

private:
    std::vector<std::unique_ptr<SafetensorsFile>> files_;
    std::map<std::string, SafetensorsRef, std::less<>> index_;
};

// One host-to-device copy: `bytes` bytes at absolute offset `offset` of `file` to device `dst`.
struct UploadRequest {
    const SafetensorsFile* file = nullptr;
    std::uint64_t offset        = 0;
    std::uint64_t bytes         = 0;
    void* dst                   = nullptr;
};

[[nodiscard]] UploadRequest upload_request(const SafetensorsFile& file,
                                           const SafetensorsTensor& tensor, void* dst);
[[nodiscard]] UploadRequest upload_request(SafetensorsRef ref, void* dst);

struct UploadOptions {
    std::size_t chunk_bytes = std::size_t(32) << 20; // staging buffer size and max bytes per copy
    int staging_buffers     = 12;                    // pinned buffers in flight
    int reader_threads      = 6;                     // threads reading file data into staging
};

struct UploadStats {
    std::uint64_t bytes = 0;
    double seconds      = 0;
    [[nodiscard]] double gib_per_second() const noexcept {
        return seconds > 0 ? double(bytes) / double(1ull << 30) / seconds : 0.0;
    }
};

// Uploads file ranges to device memory. Reader threads pread chunks into pinned staging buffers and
// enqueue each chunk's cudaMemcpyAsync as soon as it is read, so disk/page-cache reads, the copies of
// other chunks and further reads overlap. Owns the pinned staging memory (chunk_bytes *
// staging_buffers) and the reader threads for its lifetime: create one for a whole model load.
class DeviceUploader {
public:
    explicit DeviceUploader(UploadOptions options = {});
    ~DeviceUploader();

    DeviceUploader(const DeviceUploader&)            = delete;
    DeviceUploader& operator=(const DeviceUploader&) = delete;

    // Copies every request (on `stream`, after the work already queued there) and returns once all
    // bytes have arrived on the device. Destinations must not overlap. The calling thread's current
    // CUDA device must be the one current at construction. One upload() at a time per uploader.
    UploadStats upload(std::span<const UploadRequest> requests, cudaStream_t stream);

    [[nodiscard]] const UploadOptions& options() const noexcept { return options_; }

private:
    struct Chunk {
        const SafetensorsFile* file;
        std::uint64_t offset;
        std::uint64_t bytes;
        void* dst;
    };
    struct Slot {
        std::byte* host    = nullptr;
        cudaEvent_t done   = nullptr; // recorded after the slot's last copy
        bool copy_inflight = false;
    };

    void worker_main();
    void run_chunks();
    int acquire_slot();
    void release_slot(int slot);

    UploadOptions options_;
    void* pinned_ = nullptr;
    std::vector<Slot> slots_;
    std::deque<int> free_slots_; // FIFO: reuse the slot whose copy was issued longest ago
    std::vector<std::thread> workers_;

    std::mutex mutex_;
    std::condition_variable work_cv_;
    std::condition_variable done_cv_;
    std::condition_variable slot_cv_;
    std::mutex stream_mutex_;

    // Current job (guarded by mutex_ unless noted).
    std::uint64_t generation_ = 0;
    bool stopping_            = false;
    int active_workers_       = 0;
    std::vector<Chunk> chunks_;
    std::size_t next_chunk_ = 0;
    cudaStream_t stream_    = nullptr;
    int device_             = 0; // CUDA device current at construction (owns the events)
    std::exception_ptr error_;
};

} // namespace ingine
