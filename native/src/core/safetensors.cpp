#include "core/safetensors.h"

#include "core/nvtx.h"

#include <nlohmann/json.hpp>

#include <algorithm>
#include <cerrno>
#include <chrono>
#include <cstring>
#include <fstream>
#include <limits>
#include <set>
#include <stdexcept>
#include <utility>

#include <fcntl.h>
#include <sys/stat.h>
#include <unistd.h>

namespace ingine {
namespace {

constexpr std::uint64_t kMaxHeaderBytes = 100ull << 20; // the safetensors reference limit

[[noreturn]] void fail(const std::filesystem::path& path, const std::string& what) {
    throw std::runtime_error("safetensors " + path.string() + ": " + what);
}

[[noreturn]] void fail_errno(const std::filesystem::path& path, const std::string& what) {
    fail(path, what + ": " + std::strerror(errno));
}

std::string cuda_message(const char* what, cudaError_t err) {
    return std::string(what) + ": " + cudaGetErrorName(err) + ": " + cudaGetErrorString(err);
}

void cuda_throw(cudaError_t err, const char* what) {
    if (err != cudaSuccess) { throw std::runtime_error(cuda_message(what, err)); }
}

std::uint64_t checked_mul(std::uint64_t a, std::uint64_t b, const std::filesystem::path& path) {
    if (b != 0 && a > std::numeric_limits<std::uint64_t>::max() / b) {
        fail(path, "tensor size overflows");
    }
    return a * b;
}

} // namespace

std::size_t safetensors_dtype_size(std::string_view dtype) {
    if (dtype == "BOOL" || dtype == "U8" || dtype == "I8" || dtype == "F8_E4M3" ||
        dtype == "F8_E5M2" || dtype == "F8_E8M0") {
        return 1;
    }
    if (dtype == "I16" || dtype == "U16" || dtype == "F16" || dtype == "BF16") { return 2; }
    if (dtype == "I32" || dtype == "U32" || dtype == "F32") { return 4; }
    if (dtype == "I64" || dtype == "U64" || dtype == "F64") { return 8; }
    throw std::invalid_argument("unsupported safetensors dtype " + std::string(dtype));
}

std::optional<DType> safetensors_to_dtype(std::string_view dtype) {
    if (dtype == "BF16") { return DType::BF16; }
    if (dtype == "F32") { return DType::FP32; }
    if (dtype == "I32") { return DType::I32; }
    if (dtype == "U8") { return DType::U8; }
    if (dtype == "I64") { return DType::I64; }
    if (dtype == "I8") { return DType::I8; }
    if (dtype == "F16") { return DType::FP16; }
    if (dtype == "F8_E4M3") { return DType::FP8_E4M3FN; }
    return std::nullopt;
}

std::int64_t SafetensorsTensor::numel() const noexcept {
    std::int64_t n = 1;
    for (std::int64_t d : shape) { n *= d; }
    return n;
}

DType SafetensorsTensor::ingine_dtype() const {
    const std::optional<DType> d = safetensors_to_dtype(dtype);
    if (!d) { throw std::invalid_argument("tensor " + name + ": no ingine DType for " + dtype); }
    return *d;
}

Tensor SafetensorsTensor::view(void* data) const {
    if (shape.size() > 4) {
        throw std::invalid_argument("tensor " + name + ": rank " + std::to_string(shape.size()) +
                                    " exceeds ingine::Tensor rank 4");
    }
    Tensor t;
    t.data  = data;
    t.dtype = ingine_dtype();
    const std::size_t rank = shape.size();
    for (std::size_t i = 0; i < rank; ++i) {
        const std::int64_t d = shape[rank - 1 - i];
        if (d <= 0 || d > std::numeric_limits<std::int32_t>::max()) {
            throw std::invalid_argument("tensor " + name + ": dimension " + std::to_string(d) +
                                        " not representable in ingine::Tensor");
        }
        t.ne[i] = static_cast<std::int32_t>(d);
    }
    t.nb[0] = static_cast<std::int64_t>(dtype_size(t.dtype));
    for (int i = 1; i < 4; ++i) { t.nb[i] = t.nb[i - 1] * t.ne[i - 1]; }
    return t;
}

SafetensorsFile::SafetensorsFile(const std::filesystem::path& path) : path_(path) {
    fd_ = ::open(path.c_str(), O_RDONLY | O_CLOEXEC);
    if (fd_ < 0) { fail_errno(path_, "open failed"); }
    try {
        struct stat st {};
        if (::fstat(fd_, &st) != 0) { fail_errno(path_, "fstat failed"); }
        file_size_ = static_cast<std::uint64_t>(st.st_size);
        (void)::posix_fadvise(fd_, 0, 0, POSIX_FADV_SEQUENTIAL);

        if (file_size_ < 8) { fail(path_, "file shorter than the 8-byte header length"); }
        unsigned char len_bytes[8];
        read_range(0, 8, len_bytes);
        std::uint64_t header_len = 0;
        for (int i = 7; i >= 0; --i) { header_len = (header_len << 8) | len_bytes[i]; }
        if (header_len > kMaxHeaderBytes || header_len > file_size_ - 8) {
            fail(path_, "invalid header length " + std::to_string(header_len));
        }
        data_offset_ = 8 + header_len;
        std::string header(header_len, '\0');
        read_range(8, header_len, header.data());

        nlohmann::json j;
        try {
            j = nlohmann::json::parse(header);
        } catch (const nlohmann::json::exception& e) {
            fail(path_, std::string("header is not valid JSON: ") + e.what());
        }
        if (!j.is_object()) { fail(path_, "header is not a JSON object"); }

        const std::uint64_t data_size = file_size_ - data_offset_;
        for (auto it = j.begin(); it != j.end(); ++it) {
            if (it.key() == "__metadata__") {
                if (!it.value().is_object()) { fail(path_, "__metadata__ is not an object"); }
                for (auto m = it.value().begin(); m != it.value().end(); ++m) {
                    if (!m.value().is_string()) {
                        fail(path_, "__metadata__ value for " + m.key() + " is not a string");
                    }
                    metadata_.emplace(m.key(), m.value().get<std::string>());
                }
                continue;
            }
            const nlohmann::json& e = it.value();
            if (!e.is_object() || !e.contains("dtype") || !e.contains("shape") ||
                !e.contains("data_offsets") || !e["dtype"].is_string() || !e["shape"].is_array() ||
                !e["data_offsets"].is_array() || e["data_offsets"].size() != 2) {
                fail(path_, "malformed entry for tensor " + it.key());
            }
            SafetensorsTensor t;
            t.name  = it.key();
            t.dtype = e["dtype"].get<std::string>();
            for (const nlohmann::json& d : e["shape"]) {
                if (!d.is_number_unsigned()) {
                    fail(path_, "tensor " + t.name + " has a non-integer or negative dimension");
                }
                t.shape.push_back(static_cast<std::int64_t>(d.get<std::uint64_t>()));
            }
            const nlohmann::json& offs = e["data_offsets"];
            if (!offs[0].is_number_unsigned() || !offs[1].is_number_unsigned()) {
                fail(path_, "tensor " + t.name + " has invalid data_offsets");
            }
            const auto begin = offs[0].get<std::uint64_t>();
            const auto end   = offs[1].get<std::uint64_t>();
            if (begin > end || end > data_size) {
                fail(path_, "tensor " + t.name + " data_offsets [" + std::to_string(begin) + ", " +
                                std::to_string(end) + ") outside the data region of " +
                                std::to_string(data_size) + " bytes");
            }
            t.offset = data_offset_ + begin;
            t.bytes  = end - begin;
            std::size_t elem = 0;
            try {
                elem = safetensors_dtype_size(t.dtype);
            } catch (const std::invalid_argument&) {
                elem = 0; // dtype without a whole-byte element: no size check
            }
            if (elem != 0) {
                std::uint64_t expect = elem;
                for (std::int64_t d : t.shape) {
                    expect = checked_mul(expect, static_cast<std::uint64_t>(d), path_);
                }
                if (expect != t.bytes) {
                    fail(path_, "tensor " + t.name + " holds " + std::to_string(t.bytes) +
                                    " bytes, its dtype and shape need " + std::to_string(expect));
                }
            }
            tensors_.push_back(std::move(t));
        }
        std::sort(tensors_.begin(), tensors_.end(),
                  [](const SafetensorsTensor& a, const SafetensorsTensor& b) {
                      return a.offset != b.offset ? a.offset < b.offset : a.name < b.name;
                  });
        index_.reserve(tensors_.size());
        for (std::size_t i = 0; i < tensors_.size(); ++i) { index_.emplace(tensors_[i].name, i); }
    } catch (...) {
        ::close(fd_);
        fd_ = -1;
        throw;
    }
}

SafetensorsFile::~SafetensorsFile() {
    if (fd_ >= 0) { ::close(fd_); }
}

SafetensorsFile::SafetensorsFile(SafetensorsFile&& other) noexcept
    : path_(std::move(other.path_)), fd_(std::exchange(other.fd_, -1)),
      file_size_(other.file_size_), data_offset_(other.data_offset_),
      metadata_(std::move(other.metadata_)), tensors_(std::move(other.tensors_)),
      index_(std::move(other.index_)) {}

SafetensorsFile& SafetensorsFile::operator=(SafetensorsFile&& other) noexcept {
    if (this != &other) {
        if (fd_ >= 0) { ::close(fd_); }
        path_        = std::move(other.path_);
        fd_          = std::exchange(other.fd_, -1);
        file_size_   = other.file_size_;
        data_offset_ = other.data_offset_;
        metadata_    = std::move(other.metadata_);
        tensors_     = std::move(other.tensors_);
        index_       = std::move(other.index_);
    }
    return *this;
}

std::vector<std::string> SafetensorsFile::names() const {
    std::vector<std::string> out;
    out.reserve(tensors_.size());
    for (const SafetensorsTensor& t : tensors_) { out.push_back(t.name); }
    return out;
}

const SafetensorsTensor* SafetensorsFile::find(std::string_view name) const {
    const auto it = index_.find(name);
    return it == index_.end() ? nullptr : &tensors_[it->second];
}

const SafetensorsTensor& SafetensorsFile::at(std::string_view name) const {
    const SafetensorsTensor* t = find(name);
    if (t == nullptr) { fail(path_, "no tensor named " + std::string(name)); }
    return *t;
}

void SafetensorsFile::read_range(std::uint64_t offset, std::uint64_t bytes, void* dst) const {
    if (offset > file_size_ || bytes > file_size_ - offset) {
        fail(path_, "read of " + std::to_string(bytes) + " bytes at " + std::to_string(offset) +
                        " exceeds the file");
    }
    auto* out = static_cast<char*>(dst);
    while (bytes > 0) {
        const std::size_t want =
            static_cast<std::size_t>(std::min<std::uint64_t>(bytes, std::uint64_t(1) << 30));
        const ssize_t got = ::pread(fd_, out, want, static_cast<off_t>(offset));
        if (got < 0) {
            if (errno == EINTR) { continue; }
            fail_errno(path_, "pread failed");
        }
        if (got == 0) { fail(path_, "unexpected end of file"); }
        out += got;
        offset += static_cast<std::uint64_t>(got);
        bytes -= static_cast<std::uint64_t>(got);
    }
}

void SafetensorsFile::read(const SafetensorsTensor& tensor, void* dst) const {
    read_range(tensor.offset, tensor.bytes, dst);
}

std::vector<std::byte> SafetensorsFile::read(std::string_view name) const {
    const SafetensorsTensor& t = at(name);
    std::vector<std::byte> out(t.bytes);
    read(t, out.data());
    return out;
}

SafetensorsModel::SafetensorsModel(const std::filesystem::path& path) {
    namespace fs = std::filesystem;
    fs::path index_path;
    fs::path single;
    if (fs::is_directory(path)) {
        if (fs::exists(path / "model.safetensors.index.json")) {
            index_path = path / "model.safetensors.index.json";
        } else if (fs::exists(path / "model.safetensors")) {
            single = path / "model.safetensors";
        } else {
            throw std::runtime_error("safetensors model " + path.string() +
                                     ": neither model.safetensors.index.json nor model.safetensors");
        }
    } else if (path.extension() == ".json") {
        index_path = path;
    } else {
        single = path;
    }

    if (!single.empty()) {
        files_.push_back(std::make_unique<SafetensorsFile>(single));
        const SafetensorsFile& f = *files_.back();
        for (const SafetensorsTensor& t : f.tensors()) { index_.emplace(t.name, SafetensorsRef{&f, &t}); }
        return;
    }

    std::ifstream in(index_path);
    if (!in) { throw std::runtime_error("cannot open " + index_path.string()); }
    nlohmann::json j;
    try {
        j = nlohmann::json::parse(in);
    } catch (const nlohmann::json::exception& e) {
        throw std::runtime_error(index_path.string() + ": invalid JSON: " + e.what());
    }
    if (!j.is_object() || !j.contains("weight_map") || !j["weight_map"].is_object()) {
        throw std::runtime_error(index_path.string() + ": no weight_map object");
    }
    const fs::path dir = index_path.parent_path();
    std::map<std::string, const SafetensorsFile*, std::less<>> by_name;
    for (auto it = j["weight_map"].begin(); it != j["weight_map"].end(); ++it) {
        if (!it.value().is_string()) {
            throw std::runtime_error(index_path.string() + ": weight_map entry " + it.key() +
                                     " is not a file name");
        }
        const std::string file_name = it.value().get<std::string>();
        auto f                      = by_name.find(file_name);
        if (f == by_name.end()) {
            files_.push_back(std::make_unique<SafetensorsFile>(dir / file_name));
            f = by_name.emplace(file_name, files_.back().get()).first;
        }
        const SafetensorsTensor* t = f->second->find(it.key());
        if (t == nullptr) {
            throw std::runtime_error(index_path.string() + ": " + it.key() + " is not in " +
                                     file_name);
        }
        index_.emplace(it.key(), SafetensorsRef{f->second, t});
    }
}

std::vector<std::string> SafetensorsModel::names() const {
    std::vector<std::string> out;
    out.reserve(index_.size());
    for (const auto& [name, ref] : index_) { out.push_back(name); }
    return out;
}

SafetensorsRef SafetensorsModel::find(std::string_view name) const {
    const auto it = index_.find(name);
    return it == index_.end() ? SafetensorsRef{} : it->second;
}

SafetensorsRef SafetensorsModel::at(std::string_view name) const {
    const SafetensorsRef ref = find(name);
    if (!ref) { throw std::runtime_error("safetensors model: no tensor named " + std::string(name)); }
    return ref;
}

std::vector<std::byte> SafetensorsModel::read(std::string_view name) const {
    const SafetensorsRef ref = at(name);
    std::vector<std::byte> out(ref->bytes);
    ref.file->read(*ref.tensor, out.data());
    return out;
}

UploadRequest upload_request(const SafetensorsFile& file, const SafetensorsTensor& tensor,
                             void* dst) {
    return UploadRequest{&file, tensor.offset, tensor.bytes, dst};
}

UploadRequest upload_request(SafetensorsRef ref, void* dst) {
    if (!ref) { throw std::invalid_argument("upload_request: empty tensor reference"); }
    return upload_request(*ref.file, *ref.tensor, dst);
}

DeviceUploader::DeviceUploader(UploadOptions options) : options_(options) {
    if (options_.chunk_bytes == 0 || options_.staging_buffers <= 0 || options_.reader_threads <= 0) {
        throw std::invalid_argument("DeviceUploader: chunk_bytes, staging_buffers and "
                                    "reader_threads must be positive");
    }
    cuda_throw(cudaGetDevice(&device_), "cudaGetDevice");
    const std::size_t n = static_cast<std::size_t>(options_.staging_buffers);
    if (options_.chunk_bytes > std::numeric_limits<std::size_t>::max() / n) {
        throw std::invalid_argument("DeviceUploader: staging size overflows");
    }
    cuda_throw(cudaHostAlloc(&pinned_, options_.chunk_bytes * n, cudaHostAllocPortable),
               "DeviceUploader: cudaHostAlloc of the staging buffers");
    slots_.resize(n);
    try {
        for (std::size_t i = 0; i < n; ++i) {
            slots_[i].host = static_cast<std::byte*>(pinned_) + i * options_.chunk_bytes;
            cuda_throw(cudaEventCreateWithFlags(&slots_[i].done, cudaEventDisableTiming),
                       "DeviceUploader: cudaEventCreateWithFlags");
            free_slots_.push_back(static_cast<int>(i));
        }
        workers_.reserve(static_cast<std::size_t>(options_.reader_threads));
        for (int i = 0; i < options_.reader_threads; ++i) {
            workers_.emplace_back([this] { worker_main(); });
        }
    } catch (...) {
        {
            std::lock_guard lock(mutex_);
            stopping_ = true;
        }
        work_cv_.notify_all();
        for (std::thread& w : workers_) { w.join(); }
        for (Slot& s : slots_) {
            if (s.done != nullptr) { (void)cudaEventDestroy(s.done); }
        }
        (void)cudaFreeHost(pinned_);
        throw;
    }
}

DeviceUploader::~DeviceUploader() {
    {
        std::lock_guard lock(mutex_);
        stopping_ = true;
    }
    work_cv_.notify_all();
    for (std::thread& w : workers_) { w.join(); }
    for (Slot& s : slots_) {
        if (s.copy_inflight) { (void)cudaEventSynchronize(s.done); }
        (void)cudaEventDestroy(s.done);
    }
    (void)cudaFreeHost(pinned_);
}

UploadStats DeviceUploader::upload(std::span<const UploadRequest> requests, cudaStream_t stream) {
    nvtx::ScopedRange range(nvtx::Name::WeightUpload, nvtx::Category::Load);
    const auto start = std::chrono::steady_clock::now();
    int current      = 0;
    cuda_throw(cudaGetDevice(&current), "cudaGetDevice");
    if (current != device_) {
        throw std::logic_error("DeviceUploader::upload: current device " + std::to_string(current) +
                               " differs from the construction device " + std::to_string(device_));
    }

    std::vector<Chunk> chunks;
    UploadStats stats;
    for (const UploadRequest& r : requests) {
        if (r.bytes == 0) { continue; }
        if (r.file == nullptr || r.dst == nullptr) {
            throw std::invalid_argument("DeviceUploader::upload: request without file or destination");
        }
        if (r.offset > r.file->file_size() || r.bytes > r.file->file_size() - r.offset) {
            throw std::out_of_range("DeviceUploader::upload: range exceeds " + r.file->path().string());
        }
        for (std::uint64_t done = 0; done < r.bytes;) {
            const std::uint64_t n = std::min<std::uint64_t>(r.bytes - done, options_.chunk_bytes);
            chunks.push_back(Chunk{r.file, r.offset + done, n, static_cast<std::byte*>(r.dst) + done});
            done += n;
        }
        stats.bytes += r.bytes;
    }
    if (chunks.empty()) { return stats; }

    {
        std::unique_lock lock(mutex_);
        chunks_         = std::move(chunks);
        next_chunk_     = 0;
        stream_         = stream;
        error_          = nullptr;
        active_workers_ = static_cast<int>(workers_.size());
        ++generation_;
        work_cv_.notify_all();
        done_cv_.wait(lock, [this] { return active_workers_ == 0; });
        chunks_.clear();
    }

    // Every chunk's copy has been issued (or the job failed); wait until the data has landed so the
    // staging buffers and the caller's destinations are settled.
    cudaError_t sync_err = cudaSuccess;
    for (Slot& s : slots_) {
        if (s.copy_inflight) {
            const cudaError_t err = cudaEventSynchronize(s.done);
            if (sync_err == cudaSuccess) { sync_err = err; }
            s.copy_inflight = false;
        }
    }
    if (error_) { std::rethrow_exception(std::exchange(error_, nullptr)); }
    cuda_throw(sync_err, "DeviceUploader: waiting for the copies");
    stats.seconds =
        std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
    return stats;
}

void DeviceUploader::worker_main() {
    std::uint64_t seen = 0;
    for (;;) {
        {
            std::unique_lock lock(mutex_);
            work_cv_.wait(lock, [&] { return stopping_ || generation_ != seen; });
            if (stopping_) { return; }
            seen = generation_;
        }
        try {
            cuda_throw(cudaSetDevice(device_), "DeviceUploader: cudaSetDevice");
            run_chunks();
        } catch (...) {
            std::lock_guard lock(mutex_);
            if (!error_) { error_ = std::current_exception(); }
        }
        {
            std::lock_guard lock(mutex_);
            if (--active_workers_ == 0) { done_cv_.notify_all(); }
        }
    }
}

void DeviceUploader::run_chunks() {
    for (;;) {
        Chunk chunk{};
        {
            std::lock_guard lock(mutex_);
            if (error_ || next_chunk_ >= chunks_.size()) { return; }
            chunk = chunks_[next_chunk_++];
        }
        const int slot_index = acquire_slot();
        Slot& slot           = slots_[static_cast<std::size_t>(slot_index)];
        try {
            if (slot.copy_inflight) {
                cuda_throw(cudaEventSynchronize(slot.done), "DeviceUploader: staging buffer wait");
                slot.copy_inflight = false;
            }
            {
                nvtx::ScopedRange read_range(nvtx::Name::WeightRead, nvtx::Category::Load, chunk.bytes);
                chunk.file->read_range(chunk.offset, chunk.bytes, slot.host);
            }
            {
                std::lock_guard lock(stream_mutex_);
                cuda_throw(cudaMemcpyAsync(chunk.dst, slot.host, chunk.bytes, cudaMemcpyHostToDevice,
                                           stream_),
                           "DeviceUploader: cudaMemcpyAsync");
                slot.copy_inflight = true;
                cuda_throw(cudaEventRecord(slot.done, stream_), "DeviceUploader: cudaEventRecord");
            }
        } catch (...) {
            if (slot.copy_inflight) {
                // The copy may have been issued without its event: wait for the whole stream.
                (void)cudaStreamSynchronize(stream_);
                slot.copy_inflight = false;
            }
            release_slot(slot_index);
            throw;
        }
        release_slot(slot_index);
    }
}

int DeviceUploader::acquire_slot() {
    std::unique_lock lock(mutex_);
    slot_cv_.wait(lock, [this] { return !free_slots_.empty(); });
    const int slot = free_slots_.front();
    free_slots_.pop_front();
    return slot;
}

void DeviceUploader::release_slot(int slot) {
    {
        std::lock_guard lock(mutex_);
        free_slots_.push_back(slot);
    }
    slot_cv_.notify_one();
}

} // namespace ingine
