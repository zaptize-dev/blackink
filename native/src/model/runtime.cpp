// Model runtime: weights + ablation + state + decoder, decode graphs per call width (engine.Engine._capture),
// state snapshots and speculative rollback (engine.Engine._snapshot/_restore/_rollback), pipelined width-1
// steps and the switch graphs of device-selected verify widths.
#include "model/runtime.h"

#include "core/device.h"

#include <algorithm>
#include <array>
#include <stdexcept>

namespace ingine::model {
namespace {

constexpr int kMaxWidth = 8;
constexpr std::array<int, 3> kVerifyWidths = {2, 4, 8};

DeviceBuffer device_copy(const Tensor& t, cudaStream_t stream) {
  DeviceBuffer b(t.bytes());
  CUDA_CHECK(cudaMemcpyAsync(b.p, t.data, t.bytes(), cudaMemcpyDeviceToDevice, stream));
  return b;
}

// Carves consecutive 256-byte aligned pieces out of one allocation.
struct Carve {
  char* p;
  template <class T>
  T* take(std::size_t count) {
    T* r = reinterpret_cast<T*>(p);
    p += (count * sizeof(T) + 255) / 256 * 256;
    return r;
  }
};
constexpr std::size_t carved(std::size_t bytes) { return (bytes + 255) / 256 * 256; }

constexpr std::size_t kUndoBytes = carved(std::size_t(NUM_VISITS) * CONV_ROW_FLOATS * 4) +
                                   2 * carved(std::size_t(NUM_SLIDING_VISITS) * KV_DIM * 2) +
                                   carved(std::size_t(NUM_SLIDING_VISITS) * 8);
constexpr std::size_t kBackupBytes = 2 * carved(std::size_t(NUM_VISITS) * 3 * KV_DIM * 4) +
                                     2 * carved(std::size_t(NUM_VISITS) * 3 * HIDDEN * 4) +
                                     2 * carved(std::size_t(NUM_SLIDING_VISITS) * kMaxWidth * KV_DIM * 2) +
                                     carved(std::size_t(NUM_SLIDING_VISITS) * kMaxWidth * 8);

}  // namespace

GraphExec::~GraphExec() {
  if (exec_) cudaGraphExecDestroy(exec_);
}

GraphExec& GraphExec::operator=(GraphExec&& o) noexcept {
  if (this != &o) {
    if (exec_) cudaGraphExecDestroy(exec_);
    exec_ = o.exec_;
    o.exec_ = nullptr;
  }
  return *this;
}

void GraphExec::launch(cudaStream_t stream) const {
  if (!exec_) throw std::logic_error("GraphExec::launch: empty graph");
  CUDA_CHECK(cudaGraphLaunch(exec_, stream));
}

Runtime::Runtime(const RuntimeOptions& options) : options_(options) {
  if (options.max_ctx < 16) throw std::invalid_argument("max_ctx must be at least 16");
  CUDA_CHECK(cudaStreamCreateWithFlags(&stream_, cudaStreamNonBlocking));
  weights_ = load_weights(options.model_dir, options.dense_fp8, stream_, options.full_nvfp4);
  ablation_ = load_ablation(options.ablation);
  init();
}

Runtime::Runtime(const RuntimeOptions& options, Weights weights, Ablation ablation)
    : options_(options), weights_(std::move(weights)), ablation_(std::move(ablation)) {
  if (options.max_ctx < 16) throw std::invalid_argument("max_ctx must be at least 16");
  CUDA_CHECK(cudaStreamCreateWithFlags(&stream_, cudaStreamNonBlocking));
  init();
}

void Runtime::init() {
  const RuntimeOptions& options = options_;
  state_ = make_state(options.max_ctx, options.kv_fp8, options.kv_q4);
  decoder_ = std::make_unique<Decoder>(weights_, ablation_, state_, true);
  pos_ = DeviceBuffer(sizeof(std::int64_t));
  pos_.fill(0);
  token_ = DeviceBuffer(kMaxWidth * sizeof(std::int64_t));
  token_.fill(0);
  CUDA_CHECK(cudaStreamCreateWithFlags(&capture_stream_, cudaStreamNonBlocking));
  for (cudaEvent_t& e : chain_done_) CUDA_CHECK(cudaEventCreateWithFlags(&e, cudaEventDisableTiming));
  chain_host_ = MappedHostBuffer(2 * sizeof(std::int32_t));
  for (int i = 0; i < 2; ++i) {
    undo_storage_[i] = DeviceBuffer(kUndoBytes);
    Carve c{static_cast<char*>(undo_storage_[i].p)};
    undo_[i].conv = c.take<float>(std::size_t(NUM_VISITS) * CONV_ROW_FLOATS);
    undo_[i].k = c.take<__nv_bfloat16>(std::size_t(NUM_SLIDING_VISITS) * KV_DIM);
    undo_[i].v = c.take<__nv_bfloat16>(std::size_t(NUM_SLIDING_VISITS) * KV_DIM);
    undo_[i].pos = c.take<std::int64_t>(NUM_SLIDING_VISITS);
  }

  buffers_.emplace(1, make_buffers(1));
  history_[1] = false;
  if (options.spec) {
    for (int w : kVerifyWidths) {
      buffers_.emplace(w, make_buffers(w));
      history_[w] = true;
      verify_widths_.push_back(w);
    }
    backup_storage_ = DeviceBuffer(kBackupBytes);
    Carve c{static_cast<char*>(backup_storage_.p)};
    for (int i = 0; i < 4; ++i) backup_.conv[i] = c.take<float>(std::size_t(NUM_VISITS) * 3 * CONV_COLS[i]);
    backup_.k = c.take<__nv_bfloat16>(std::size_t(NUM_SLIDING_VISITS) * kMaxWidth * KV_DIM);
    backup_.v = c.take<__nv_bfloat16>(std::size_t(NUM_SLIDING_VISITS) * kMaxWidth * KV_DIM);
    backup_.pos = c.take<std::int64_t>(std::size_t(NUM_SLIDING_VISITS) * kMaxWidth);
  }
  // every width reads its tokens from the shared buffer (set_inputs, chain_advance and the verify selector
  // write one place)
  for (auto& [m, b] : buffers_) b.token = rm(token_.p, DType::I64, {m});
  for (const auto& [m, history] : history_) capture(m, history);
  CUDA_CHECK(cudaStreamSynchronize(stream_));
}

Runtime::~Runtime() {
  if (stream_) cudaStreamSynchronize(stream_);
  for (auto& [m, exec] : graphs_) cudaGraphExecDestroy(exec);
  graphs_.clear();
  decoder_.reset();
  for (cudaEvent_t e : chain_done_)
    if (e) cudaEventDestroy(e);
  if (capture_stream_) cudaStreamDestroy(capture_stream_);
  if (stream_) cudaStreamDestroy(stream_);
}

bool Runtime::has_width(int m) const { return buffers_.count(m) != 0; }

std::span<const int> Runtime::verify_widths() const { return verify_widths_; }

Buffers& Runtime::buffers(int m) {
  auto it = buffers_.find(m);
  if (it == buffers_.end()) throw std::invalid_argument("no decode buffers for width " + std::to_string(m));
  return it->second;
}

StatePtrs Runtime::state_ptrs() const {
  StatePtrs p;
  p.conv[0] = ptr<float>(state_.conv_k);
  p.conv[1] = ptr<float>(state_.conv_v);
  p.conv[2] = ptr<float>(state_.conv_attn);
  p.conv[3] = ptr<float>(state_.conv_mlp);
  p.k_ring = ptr<__nv_bfloat16>(state_.k_ring);
  p.v_ring = ptr<__nv_bfloat16>(state_.v_ring);
  p.ring_pos = ptr<std::int64_t>(state_.ring_pos);
  return p;
}

void Runtime::capture(int m, bool history) {
  // warm up so the pointer tables and kernel attributes are set outside capture; the scratch writes to the
  // conv states and ring positions are undone afterwards (the cache rows it wrote are never valid)
  const Snapshot saved = snapshot();
  Buffers& b = buffers(m);
  for (int i = 0; i < 2; ++i) decoder_->decode(b, pos(), history, stream_);
  CUDA_CHECK(cudaStreamSynchronize(stream_));
  cudaGraph_t graph = nullptr;
  CUDA_CHECK(cudaStreamBeginCapture(stream_, cudaStreamCaptureModeThreadLocal));
  try {
    decoder_->decode(b, pos(), history, stream_);
  } catch (...) {
    cudaStreamEndCapture(stream_, &graph);
    if (graph) cudaGraphDestroy(graph);
    throw;
  }
  CUDA_CHECK(cudaStreamEndCapture(stream_, &graph));
  cudaGraphExec_t exec = nullptr;
  const cudaError_t err = cudaGraphInstantiate(&exec, graph, 0);
  cudaGraphDestroy(graph);
  CUDA_CHECK(err);
  CUDA_CHECK(cudaGraphUpload(exec, stream_));
  graphs_[m] = exec;
  restore(saved);
  CUDA_CHECK(cudaStreamSynchronize(stream_));
}

void Runtime::set_inputs(std::span<const std::int32_t> tokens, int position) {
  const int m = int(tokens.size());
  buffers(m);  // the width must exist
  set_step_inputs(tokens.data(), m, position, this->tokens(), pos(), stream_);
}

const float* Runtime::run_step(int m) {
  auto it = graphs_.find(m);
  if (it == graphs_.end()) throw std::invalid_argument("no decode graph for width " + std::to_string(m));
  CUDA_CHECK(cudaGraphLaunch(it->second, stream_));
  return static_cast<const float*>(buffers(m).logits.data);
}

const float* Runtime::run_step_eager(int m) {
  Buffers& b = buffers(m);
  decoder_->decode(b, pos(), history_.at(m), stream_);
  return static_cast<const float*>(b.logits.data);
}

const float* Runtime::decode_logits(std::int32_t token, int position) {
  const std::int32_t t[1] = {token};
  set_inputs(t, position);
  return run_step(1);
}

void Runtime::chain_advance(const std::int32_t* sampled, int position) {
  const int k = position & 1;
  model::chain_advance(sampled, tokens(), pos(), position, static_cast<std::int32_t*>(chain_host_.device_ptr()) + k,
                       state_ptrs(), undo_[k], stream_);
  CUDA_CHECK(cudaEventRecord(chain_done_[k], stream_));
}

std::int32_t Runtime::chain_token(int position) {
  const int k = position & 1;
  CUDA_CHECK(cudaEventSynchronize(chain_done_[k]));
  return static_cast<const volatile std::int32_t*>(chain_host_.data())[k];
}

void Runtime::chain_undo(int position) { model::chain_undo(state_ptrs(), undo_[position & 1], position, stream_); }

Runtime::Snapshot Runtime::snapshot() {
  Snapshot s;
  for (const Tensor* t : {&state_.ring_pos, &state_.conv_k, &state_.conv_v, &state_.conv_attn, &state_.conv_mlp})
    s.tensors.push_back(device_copy(*t, stream_));
  return s;
}

void Runtime::restore(const Snapshot& snap) {
  const Tensor* ts[5] = {&state_.ring_pos, &state_.conv_k, &state_.conv_v, &state_.conv_attn, &state_.conv_mlp};
  for (int i = 0; i < 5; ++i)
    CUDA_CHECK(cudaMemcpyAsync(ts[i]->data, snap.tensors[i].p, ts[i]->bytes(), cudaMemcpyDeviceToDevice, stream_));
}

namespace {
// the state tensors a prompt checkpoint holds, in checkpoint order
std::array<const Tensor*, 7> checkpoint_tensors(const State& st) {
  return {&st.k_ring, &st.v_ring, &st.ring_pos, &st.conv_k, &st.conv_v, &st.conv_attn, &st.conv_mlp};
}
}  // namespace

std::size_t Runtime::checkpoint_bytes() const {
  std::size_t n = 0;
  for (const Tensor* t : checkpoint_tensors(state_)) n += carved(t->bytes());
  return n;
}

void Runtime::save_checkpoint(void* dst) {
  char* p = static_cast<char*>(dst);
  for (const Tensor* t : checkpoint_tensors(state_)) {
    CUDA_CHECK(cudaMemcpyAsync(p, t->data, t->bytes(), cudaMemcpyDefault, stream_));
    p += carved(t->bytes());
  }
}

void Runtime::restore_checkpoint(const void* src) {
  const char* p = static_cast<const char*>(src);
  for (const Tensor* t : checkpoint_tensors(state_)) {
    CUDA_CHECK(cudaMemcpyAsync(t->data, p, t->bytes(), cudaMemcpyDefault, stream_));
    p += carved(t->bytes());
  }
}

std::size_t Runtime::full_rows_bytes(int n) const {
  if (n < 0 || n > options_.max_ctx)
    throw std::invalid_argument("full_rows_bytes: n = " + std::to_string(n) + " outside [0, max_ctx = " +
                                std::to_string(options_.max_ctx) + "]");
  const std::size_t row = state_.k_full.bytes() / NUM_FULL_VISITS / std::size_t(options_.max_ctx);
  return 2 * std::size_t(NUM_FULL_VISITS) * row * std::size_t(n);
}

// one 2D copy per cache: NUM_FULL_VISITS rows of n positions, pitch max_ctx positions in the cache, packed in
// the parked copy (k visits, then v visits)
void Runtime::save_full_rows(void* dst, int n) {
  const std::size_t bytes = full_rows_bytes(n) / 2, width = bytes / NUM_FULL_VISITS;
  if (n == 0) return;
  const std::size_t pitch = state_.k_full.bytes() / NUM_FULL_VISITS;
  char* p = static_cast<char*>(dst);
  for (const Tensor* t : {&state_.k_full, &state_.v_full}) {
    CUDA_CHECK(cudaMemcpy2DAsync(p, width, t->data, pitch, width, NUM_FULL_VISITS, cudaMemcpyDefault, stream_));
    p += bytes;
  }
}

void Runtime::restore_full_rows(const void* src, int n) {
  const std::size_t bytes = full_rows_bytes(n) / 2, width = bytes / NUM_FULL_VISITS;
  if (n == 0) return;
  const std::size_t pitch = state_.k_full.bytes() / NUM_FULL_VISITS;
  const char* p = static_cast<const char*>(src);
  for (const Tensor* t : {&state_.k_full, &state_.v_full}) {
    CUDA_CHECK(cudaMemcpy2DAsync(t->data, pitch, p, width, width, NUM_FULL_VISITS, cudaMemcpyDefault, stream_));
    p += bytes;
  }
}

void Runtime::save_verify(int P, int m) {
  if (verify_widths_.empty()) throw std::logic_error("save_verify needs the speculative widths (options.spec)");
  verify_save(state_ptrs(), backup_, nullptr, P, m, stream_);
}

void Runtime::rollback(int P, int commit, int m) {
  if (commit == m) return;
  if (commit < 1 || commit > m) throw std::invalid_argument("rollback: commit out of range");
  const Buffers& b = buffers(m);
  const float* hist[4] = {ptr<float>(b.hist_k), ptr<float>(b.hist_v), ptr<float>(b.hist_attn), ptr<float>(b.hist_mlp)};
  verify_rollback(state_ptrs(), backup_, hist, nullptr, P, nullptr, commit, m, stream_);
}

void Runtime::record_step(int m, cudaStream_t stream) {
  if (!graphs_.count(m)) throw std::invalid_argument("record_step: no decode graph for width " + std::to_string(m));
  decoder_->decode(buffers(m), pos(), history_.at(m), stream);
}

void Runtime::record_save_verify(int m, cudaStream_t stream) {
  if (verify_widths_.empty()) throw std::logic_error("record_save_verify needs the speculative widths (options.spec)");
  verify_save(state_ptrs(), backup_, pos(), 0, m, stream);
}

void Runtime::record_rollback(int m, const std::int32_t* accepted, cudaStream_t stream) {
  if (!history_.count(m) || !history_.at(m)) throw std::invalid_argument("record_rollback: width without history");
  const Buffers& b = buffers(m);
  const float* hist[4] = {ptr<float>(b.hist_k), ptr<float>(b.hist_v), ptr<float>(b.hist_attn), ptr<float>(b.hist_mlp)};
  verify_rollback(state_ptrs(), backup_, hist, pos(), 0, accepted, 0, m, stream);
}

GraphExec Runtime::capture_switch(const std::function<void(cudaStream_t, cudaGraphConditionalHandle)>& head,
                                  const std::function<void(cudaStream_t, int)>& branch) {
  if (verify_widths_.empty()) throw std::logic_error("capture_switch needs the speculative widths (options.spec)");
  const cudaStream_t s = capture_stream_;
  cudaGraph_t graph = nullptr;
  CUDA_CHECK(cudaGraphCreate(&graph, 0));
  struct Owned {
    cudaGraph_t g;
    ~Owned() { cudaGraphDestroy(g); }
  } owned{graph};
  // run `body` captured into `g` (an exception ends the capture first)
  auto captured = [&](cudaGraph_t g, const std::function<void()>& body) {
    CUDA_CHECK(cudaStreamBeginCaptureToGraph(s, g, nullptr, nullptr, 0, cudaStreamCaptureModeThreadLocal));
    try {
      body();
    } catch (...) {
      cudaGraph_t out = nullptr;
      cudaStreamEndCapture(s, &out);
      throw;
    }
    cudaGraph_t out = nullptr;
    CUDA_CHECK(cudaStreamEndCapture(s, &out));
  };

  cudaGraphConditionalHandle handle;
  CUDA_CHECK(cudaGraphConditionalHandleCreate(&handle, graph, 0, 0));
  cudaGraph_t* bodies = nullptr;
  captured(graph, [&] {
    head(s, handle);
    cudaStreamCaptureStatus status;
    cudaGraph_t g = nullptr;
    const cudaGraphNode_t* deps = nullptr;
    const cudaGraphEdgeData* edges = nullptr;
    std::size_t n = 0;
    CUDA_CHECK(cudaStreamGetCaptureInfo(s, &status, nullptr, &g, &deps, &edges, &n));
    cudaGraphNodeParams p{};
    p.type = cudaGraphNodeTypeConditional;
    p.conditional.handle = handle;
    p.conditional.type = cudaGraphCondTypeSwitch;
    p.conditional.size = unsigned(verify_widths_.size());
    cudaGraphNode_t node = nullptr;
    CUDA_CHECK(cudaGraphAddNode(&node, g, deps, edges, n, &p));
    bodies = p.conditional.phGraph_out;
    CUDA_CHECK(cudaStreamUpdateCaptureDependencies(s, &node, nullptr, 1, cudaStreamSetCaptureDependencies));
  });
  for (std::size_t i = 0; i < verify_widths_.size(); ++i)
    captured(bodies[i], [&] { branch(s, verify_widths_[i]); });

  cudaGraphExec_t exec = nullptr;
  CUDA_CHECK(cudaGraphInstantiate(&exec, graph, 0));
  GraphExec result(exec);
  CUDA_CHECK(cudaGraphUpload(exec, stream_));
  return result;
}

}  // namespace ingine::model
