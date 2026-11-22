// The model runtime shared by the Generator, the Prefiller and the speculative loop: one CUDA stream, the
// weights and ablation, the request state, and the fused decode step (fast.FastDecoder.decode) per call
// width M in {1, 2, 4, 8} replayed as CUDA graphs (engine.Engine.steps / graphs), with the state
// accessors engine.Engine._snapshot/_restore/_rollback need.
//
// Decode inputs live in device buffers the graphs read: buffers(m).token (int64 [m], one int64 [8] buffer
// shared by every width) and pos() (int64 [1], the first row's position; row i is at pos + i). Outputs:
// buffers(m).logits (fp32 [m, UNPADDED_VOCAB], softcapped), buffers(m).taps (bf16 [m, 4, H]) and, for widths
// captured with history, buffers(m).hist_*. Width 1 runs without history; widths 2, 4 and 8 (created when
// spec is on) with history (engine.py).
//
// Pipelined width-1 decoding (chain_*): a step's input token is the previous step's sampled token, moved on
// the device, so the host launches step N+1 before it reads token N; a step launched past the end of the
// generation (stop token, cancel) is undone. Device-selected verify widths (capture_switch, record_*): the
// speculative loop picks the verify width on the device, so the verify call is launched before the draft
// it depends on has finished.
#pragma once

#include "model/model_data.h"
#include "model/model_kernels.h"

#include "core/host_memory.h"

#include "ingine/generator.h"

#include <cuda_runtime.h>

#include <array>
#include <cstdint>
#include <filesystem>
#include <functional>
#include <map>
#include <span>
#include <vector>

namespace ingine::model {

// ---- loading (weights.load + FastDecoder._concat_projections/_stack_layer_params/_fp8_dense,
// model.Ablation.load/none, model.State)
Weights load_weights(const std::filesystem::path& model_dir, bool dense_fp8, cudaStream_t stream,
                     bool full_nvfp4 = false);
// One layer as weights.load_layer (+ the decoder's derived tensors), for kernel tests; owns its memory.
Layer load_layer(const std::filesystem::path& model_dir, int l, bool dense_fp8, cudaStream_t stream,
                 std::vector<DeviceBuffer>& storage, bool full_nvfp4 = false);
Ablation load_ablation(const std::filesystem::path& path);  // empty path: Ablation.none()
State make_state(int max_ctx, bool kv_fp8, bool kv_q4 = false);
Buffers make_buffers(int m);

// model.State.reset: ring_pos = -1, conv states = 0, length = 0 (caches keep their contents).
void reset_state(State& state, cudaStream_t stream);

// ---- fast.FastDecoder
class Decoder {
 public:
  Decoder(const Weights& w, const Ablation& abl, State& st, bool pdl = true);
  ~Decoder();
  Decoder(const Decoder&) = delete;
  Decoder& operator=(const Decoder&) = delete;

  // fast.FastDecoder.decode: runs b.m consecutive tokens b.token at positions *pos, *pos + 1, ...;
  // logits in b.logits, taps in b.taps, conv inputs in b.hist_* when `history`. Graph capturable once it
  // ran eagerly for `b` (pointer tables are built on first use).
  void decode(Buffers& b, const std::int64_t* pos, bool history, cudaStream_t stream);

 private:
  const std::int64_t* block_table(const std::vector<const float*>& blocks);

  const Weights& w_;
  const Ablation& abl_;
  State& st_;
  bool pdl_;
  DeviceBuffer attn_ws_;  // attn workspace for up to 8 rows (zeroed once)
  std::map<std::vector<const float*>, DeviceBuffer> tables_;  // device pointer tables for mix
  std::array<DeviceBuffer, 2> segments_;  // proj segment tables: [single-input sliding, three-input]
  DeviceBuffer full_segments_;
};

// An instantiated CUDA graph (move-only).
class GraphExec {
 public:
  GraphExec() = default;
  explicit GraphExec(cudaGraphExec_t exec) : exec_(exec) {}
  ~GraphExec();
  GraphExec(GraphExec&& o) noexcept : exec_(o.exec_) { o.exec_ = nullptr; }
  GraphExec& operator=(GraphExec&& o) noexcept;
  GraphExec(const GraphExec&) = delete;
  GraphExec& operator=(const GraphExec&) = delete;

  void launch(cudaStream_t stream) const;
  explicit operator bool() const { return exec_ != nullptr; }

 private:
  cudaGraphExec_t exec_ = nullptr;
};

// ---- Runtime
class Runtime {
 public:
  // Loads the weights (options.model_dir, dense_fp8) and ablation (options.ablation).
  explicit Runtime(const RuntimeOptions& options);
  // Uses the given weights and ablation (tests, tools); options.model_dir and ablation are not read.
  Runtime(const RuntimeOptions& options, Weights weights, Ablation ablation);
  ~Runtime();
  Runtime(const Runtime&) = delete;
  Runtime& operator=(const Runtime&) = delete;

  const RuntimeOptions& options() const { return options_; }
  cudaStream_t stream() const { return stream_; }
  const Weights& weights() const { return weights_; }
  const Ablation& ablation() const { return ablation_; }
  State& state() { return state_; }
  const State& state() const { return state_; }

  // Call widths available: {1}, or {1, 2, 4, 8} with options.spec.
  bool has_width(int m) const;
  std::span<const int> verify_widths() const;  // {2, 4, 8} with spec, else empty
  Buffers& buffers(int m);
  std::int64_t* pos() { return static_cast<std::int64_t*>(pos_.p); }

  // Stream-ordered step inputs: buffers(tokens.size()).token = tokens, pos = position (a kernel; no copy).
  void set_inputs(std::span<const std::int32_t> tokens, int position);
  // Replays width m's graph; returns buffers(m).logits (device). Stream-ordered, no host sync.
  const float* run_step(int m);
  // set_inputs + run_step for one token (engine._decode_logits).
  const float* decode_logits(std::int32_t token, int position);
  // The same step without graphs (validation, tests).
  const float* run_step_eager(int m);

  // ---- pipelined width-1 decoding
  // After a sampler wrote the token of `position` to `sampled` (device int32 [1]): the width-1 inputs become
  // (*sampled, position), the token is published to the host, and what the step at `position` will overwrite
  // is recorded (StepUndo, by position parity). Stream-ordered; chain_token(position) waits for it.
  void chain_advance(const std::int32_t* sampled, int position);
  // Waits for chain_advance(position) and returns its token.
  std::int32_t chain_token(int position);
  // Undoes the width-1 step at `position`: it must be the last decode step run, launched after
  // chain_advance(position) and before any later chain_advance. Stream-ordered.
  void chain_undo(int position);

  // engine._snapshot/_restore over ring_pos + conv states (device copies, stream-ordered).
  struct Snapshot {
    std::vector<DeviceBuffer> tensors;
  };
  Snapshot snapshot();
  void restore(const Snapshot& snap);

  // Prompt checkpoints: everything a decode step or prefill at later positions overwrites (the sliding rings
  // with their positions and the conv states; the full-attention caches are position-addressed and need
  // only the length). save_checkpoint copies it to `dst` (device or pinned host memory, checkpoint_bytes()
  // long); restore_checkpoint puts it back, so the state equals the one at the save up to rows past its
  // length, which nothing reads before overwriting. Stream-ordered.
  std::size_t checkpoint_bytes() const;
  void save_checkpoint(void* dst);
  void restore_checkpoint(const void* src);

  // Conversation parking: a checkpoint holds everything but the full-attention caches; save_full_rows copies
  // their rows [0, n) (k and v of every full visit, positions 0..n-1) to `dst` (device or pinned host memory,
  // full_rows_bytes(n) long, packed), restore_full_rows puts them back. A checkpoint taken at length n plus
  // its full rows [0, n) restore the state exactly, whatever ran in between. Stream-ordered.
  std::size_t full_rows_bytes(int n) const;
  void save_full_rows(void* dst, int n);
  void restore_full_rows(const void* src, int n);

  // Speculative rollback (engine._generate_spec / _rollback). save_verify before a verify call of m rows at
  // P..P+m-1 saves the conv states and the ring slots (P + i) % WINDOW the call will overwrite;
  // rollback(P, commit, m) after it leaves the state as after its first `commit` rows (no-op if
  // commit == m). Uses buffers(m).hist_* (the call must run with history). Stream-ordered kernels.
  void save_verify(int P, int m);
  void rollback(int P, int commit, int m);

  // ---- device-selected verify widths (graph building; `stream` is a capturing stream)
  // A graph of `head` (captured; it must set the switch condition to an index into verify_widths()) followed
  // by a switch node whose branch i is `branch(verify_widths()[i])` (captured).
  GraphExec capture_switch(const std::function<void(cudaStream_t, cudaGraphConditionalHandle)>& head,
                           const std::function<void(cudaStream_t, int)>& branch);
  // Records width m's decode step, save_verify(*pos(), m) and rollback(*pos(), *accepted + 1, m) on `stream`
  // (positions and the accepted count read on the device).
  void record_step(int m, cudaStream_t stream);
  void record_save_verify(int m, cudaStream_t stream);
  void record_rollback(int m, const std::int32_t* accepted, cudaStream_t stream);
  std::int64_t* tokens() { return static_cast<std::int64_t*>(token_.p); }  // the shared int64 [8] input tokens

 private:
  void init();
  void capture(int m, bool history);
  StatePtrs state_ptrs() const;

  RuntimeOptions options_;
  cudaStream_t stream_ = nullptr;
  Weights weights_;
  Ablation ablation_;
  State state_;
  std::unique_ptr<Decoder> decoder_;
  DeviceBuffer pos_, token_;
  std::map<int, Buffers> buffers_;
  std::map<int, bool> history_;
  std::map<int, cudaGraphExec_t> graphs_;
  std::vector<int> verify_widths_;
  // rollback scratch (VerifyBackup) and the width-1 undo records (StepUndo by position parity)
  DeviceBuffer backup_storage_;
  VerifyBackup backup_{};
  std::array<DeviceBuffer, 2> undo_storage_;
  std::array<StepUndo, 2> undo_{};
  MappedHostBuffer chain_host_;  // int32 [2]: tokens published by chain_advance, by position parity
  std::array<cudaEvent_t, 2> chain_done_{};
  cudaStream_t capture_stream_ = nullptr;
};

// The Generator (make_generator) over given weights and ablation, for tests: plain decoding only
// (options.spec needs the drafter's files in model_dir).
std::unique_ptr<Generator> make_generator(const RuntimeOptions& options, Weights weights, Ablation ablation);

}  // namespace ingine::model
