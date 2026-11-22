// DSpark drafter for Inkling-2-Mini (mtp.safetensors): C++ port of engine/ingine/dspark.py Drafter. Any draft
// distribution is valid for the speculative loop (greedy drafts are checked by equality, sampled ones accepted
// against the q they were drawn from), so the drafter's own weights and arithmetic are approximated where it
// buys time; only acceptance and draft time matter.
//
// * Target taps per processed row (bf16 [n, 4, H]: the partial block entering visits 5, 23, 41 and the final
//   normed hidden) become draft context through commit(): per draft layer l, c = mix_rmsnorm(feats,
//   alpha[l]), k = rope(rmsnorm_128(c Wk_ctx)), v = c Wv_ctx. Layers 0 and 2 keep the last WINDOW = 1024
//   positions in a BF16 ring (plus 8 rows for the block being drafted); layer 1 a ring of ctx = min(max_ctx,
//   DRAFT_CTX) slots (slot = position % ctx) with E4M3 keys and values: over a long context its bytes bound
//   the draft.
// * draft() runs the 8-row block (anchor embedding + 7 mask rows) through 3 bidirectional layers with
//   FP8-row projections (q/k/v, o, gate/up, down: the dense GEMV kernels) and tensor-core block attention
//   (kernels/draft/attention.cu), then the
//   Markov head over each step's top-1024 base-logit candidates. The base logits are scored over a shortlist
//   head instead of the whole unembedding: the SHORTLIST_ROWS most frequent tokens (kernels/draft/shortlist.h),
//   the special tokens, and up to CONTEXT_ROWS tokens of the current context (note_tokens), so drafts can only
//   propose those (measured on captured agent traces: +7% tok/s, tokens/verify not lower).
// * CUDA graphs: one draft graph per context bucket (1K, 2K, ... < ctx, then ctx; the global layer attends
//   to the first `bucket` slots) and one commit graph per row count 1..8.
#pragma once

#include "kernels/dense/dense.h"
#include "kernels/draft/draft.h"
#include "kernels/draft/gemm.h"
#include "model/model_data.h"

#include "core/arena.h"
#include "core/tensor.h"

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <array>
#include <cstdint>
#include <filesystem>
#include <map>
#include <span>
#include <vector>

namespace ingine::model {

class Drafter {
 public:
  static constexpr int BLOCK = kernels::draft::BLOCK;  // 8 rows per drafted block
  static constexpr int GAMMA = kernels::draft::GAMMA;  // 7 drafts per round
  static constexpr int DRAFT_CTX = 8192;               // global-layer ring slots (at most, 16 GB VRAM budget)
  static constexpr int SHORTLIST_ROWS = 16384;         // ranked tokens in the shortlist head
  static constexpr int CONTEXT_ROWS = 4096;            // context tokens added to it (until reset())

  // Loads mtp.safetensors from model_dir; `w` supplies the embedding, its norm and the unembedding (the
  // base logits use w.unembed as stored: FP8 rows with dense_fp8). `w` must outlive the drafter.
  // shortlist_rows: ranked rows of the shortlist head (tests use fewer to leave tokens of a small vocabulary out).
  Drafter(const std::filesystem::path& model_dir, const Weights& w, int max_ctx, cudaStream_t stream,
          int shortlist_rows = SHORTLIST_ROWS);
  ~Drafter();
  Drafter(const Drafter&) = delete;
  Drafter& operator=(const Drafter&) = delete;

  // Captures the draft graphs (per bucket) and the small-commit graph, then reset()s.
  void capture();
  // Empty context (local rings invalidated, length 0, no context tokens in the shortlist). Stream-ordered.
  void reset();
  // Positions committed so far (start + n of the last commit).
  int length() const { return length_; }
  int ctx() const { return ctx_; }

  // Committed target rows (feats bf16 [n, 4, H] on the device, positions start..start+n-1) become draft
  // context. n of 1..8 replays the small-commit graph (feats must stay valid until it ran); larger n
  // (prefill chunks) runs eagerly in pieces of <= 1024.
  void commit(const Tensor& feats, int start);
  void commit(const __nv_bfloat16* feats, int n, int start);

  // Drafts for positions P+1..P+7 after the anchor token at P. Greedy at temperature <= 0, otherwise
  // Gumbel-max at the temperature. Outputs are fixed device buffers, overwritten by the next draft.
  struct Draft {
    const std::int32_t* tokens;  // int32 [7]
    const float* logits;         // fp32 [7, V]: softcapped corrected logits, -inf outside the candidates
    const float* conf;           // fp32 [7]: max softmax probability per row (softmax(logits).amax(-1))
    const float* qstats;         // fp32 [7, 2]: sampled drafts' q statistics (kernels::draft::markov_finish)
  };
  Draft draft(int anchor, int P, float temperature);

  // Seed of the Gumbel noise of sampled drafts (resets the draw counter). Stream-ordered.
  void set_seed(std::uint64_t seed);

  // Tokens of the context (prompt and generated): those outside the shortlist join it (up to CONTEXT_ROWS
  // until reset()), so the drafter can propose them. Stream-ordered; call before the drafts that should see them.
  void note_tokens(std::span<const std::int32_t> ids);

  // A speculative round's transition, entirely from device values (no host synchronisation), for a verify
  // call at the last draft's position `start`: commits the accepted rows (n = *in.accepted + 1 rows of the
  // taps of width *in.m) and drafts for the next anchor at start + n (kernels::draft::round_begin).
  // bind_round() fixes the inputs and captures one graph per context bucket (after capture()); advance()
  // replays the one for `start`. length() is unknown until advanced(n) reports n.
  void bind_round(const kernels::draft::RoundInputs& in);
  Draft advance(int start);
  void advanced(int n) { length_ = pending_start_ + n; }

  // Prefix-reuse checkpoints of the draft context at the current length L (DecodeLoop's prompt checkpoints):
  // the local rings and their positions, f_rnn, L, and the global-ring slots [0, min(L, ctx)) (the global
  // ring has no position tags: a draft at P reads slots below min(P + 8, ctx) as positions == slot mod ctx).
  // checkpoint_bytes() is the fixed size of any checkpoint (allocate once); save_checkpoint copies what L
  // needs (stream-ordered, after the last commit). restore_checkpoint(src) of the last save into src brings
  // the drafter back to exactly that state (draft()/commit() bit-identical, length() == L, the same context
  // tokens in the shortlist); the global slots are copied back only if a commit or draft since the save wrote
  // one of them.
  std::size_t checkpoint_bytes() const;
  void save_checkpoint(void* dst);
  void restore_checkpoint(const void* src);

  // Conversation parking: the bytes of a checkpoint of length L that hold data, as {offset, bytes} regions of
  // its checkpoint_bytes() (the rest is unused), and the context tokens of the last save (shortlist rows, in
  // row order). adopt_checkpoint(src, L, context) declares that `src` holds a checkpoint of length L with
  // these context tokens, put there by the caller (its regions copied back from a parked copy of an earlier
  // save), so the next restore_checkpoint(src) restores all of it, context rows in the same order.
  struct Region {
    std::size_t offset, bytes;
  };
  std::array<Region, 3> checkpoint_regions(int length) const;
  const std::vector<std::int32_t>& checkpoint_context() const { return checkpoint_extra_; }
  void adopt_checkpoint(const void* src, int length, std::span<const std::int32_t> context);

  int vocab() const { return V_; }

  // Device scratch of the acceptance step (engine._accept), owned here so it is allocated once.
  struct AcceptScratch {
    std::int32_t* target;                    // int32 [8]: argmax of the verify rows
    float* probs;                            // fp32 [8, V]: engine._probs of the verify rows
    float* resid;                            // fp32 [V]
    float* uniforms;                         // fp32 [7]
    float* ws;                               // fp32 [sampled_accept_workspace_floats(V)]
    kernels::draft::AcceptResult* result;    // device
  };
  const AcceptScratch& accept_scratch() const { return accept_; }
  cudaStream_t stream() const { return stream_; }

 private:
  struct DraftLayer {
    const __nv_bfloat16 *in_norm, *post_norm, *ctx_norm, *q_norm, *k_norm, *k_norm_ctx;
    kernels::dense::Matrix qkv, gate_up;  // FP8 rows + scales
    kernels::dense::Matrix wo, w_down;    // FP8 rows + scales: [H, 3072], [H, inter]
    const __nv_bfloat16* wkv_ctx;         // BF16 [2048, H]: k_proj_ctx | v_proj_ctx (large commits)
    void *k_ctx, *v_ctx;                  // [slots, 8, 128]: BF16 (local rings), BF16 or E4M3 (the global ring)
    std::int32_t* ctx_pos;                // local layers: int32 [WINDOW]; null for the global layer
  };

  void* alloc(std::size_t bytes);
  void load(const std::filesystem::path& model_dir);
  void run_draft(int bucket);
  void run_layers(int bucket);                                                  // a draft after its input rows
  void run_commit(const __nv_bfloat16* feats, int n, int row0, bool last_piece);  // large commits
  void run_small_commit();                                                      // from params_
  void run_advance(int bucket);                                                 // round transition
  int bucket_for(int P) const;

  const Weights& w_;
  cudaStream_t stream_;
  int ctx_ = 0;
  int V_ = 0;
  int inter_ = 0;
  int length_ = 0;
  int pending_start_ = 0;  // start of the last advance()
  // the last checkpoint: where, its length, and whether the global slots it holds were overwritten since
  const void* checkpoint_ = nullptr;
  int checkpoint_length_ = 0;
  bool checkpoint_clobbered_ = false;
  void global_written(int begin, int end);  // positions [begin, end) written into the global ring
  std::vector<DeviceBuffer> storage_;
  kernels::draft::Bf16Gemm gemm_;

  std::array<DraftLayer, 3> layers_{};
  const __nv_bfloat16* alpha_ = nullptr;  // bf16 [4, 4] softmax(fuse_weight)
  const __nv_bfloat16 *final_norm_ = nullptr, *mask_embed_ = nullptr, *f_norm_ = nullptr;
  const __nv_bfloat16 *w1_ = nullptr, *rnn_w_ = nullptr, *rnn_b_ = nullptr;
  __nv_bfloat16 *fold_w_ = nullptr, *fold_b_ = nullptr;
  kernels::dense::Matrix head_{};         // the unembedding
  // the shortlist head: rows [0, short_K_) = ranked tokens and specials (ascending ids), then context tokens
  kernels::dense::Matrix short_head_{};   // [short_K_ + CONTEXT_ROWS, H] like head_
  std::int32_t* short_map_ = nullptr;     // row -> token id
  std::int32_t* short_ids_ = nullptr;     // staging of ids to gather
  int short_K_ = 0, short_E_ = 0;         // ranked rows, context rows in use
  std::vector<std::uint8_t> short_in_;    // host: token has a row
  std::vector<std::int32_t> short_extra_; // host: the context tokens, in row order
  std::vector<std::int32_t> checkpoint_extra_;  // the context tokens of the checkpoint
  void set_context_rows(int rows);        // drops context rows past `rows` (stream-ordered)
  kernels::dense::Matrix wkv_all_{};      // FP8 rows [3 * 2048, H]: the three layers' k/v context projections
  kernels::draft::CtxLayers ctx_layers_{};

  // per-call state and scratch
  kernels::draft::DraftParams* params_ = nullptr;
  __nv_bfloat16 *res_ = nullptr, *xn_ = nullptr, *q_ = nullptr, *attn_o_ = nullptr, *act_ = nullptr,
                *f_rnn_ = nullptr, *state_ = nullptr, *latent_ = nullptr;
  float *qkv_out_ = nullptr, *gu_out_ = nullptr, *delta_ = nullptr, *base_ = nullptr, *attn_ws_ = nullptr;
  float *cand_val_ = nullptr, *cand_logits_ = nullptr, *part_ = nullptr;
  void* topk_ws_ = nullptr;
  std::int32_t *cand_idx_ = nullptr, *prev_cand_ = nullptr;
  unsigned long long* picks_ = nullptr;
  float *draft_logits_ = nullptr, *conf_ = nullptr, *qstats_ = nullptr;
  std::int32_t* tokens_ = nullptr;
  int *seg_qkv_ = nullptr, *seg_gate_up_ = nullptr, *seg_short_ = nullptr, *seg_kv_ = nullptr;
  // commit scratch: large commits (cuBLASLt) and the small commit (dense proj over the three layers)
  __nv_bfloat16 *warm_feats_ = nullptr, *ctx_c_ = nullptr, *ctx_kv_ = nullptr, *commit_xin_ = nullptr;
  float* commit_kv_ = nullptr;

  AcceptScratch accept_{};

  std::map<int, cudaGraphExec_t> draft_graphs_;    // bucket -> graph, ascending
  cudaGraphExec_t commit_graph_ = nullptr;         // the small commit (1..8 rows, from params_)
  kernels::draft::RoundInputs round_{};            // bind_round()
  bool round_bound_ = false;
  std::map<int, cudaGraphExec_t> advance_graphs_;  // bucket -> round transition graph
};

}  // namespace ingine::model
