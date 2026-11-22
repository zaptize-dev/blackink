// DSpark drafter kernels (engine/ingine/dspark.py) and the speculative acceptance helpers of
// engine.Engine._accept. Raw-pointer launchers on `stream`; they never allocate or synchronise and read
// every per-call value (anchor, position, temperature, commit rows, RNG draw) from device memory, so all
// of them are CUDA-graph capturable. Shapes are row-major (Python order). Rounding points follow dspark.py
// (each `bf(...)` there is a BF16 rounding here).
//
// The kernels of the commit and draft chains are launched as programmatic dependents (PDL, see
// kernels/common/pdl.cuh): each triggers its dependents first and waits for its predecessor before it reads
// anything a kernel wrote, so launch latency overlaps the predecessor's tail. The set_* parameter kernels
// are ordinary launches.
#pragma once

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>

namespace ingine::kernels::draft {

inline constexpr int H = 2304;
inline constexpr int HEADS = 24, KV_HEADS = 8, HEAD_DIM = 128;
inline constexpr int Q_DIM = HEADS * HEAD_DIM;     // 3072
inline constexpr int KV_DIM = KV_HEADS * HEAD_DIM; // 1024
inline constexpr int QKV_ROWS = Q_DIM + 2 * KV_DIM; // 5120
inline constexpr int BLOCK = 8;                     // drafted block: anchor + 7 masks
inline constexpr int GAMMA = BLOCK - 1;             // drafts per round
inline constexpr int RANK = 256;                    // Markov head state width
inline constexpr int RNN_IN = 2 * RANK + 2 * H;     // 5120: [state, W1[prev], h_i, f_rnn]
inline constexpr int RNN_OUT = 3 * RANK;            // 768: gate | cell | latent
inline constexpr int WINDOW = 1024;                 // local layers' context ring
inline constexpr int CANDIDATES = 1024;             // Markov-head vocabulary per step
inline constexpr float ROPE_THETA = 1e6f;
inline constexpr float EPS = 1e-6f;

// Per-call scalars, device resident (the graphs read them).
struct DraftParams {
  std::int32_t anchor = 0;      // token at position P (sampled, not yet processed)
  std::int32_t P = 0;           // position of the anchor
  float temperature = 0.f;      // <= 0: greedy drafts, else Gumbel-max at this temperature
  std::int32_t start = 0;       // first position of the rows being committed
  std::uint64_t seed = 0;       // Gumbel noise: gumbel(seed, draws, step, candidate) (rng.h)
  std::uint64_t draws = 0;      // drafts so far (advanced by markov_finish)
  std::int32_t ncommit = 0;     // rows of the small commit (1..BLOCK)
  std::int32_t round_start = 0; // P of the last draft (set_draft_inputs, markov_finish): where a round commits
  const __nv_bfloat16* commit_feats = nullptr;  // their taps, bf16 [ncommit, 4, H]
};
void set_draft_inputs(DraftParams* p, int anchor, int P, float temperature, cudaStream_t stream);
void set_commit_start(DraftParams* p, int start, cudaStream_t stream);
void set_seed(DraftParams* p, std::uint64_t seed, cudaStream_t stream);  // also resets draws
// A small commit of n (1..BLOCK) rows from feats at positions start..start+n-1.
void set_commit(DraftParams* p, const __nv_bfloat16* feats, int n, int start, cudaStream_t stream);

// The device-side transition of a speculative round (engine._generate_spec after the acceptance), for a
// verify call at the last draft's position: start = p->round_start. The call (width *m, rows
// start..start+*m-1) accepted *accepted drafts; its first n = *accepted + 1 rows (taps[i] for m == widths[i],
// bf16 [m, 4, H]) are committed at start, and the next draft is at P = start + n from the anchor
// *next_greedy (p->temperature <= 0) or *next_sampled. The temperature stays. See round_begin.
struct RoundInputs {
  const __nv_bfloat16* taps[3] = {};
  int widths[3] = {};
  int nwidths = 0;
  const std::int32_t* m = nullptr;
  const std::int32_t* accepted = nullptr;
  const std::int32_t* next_greedy = nullptr;
  const std::int32_t* next_sampled = nullptr;
};

// ---- context (commit) --------------------------------------------------------------------------------

// mix_rmsnorm: out[r] = bf16(rms(sum_j float(feats[r, j]) * float(alpha[j])) * w), feats bf16 [n, 4, H],
// alpha bf16 [4], w bf16 [H], out bf16 [n, H].
void mix_rmsnorm(const __nv_bfloat16* feats, int n, const __nv_bfloat16* alpha, const __nv_bfloat16* w,
                 __nv_bfloat16* out, cudaStream_t stream);

// Cache rows (k_cache / v_cache, [slots, 8, 128]) are BF16, or float8_e4m3fn without a scale (`fp8`: the
// BF16 value rounded to nearest even, saturated at +-448; keys are RMS-normed and values a normed input's
// projection, both well inside the range). The local rings are BF16; the global ring may be either.

// Commit of context keys/values: kv bf16 [n, 2048] = (c Wk_ctx^T | c Wv_ctx^T). Row r is at position
// p->start + row0 + r; rows r >= keep_from are stored at slot position % size:
// k = bf16(rope(bf16(rms(k_r per head) * k_norm))), v as is. ctx_pos (local rings, int32 [size]) gets the
// position, or is null (global ring).
void ctx_kv_store(const __nv_bfloat16* kv, int n, const DraftParams* p, int row0, int keep_from,
                  const __nv_bfloat16* k_norm, void* k_cache, void* v_cache, bool fp8, int size,
                  std::int32_t* ctx_pos, cudaStream_t stream);

// The three draft layers' context state, for the small (device-driven) commit.
struct CtxLayers {
  const __nv_bfloat16* ctx_norm[3] = {};    // bf16 [H]
  const __nv_bfloat16* k_norm_ctx[3] = {};  // bf16 [128]
  void* k_cache[3] = {};                    // [size, 8, 128], BF16 or E4M3 (fp8)
  void* v_cache[3] = {};
  bool fp8[3] = {};
  std::int32_t* ctx_pos[3] = {};            // int32 [size] (local rings) or null (global ring)
  int size[3] = {};                         // ring slots (>= BLOCK)
};

// Small commit, part 1 (n = p->ncommit rows of p->commit_feats): xin[l, r] = mix_rmsnorm(feats[r],
// alpha[l], ctx_norm[l]) for r < n (zero rows above), xin bf16 [3, BLOCK, H] (the dense proj input of the
// three layers' k/v projections); f_rnn = mix_rmsnorm(feats[n - 1], alpha[3], f_norm). alpha bf16 [4, 4].
void commit_mix(const DraftParams* p, const __nv_bfloat16* alpha, const CtxLayers& L, const __nv_bfloat16* f_norm,
                __nv_bfloat16* xin, __nv_bfloat16* f_rnn, cudaStream_t stream);
// Small commit, part 2: kv fp32 [BLOCK, 3 * 2048] (BF16 values; layer l at columns l * 2048: k | v) of
// rows r < n stored as ctx_kv_store does, at positions p->start + r.
void commit_store(const float* kv, const DraftParams* p, const CtxLayers& L, cudaStream_t stream);

// ---- draft block ---------------------------------------------------------------------------------------

// Block input: res[0] = bf16(rms(float(embed[anchor])) * embed_norm), res[1..7] = mask_embed;
// xn = bf16(rms(res) * in_norm). Also clears the per-step picks (uint64 [GAMMA]).
void embed_block(const void* embed, const float* embed_scale, const float* embed_norm, const __nv_bfloat16* mask_embed,
                 const __nv_bfloat16* in_norm, const DraftParams* p, __nv_bfloat16* res, __nv_bfloat16* xn,
                 unsigned long long* picks, cudaStream_t stream);
inline void embed_block(const __nv_bfloat16* embed, const float* embed_norm, const __nv_bfloat16* mask_embed,
                        const __nv_bfloat16* in_norm, const DraftParams* p, __nv_bfloat16* res, __nv_bfloat16* xn,
                        unsigned long long* picks, cudaStream_t stream) {
  embed_block(static_cast<const void*>(embed), nullptr, embed_norm, mask_embed, in_norm, p, res, xn, picks, stream);
}

// A round's transition in one launch: sets p (commit_feats, ncommit, start, anchor, P) from the round inputs,
// and does commit_mix and embed_block for those values (the same arithmetic as the two launches).
void round_begin(DraftParams* p, const RoundInputs& in, const __nv_bfloat16* alpha, const CtxLayers& L,
                 const __nv_bfloat16* f_norm, const void* embed, const float* embed_scale, const float* embed_norm,
                 const __nv_bfloat16* mask_embed, const __nv_bfloat16* in_norm, __nv_bfloat16* xin,
                 __nv_bfloat16* f_rnn, __nv_bfloat16* res, __nv_bfloat16* xn, unsigned long long* picks,
                 cudaStream_t stream);
inline void round_begin(DraftParams* p, const RoundInputs& in, const __nv_bfloat16* alpha, const CtxLayers& L,
                        const __nv_bfloat16* f_norm, const __nv_bfloat16* embed, const float* embed_norm,
                        const __nv_bfloat16* mask_embed, const __nv_bfloat16* in_norm, __nv_bfloat16* xin,
                        __nv_bfloat16* f_rnn, __nv_bfloat16* res, __nv_bfloat16* xn, unsigned long long* picks,
                        cudaStream_t stream) {
  round_begin(p, in, alpha, L, f_norm, static_cast<const void*>(embed), nullptr, embed_norm, mask_embed, in_norm, xin,
              f_rnn, res, xn, picks, stream);
}

// q/k/v of the block rows: qkv fp32 [8, 5120] (BF16 values). q, k: bf16(rope(bf16(rms_head * norm))) at
// positions P + i; v as is. q -> q_out bf16 [8, 24, 128]. Keys/values go to the cache (BF16 or E4M3 rows):
// local layers at rows WINDOW + i (after the ring), the global layer at slot (P + i) % ctx.
void qkv_rope(const float* qkv, const __nv_bfloat16* q_norm, const __nv_bfloat16* k_norm, const DraftParams* p,
              __nv_bfloat16* q_out, void* k_cache, void* v_cache, bool fp8, bool local, int ctx,
              cudaStream_t stream);

// Bidirectional block attention (scale 1/sqrt(128), GQA head h -> kv head h / 3), out bf16 [8, 3072].
// Local (ctx_pos != null): keys = ring rows [0, WINDOW) valid when P - WINDOW <= ctx_pos < P, plus the 8
// block rows [WINDOW, WINDOW + 8); nkeys must be WINDOW + 8, BF16 rows. Global (ctx_pos null): keys
// [0, nkeys) valid below P + 8, BF16 or E4M3 rows (fp8). Tensor cores: the softmax probabilities are rounded
// to BF16 before they weigh the values. ws: attention_workspace_floats(nkeys) floats.
std::size_t attention_workspace_floats(int nkeys);
void block_attention(const __nv_bfloat16* q, const void* k_cache, const void* v_cache, int nkeys,
                     const std::int32_t* ctx_pos, bool fp8, const DraftParams* p, float* ws, __nv_bfloat16* out,
                     cudaStream_t stream);

// res = bf16(res + delta); xn = bf16(rms(res) * w). res, xn bf16 [rows, H], delta fp32 [rows, H] (BF16 values),
// w bf16 [H].
void add_rmsnorm(__nv_bfloat16* res, const float* delta, const __nv_bfloat16* w, __nv_bfloat16* xn, int rows,
                 cudaStream_t stream);

// out = bf16(silu(gate) * up), gu fp32 [rows, 2I] (gate | up), out bf16 [rows, I].
void silu_mul(const float* gu, int rows, int inter, __nv_bfloat16* out, cudaStream_t stream);

// ---- Markov head ---------------------------------------------------------------------------------------

// Top-CANDIDATES per row of base fp32 [rows, V] (BF16-representable values): the elements above the 1024th
// value and, of those equal to it, the ones with the lowest indices. Order (fixed, not sorted): the elements
// above it by ascending index, then the ties by ascending index. cand_val fp32 [rows, 1024]; cand_idx int32
// [rows, 1024] = the element's index, or map[index] (a shortlist head's row -> token id) if map is not null.
// A radix select over many CTAs per row; ws: topk_workspace_bytes(rows, V) bytes, zeroed before the first
// call (every call leaves its histograms zeroed).
std::size_t topk_workspace_bytes(int rows, int V);
void topk_candidates(const float* base, int rows, int V, const std::int32_t* map, float* cand_val,
                     std::int32_t* cand_idx, void* ws, cudaStream_t stream);

// Copies head rows ids[0..n) of src [V, row_bytes] (+ the FP32 row scales of FP8 rows if src_scale) to dst
// rows 0..n (+ dst_scale) and sets map[i] = ids[i]. row_bytes: a multiple of 16.
void gather_head_rows(const void* src, const float* src_scale, std::size_t row_bytes, const std::int32_t* ids, int n,
                      void* dst, float* dst_scale, std::int32_t* map, cudaStream_t stream);

// part[i, r] = sum_{c >= 512} rnn_w[r, c] z_i[c] in fp32, z_i[512:] = (h[i], f_rnn): the step-independent
// part of the rnn projection. rnn_w bf16 [768, 5120], h bf16 [GAMMA, H], f_rnn bf16 [H], part fp32 [7, 768].
void markov_pre(const __nv_bfloat16* rnn_w, const __nv_bfloat16* h, const __nv_bfloat16* f_rnn, float* part,
                cudaStream_t stream);

// Step i's cell: prev = anchor (i = 0) or cand_idx[i-1, argmax pick of step i-1];
// raw = bf16(bf16(rnn_w [state_i, W1[prev]] + part[i]) + rnn_b); gate = bf16(sigmoid), cell = bf16(tanh),
// state_{i+1} = bf16(bf16(gate state) + bf16(bf16(1 - gate) cell)), latent = bf16(tanh(raw[512:])).
// state bf16 [GAMMA + 1, 256] (row 0 must be zero), latent bf16 [256].
void markov_step(int i, const __nv_bfloat16* rnn_w, const __nv_bfloat16* rnn_b, const __nv_bfloat16* w1,
                 const float* part, const unsigned long long* picks, const std::int32_t* cand_idx,
                 const DraftParams* p, __nv_bfloat16* state, __nv_bfloat16* latent, cudaStream_t stream);

// Step i's candidate logits: bias = bf16(fold_b[c] + bf16(fold_w[c] . latent)),
// logit = bf16(30 tanh(bf16(cand_val + bias) / 30)) -> cand_logits fp32 [7, 1024]; picks[i] = argmax of the
// logits (greedy) or of logits / max(T, 1e-6) - log(-log(u)) (Gumbel-max), packed (pack_argmax).
void markov_logits(int i, const __nv_bfloat16* fold_w, const __nv_bfloat16* fold_b, const float* cand_val,
                   const std::int32_t* cand_idx, const __nv_bfloat16* latent, const DraftParams* p,
                   float* cand_logits, unsigned long long* picks, cudaStream_t stream);

// Outputs: tokens int32 [7] = picked candidates; draft_logits fp32 [7, V] = -inf except the step's
// candidates (the entries of the previous draft, prev_cand int32 [7, 1024], are reset first; the buffer
// must start all -inf); conf fp32 [7] = max softmax probability of each row; with p->temperature T > 0,
// qstats fp32 [7, 2] = (max_x l_x / T, sum_x exp(l_x / T - max)) of each row, the statistics of the draft
// distribution q = softmax(draft_logits / T) the acceptance needs; prev_cand <- cand_idx; p->draws += 1.
void markov_finish(const std::int32_t* cand_idx, const float* cand_logits, const unsigned long long* picks,
                   std::int32_t* prev_cand, float* draft_logits, int V, std::int32_t* tokens, float* conf,
                   float* qstats, DraftParams* p, cudaStream_t stream);

// ---- load time -----------------------------------------------------------------------------------------

// fold_w = bf16(U hw), fold_b = bf16(U hb) in FP64, U = the unembedding as weights.dequant gives it
// (u_scale != null: bf16(float(fp8) * scale) per row, else BF16 rows) [rows, H]; hw bf16 [H, 256],
// hb bf16 [H]; fold_w bf16 [rows, 256], fold_b bf16 [rows]. FP64 results round to FP32, then BF16 (torch's
// double -> bfloat16 conversion).
void fold_head(const void* u, const float* u_scale, int rows, const __nv_bfloat16* hw, const __nv_bfloat16* hb,
               __nv_bfloat16* fold_w, __nv_bfloat16* fold_b, cudaStream_t stream, const void* u_bscale = nullptr);

// Fills n floats with `value`.
void fill(float* x, std::size_t n, float value, cudaStream_t stream);

// ---- acceptance (engine.Engine._accept) ----------------------------------------------------------------

struct AcceptResult {
  std::int32_t accepted = 0;  // accepted draft count
  std::int32_t next = 0;      // the token after them (correction or bonus)
};

// Greedy: target int32 [n + 1] = argmax of each verify row; accepted = longest prefix with
// drafts[i] == target[i] (i < n), next = target[accepted].
void greedy_accept(const std::int32_t* target, const std::int32_t* drafts, int n, AcceptResult* out,
                   cudaStream_t stream);

// Sampled: p fp32 [n + 1, V] (engine._probs of the verify rows), draft_logits fp32 [n, V] with their q
// statistics qstats fp32 [n, 2] (markov_finish), u fp32 [n] uniforms. q_r = softmax(draft_logits[r] / T);
// accepted = number of leading r with u_r q_r(d_r) < p_r(d_r). resid fp32 [V] = max(p_a - q_a, 0)
// (a = accepted < n) or p_n (a = n); when its sum is 0 it is replaced by p_a. The next token is then a
// multinomial draw from resid (unnormalised), which the caller writes to out->next.
// ws: sampled_accept_workspace_floats(V) floats. n in 1..GAMMA.
std::size_t sampled_accept_workspace_floats(int V);
void sampled_accept(const float* p, const float* draft_logits, int V, const std::int32_t* drafts, int n,
                    float temperature, const float* qstats, const float* u, float* ws, float* resid,
                    AcceptResult* out, cudaStream_t stream);

}  // namespace ingine::kernels::draft
