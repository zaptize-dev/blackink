// C++ mirrors of the Python engine's model data: weights.py (constants, visit plan, Weights/Layer/Experts),
// model.py (State, Ablation, TAP_VISITS) and fast.py (projection layout, Buffers per call width).
//
// Tensors are ingine::Tensor views over device memory. Shape convention: Tensor::ne[] lists dimensions
// innermost first (ggml order), so a Python tensor of shape [a, b, c] has ne = {c, b, a}; every field
// comment below gives the Python (row-major) shape. All tensors are contiguous unless noted.
// `rm(data, dtype, {a, b, c})` builds a view from a Python-order shape.
#pragma once

#include "core/arena.h"
#include "core/tensor.h"

#include <array>
#include <cstdint>
#include <initializer_list>
#include <vector>

namespace ingine::model {

// ------------------------------------------------------------------------------------ weights.py constants

inline constexpr int HIDDEN = 2304;
inline constexpr int HEADS = 24, KV_HEADS = 8, HEAD_DIM = 128;
inline constexpr int D_REL = 16, REL_EXTENT = 1024, WINDOW = 1024;
inline constexpr int INTER = 1152, EXPERTS = 128, TOP_K = 8;
inline constexpr int CONV_K = 4;
inline constexpr int LAYERS = 24, BLOCK = 6;
inline constexpr int LOOP_START = 6, LOOP_END = 18, PASSES = 3;
inline constexpr int VOCAB = 201024, UNPADDED_VOCAB = 200058;
inline constexpr int GATE_RANK = 64;
inline constexpr float EPS = 1e-6f;
inline constexpr double QK_CLIP = 22.627416997969522, REL_CLIP = 8.0;
inline constexpr double LOG_ALPHA = 0.1, LOG_FLOOR = 8192.0;
inline constexpr float GATE_SOFTCAP = 4.0f, UP_SOFTCAP = 25.0f, LOGIT_SOFTCAP = 30.0f;
inline constexpr float ROUTE_SCALE = 8.0f;
inline constexpr float FP8_MAX = 448.0f;  // largest finite float8_e4m3fn (model.FP8_MAX)
inline constexpr int KV_DIM = KV_HEADS * HEAD_DIM;  // 1024

inline constexpr bool is_full_layer(int l) { return l == 5 || l == 11 || l == 17 || l == 23; }
inline constexpr bool is_loop_layer(int l) { return LOOP_START <= l && l < LOOP_END; }

// --------------------------------------------------------------------------------------------- visit plan

struct Visit {
  int layer;
  int pass;  // loop pass (0 outside the loop)
};

inline constexpr int NUM_VISITS = 48;
inline constexpr int NUM_FULL_VISITS = 8;
inline constexpr int NUM_SLIDING_VISITS = NUM_VISITS - NUM_FULL_VISITS;

// weights.visit_plan(): layers 0-5, layers 6-17 three times, layers 18-23
inline constexpr std::array<Visit, NUM_VISITS> VISITS = [] {
  std::array<Visit, NUM_VISITS> plan{};
  int v = 0;
  for (int l = 0; l < LOOP_START; ++l) plan[v++] = {l, 0};
  for (int p = 0; p < PASSES; ++p)
    for (int l = LOOP_START; l < LOOP_END; ++l) plan[v++] = {l, p};
  for (int l = LOOP_END; l < LAYERS; ++l) plan[v++] = {l, 0};
  return plan;
}();

// model.State.ring_index / full_index: visit -> index into the sliding rings / full caches (-1: other kind)
inline constexpr std::array<int, NUM_VISITS> RING_INDEX = [] {
  std::array<int, NUM_VISITS> r{};
  int n = 0;
  for (int v = 0; v < NUM_VISITS; ++v) r[v] = is_full_layer(VISITS[v].layer) ? -1 : n++;
  return r;
}();
inline constexpr std::array<int, NUM_VISITS> FULL_INDEX = [] {
  std::array<int, NUM_VISITS> r{};
  int n = 0;
  for (int v = 0; v < NUM_VISITS; ++v) r[v] = is_full_layer(VISITS[v].layer) ? n++ : -1;
  return r;
}();

// model.TAP_VISITS: DSpark features = the partial block entering these visits, then the final normed hidden
inline constexpr std::array<int, 3> TAP_VISITS = {5, 23, 41};
inline constexpr int NUM_TAPS = 4;

// ----------------------------------------------------------------------------- fast.py projection layout

// SEG_ROWS order (KERNELS.md): q | gate | rel | sink | k | v | (full layers) k_dyn | v_dyn
enum Seg : int { SEG_Q = 0, SEG_GATE, SEG_REL, SEG_SINK, SEG_K, SEG_V, SEG_K_DYN, SEG_V_DYN, NUM_SEGS };
inline constexpr std::array<int, NUM_SEGS> SEG_ROWS = {HEADS * HEAD_DIM, HEADS * HEAD_DIM, HEADS * D_REL, HEADS,
                                                       KV_DIM,           KV_DIM,           KV_HEADS * CONV_K,
                                                       KV_HEADS * CONV_K};
inline constexpr std::array<int, NUM_SEGS> SEG_INPUT = {0, 0, 0, 0, 1, 2, 1, 2};  // xin index (q, k, v inputs)
inline constexpr std::array<int, NUM_SEGS + 1> SEG_BEGIN = [] {
  std::array<int, NUM_SEGS + 1> b{};
  for (int s = 0; s < NUM_SEGS; ++s) b[s + 1] = b[s] + SEG_ROWS[s];
  return b;
}();
inline constexpr int PROJ_ROWS_SLIDING = SEG_BEGIN[SEG_K_DYN];  // 8600
inline constexpr int PROJ_ROWS_FULL = SEG_BEGIN[NUM_SEGS];      // 8664
inline constexpr int num_segs(bool full) { return full ? NUM_SEGS : SEG_K_DYN; }
inline constexpr int proj_rows(bool full) { return full ? PROJ_ROWS_FULL : PROJ_ROWS_SLIDING; }

// --------------------------------------------------------------------------------------------- helpers

// View with a Python-order (row-major) shape.
inline Tensor rm(void* data, DType dtype, std::initializer_list<std::int32_t> python_shape) {
  std::int32_t s[4] = {1, 1, 1, 1};
  const int n = static_cast<int>(python_shape.size());
  int i = n - 1;
  for (std::int32_t d : python_shape) s[i--] = d;
  switch (n) {
    case 1: return Tensor(data, dtype, {s[0]});
    case 2: return Tensor(data, dtype, {s[0], s[1]});
    case 3: return Tensor(data, dtype, {s[0], s[1], s[2]});
    default: return Tensor(data, dtype, {s[0], s[1], s[2], s[3]});
  }
}

template <class T>
inline T* ptr(const Tensor& t) {
  return static_cast<T*>(t.data);
}

// ----------------------------------------------------------------------------------------------- weights

// A dense matrix as the decoder stores it: BF16 rows, or float8_e4m3fn rows with FP32 per-row scales
// (`scale.data != nullptr`; value = fp8 * scale, see weights.dequant / dense fp8_rows).
struct Dense {
  Tensor w;      // [rows, cols] bf16 or fp8
  Tensor scale;  // [rows] fp32, or empty (data == nullptr) for BF16
  bool fp8() const { return scale.data != nullptr; }
  int rows() const { return w.ne[1]; }
  int cols() const { return w.ne[0]; }
};

enum class ExpertFormat : int { NVFP4 = 0, MXFP8 = 1 };  // the moe kernels' `fmt`

// weights.Experts (checkpoint encoding; see weights.py docstring)
struct Experts {
  ExpertFormat fmt = ExpertFormat::NVFP4;
  // NVFP4: w13 u8 [128, 2304, 1152], w13_scale e4m3 [128, 2304, 144], w2 u8 [128, 2304, 576],
  //        w2_scale e4m3 [128, 2304, 72], factors fp32 [128] (amax / outer_scale_denominator)
  // MXFP8: w13 e4m3 [128, 2304, 2304], w13_scale u8 (E8M0) [128, 2304, 72], w2 e4m3 [128, 2304, 1152],
  //        w2_scale u8 [128, 2304, 36], factors empty
  Tensor w13, w13_scale, w2, w2_scale;
  Tensor w13_factor, w2_factor;
};

// weights.Layer plus the per-layer derived tensors of fast.FastDecoder (_concat_projections,
// _stack_layer_params, _fp8_dense). Absent fields have data == nullptr.
struct Layer {
  int idx = 0;
  bool full = false, loop = false;

  // norms (fp32 [H])
  Tensor q_norm, k_norm, v_norm;  // full layers
  Tensor attn_norm;               // sliding layers
  Tensor mlp_norm;

  // attention projections: one [R, H] matrix in SEG order (R = proj_rows(full)); the per-projection
  // entries are row views of it (with the matching rows of its scales when FP8)
  Dense wproj;
  Dense wq, wgate, wr, wsink, wk, wv, wk_dyn, wv_dyn;
  Dense wo;  // [H, 3072]

  Tensor q_scale;   // fp32 [24, 128]
  Tensor r_scale;   // fp32 [24, 16]
  Tensor rel_proj;  // fp32 [16, 1024]
  Tensor k_static, v_static;  // full: fp32 [4, 1024] by lag (row 0 = current token)
  Tensor k_conv, v_conv;      // sliding: fp32 [1024, 4] conv1d order (tap 3 = current token)
  Tensor attn_conv, mlp_conv;  // fp32 [H, 4]
  Tensor router;       // fp32 [128, H]
  Tensor router_bias;  // fp32 [128]
  Tensor router_bf16;  // bf16 [128, H] (exact copy of the checkpoint values)
  Tensor attn_query;   // fp32 [Q, H] (Q = 3 full, 1 sliding), layers > 0
  Tensor mlp_query;    // fp32 [1, H]

  // loop layers: message gates. Stacks the kernels take, with per-gate views into them.
  Tensor qkv_gate_down;                      // bf16 [3, 64, 2H]
  Tensor qkv_gate_forget, qkv_gate_gain;     // fp32 [3, H, 64]
  Tensor qkv_gate_forget16, qkv_gate_gain16; // bf16 [3, H, 64]
  Tensor mlp1_gate_down;                     // bf16 [1, 64, 2H] (= mlp_gate_down)
  Tensor mlp1_gate_forget, mlp1_gate_gain;   // fp32 [1, H, 64]
  Tensor mlp1_gate_forget16, mlp1_gate_gain16;
  Tensor q_gate_down, k_gate_down, v_gate_down, mlp_gate_down;          // bf16 [64, 2H] views
  Tensor q_gate_forget, k_gate_forget, v_gate_forget, mlp_gate_forget;  // fp32 [H, 64] views
  Tensor q_gate_gain, k_gate_gain, v_gate_gain, mlp_gate_gain;          // fp32 [H, 64] views

  // stacked norm weights (fast._stack_layer_params)
  Tensor attn_norm3;     // fp32 [3, H]: (q, k, v) norms, or attn_norm three times
  Tensor attn_norm_mix;  // fp32 [3, H] (full) or [1, H] (sliding)
  Tensor mlp_norm1;      // fp32 [1, H]

  Experts experts;
};

// weights.Weights (+ fast.FastDecoder.final_norm1 / head_w). Owns its device memory.
struct Weights {
  std::vector<Layer> layers;  // LAYERS
  Tensor embed;               // bf16 or fp8 [VOCAB, H] (padded rows kept, as weights.load)
  Tensor embed_scale;         // fp32 [VOCAB] when embed is FP8, else empty
  Tensor embed_norm;          // fp32 [H]
  Tensor final_query;         // fp32 [1, H]
  Tensor final_norm;          // fp32 [H] (also final_norm1 = [1, H] view of the same memory)
  Dense unembed;              // [UNPADDED_VOCAB, H] bf16, or FP8 rows + scales (dense_fp8)
  Tensor message_queries;     // fp32 [12, 4, H]: q, k, v, mlp per loop layer
  std::array<ExpertFormat, LAYERS> expert_formats{};
  bool dense_fp8 = false;     // wproj (+ views), wo and unembed are FP8 rows

  std::vector<DeviceBuffer> storage;  // device allocations behind the views
};

// model.Ablation: per visit y <- y - w * r * (r . y) on the attention / MLP writes. Without an ablation
// file (Ablation.none) directions and weights are zero.
struct Ablation {
  Tensor attn_dirs;  // fp32 [48, H] unit rows
  Tensor attn_w;     // fp32 [48]
  Tensor mlp_dirs;   // fp32 [48, H]
  Tensor mlp_w;      // fp32 [48]
  bool active = false;  // loaded from a file

  std::vector<DeviceBuffer> storage;
};

// ------------------------------------------------------------------------------------------------- state

// model.State: per-request caches.
struct State {
  int max_ctx = 0;
  bool kv_fp8 = false;
  bool kv_q4 = false;
  Tensor k_ring, v_ring;      // bf16 [40, 1024, 8, 128] (sliding visits, RING_INDEX)
  Tensor ring_pos;            // int64 [40, 1024], absolute position per slot, -1 = empty
  Tensor k_full, v_full;      // bf16, fp8 e4m3 [8, max_ctx, 8, 128], or u8 q4 [8, max_ctx, 8, 64] (full visits, FULL_INDEX)
  Tensor conv_k, conv_v;      // fp32 [48, 3, 1024] previous 3 key/value conv inputs (oldest first)
  Tensor conv_attn, conv_mlp; // fp32 [48, 3, H]
  int length = 0;             // tokens in the context (host copy; decode steps do not advance it)

  std::vector<DeviceBuffer> storage;

  // per-visit views
  Tensor k_ring_at(int ring) const;
  Tensor v_ring_at(int ring) const;
  std::int64_t* ring_pos_at(int ring) const;
  Tensor k_full_at(int full) const;
  Tensor v_full_at(int full) const;
  float* conv_k_at(int v) const { return ptr<float>(conv_k) + std::size_t(v) * 3 * KV_DIM; }
  float* conv_v_at(int v) const { return ptr<float>(conv_v) + std::size_t(v) * 3 * KV_DIM; }
  float* conv_attn_at(int v) const { return ptr<float>(conv_attn) + std::size_t(v) * 3 * HIDDEN; }
  float* conv_mlp_at(int v) const { return ptr<float>(conv_mlp) + std::size_t(v) * 3 * HIDDEN; }
};

// ----------------------------------------------------------------------------------------------- buffers

// fast.Buffers: fixed device buffers of one call width M (one set per captured CUDA graph).
//   B0 = embedding; B1 = hidden entering layer 6 (pass 0); B2[p] = hidden entering layer 12 in pass p;
//   E[p] = hidden after layer 17 in pass p (E[2] is also the block entering layer 18).
struct Buffers {
  int m = 0;
  Tensor token;                     // int64 [m] input token ids (read by embed)
  Tensor B0, B1;                    // fp32 [m, H]
  std::array<Tensor, PASSES> B2, E; // fp32 [m, H]
  Tensor hid, part, y, y2;          // fp32 [m, H]
  Tensor xin;                       // bf16 [3, m, H]
  Tensor mixf;                      // fp32 [3, m, H]
  Tensor msgs;                      // fp32 [4, m, H]
  Tensor xm;                        // bf16 [1, m, H]
  Tensor mixm;                      // fp32 [1, m, H]
  Tensor proj_sliding, proj_full;   // fp32 [m, 8600], [m, 8664] (fast: proj_out keyed by row count)
  Tensor attn_out;                  // bf16 [m, 3072]
  Tensor idx;                       // int32 [m, 8]
  Tensor wts;                       // fp32 [m, 8]
  Tensor act;                       // bf16 [m, 8, I]
  Tensor xf;                        // bf16 [1, m, H]
  Tensor logits;                    // fp32 [m, UNPADDED_VOCAB]
  Tensor taps;                      // bf16 [m, 4, H]: partial block entering TAP_VISITS + final normed hidden
  // convolution inputs of each row per visit, for rolling back rejected rows (history = true)
  Tensor hist_attn, hist_mlp;       // fp32 [48, m, H]
  Tensor hist_k, hist_v;            // fp32 [48, m, 1024]

  std::vector<DeviceBuffer> storage;

  // fast.Buffers.blocks: completed blocks visible to layer l's attention read in pass p
  std::vector<const float*> blocks(int l, int p) const;
};

}  // namespace ingine::model
