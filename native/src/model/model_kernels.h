// Small device helpers of the model runtime (not part of the ported decode kernels): step inputs, the
// bookkeeping between pipelined width-1 steps (Runtime::chain_*), the speculative verify call's state backup
// and rollback, and the selectors of the device-chosen verify width (conditional graph nodes).
#pragma once

#include "model/model_data.h"

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <cstdint>

namespace ingine::model {

// dst[r * dst_stride + c] = bf16(src[r * cols + c]) (round to nearest even, torch copy_ to BF16).
void f32_to_bf16_rows(const float* src, int rows, int cols, __nv_bfloat16* dst, std::int64_t dst_stride,
                      cudaStream_t stream);

// Device pointers of the recurrent decode state a step overwrites besides the cache rows at its own
// positions: the conv states (fp32 [48, 3, C], oldest input first; C = KV_DIM for k/v, HIDDEN for attn/mlp)
// and the sliding rings (k/v bf16 [40, WINDOW, KV_DIM], positions int64 [40, WINDOW]).
struct StatePtrs {
  float* conv[4];  // conv_k, conv_v, conv_attn, conv_mlp
  __nv_bfloat16* k_ring;
  __nv_bfloat16* v_ring;
  std::int64_t* ring_pos;
};
inline constexpr int CONV_COLS[4] = {KV_DIM, KV_DIM, HIDDEN, HIDDEN};
inline constexpr int CONV_ROW_FLOATS = 2 * KV_DIM + 2 * HIDDEN;  // one row of all four conv states per visit

// What one width-1 step at position p overwrites: row 0 of every conv state (shifted out by the step) and
// ring slot p % WINDOW of every sliding visit.
struct StepUndo {
  float* conv;           // fp32 [48, CONV_ROW_FLOATS]: per visit k | v | attn | mlp
  __nv_bfloat16* k;      // bf16 [40, KV_DIM]
  __nv_bfloat16* v;      // bf16 [40, KV_DIM]
  std::int64_t* pos;     // int64 [40]
};

// What a verify call of up to 8 rows at P overwrites: conv state rows 1 and 2 (row 0 is never restored:
// a rollback keeps at least one row) and ring slots (P + r) % WINDOW, r < m.
struct VerifyBackup {
  float* conv[4];        // fp32 [48, 3, C] like the conv states (rows 1, 2 written)
  __nv_bfloat16* k;      // bf16 [40, 8, KV_DIM]
  __nv_bfloat16* v;      // bf16 [40, 8, KV_DIM]
  std::int64_t* pos;     // int64 [40, 8]
};

// token[i] = tokens[i] (i < m), *pos = position: the inputs of a decode call (m <= 8).
void set_step_inputs(const std::int32_t* tokens, int m, int position, std::int64_t* token, std::int64_t* pos,
                     cudaStream_t stream);

// Between pipelined width-1 steps: *token = *sampled, *pos = position, *host_token = *sampled (mapped host
// memory, visible to the host once the kernel completed), and `undo` = what the step at `position` will
// overwrite (taken from the current state).
void chain_advance(const std::int32_t* sampled, std::int64_t* token, std::int64_t* pos, int position,
                   std::int32_t* host_token, const StatePtrs& st, const StepUndo& undo, cudaStream_t stream);

// Undoes the width-1 step at `position`, the last step run on `st`, from the record chain_advance took
// before it: conv rows shift back (row 2 = row 1, row 1 = row 0, row 0 = undo) and the ring slot returns.
void chain_undo(const StatePtrs& st, const StepUndo& undo, int position, cudaStream_t stream);

// Saves what a verify call of m rows at P overwrites. P = *pos_dev if pos_dev, else pos_host.
void verify_save(const StatePtrs& st, const VerifyBackup& bk, const std::int64_t* pos_dev, int pos_host, int m,
                 cudaStream_t stream);

// After a verify call of m rows at P (history on: hist[i] fp32 [48, m, C] = each row's conv inputs) keeps
// its first `commit` rows: conv state = cat([old, hist])[:, commit:commit + 3], ring slots of rows >= commit
// back from the backup. commit = *accepted_dev + 1 if accepted_dev, else commit_host; nothing to do when
// commit == m. P = *pos_dev if pos_dev, else pos_host.
void verify_rollback(const StatePtrs& st, const VerifyBackup& bk, const float* const hist[4],
                     const std::int64_t* pos_dev, int pos_host, const std::int32_t* accepted_dev, int commit_host,
                     int m, cudaStream_t stream);

// Up to 3 verify widths, ascending (Runtime::verify_widths()).
struct Widths {
  int n = 0;
  int w[3] = {};
};

// Verify width selection on the device (engine._generate_spec): lead = leading conf[i] > threshold (i < 7),
// m = the smallest width >= lead + 1; token[1 + i] = drafts[i] (i < 7), *m_out = m, and the switch
// condition `handle` = index of m in `widths`. Captured into the graph that owns `handle`.
void verify_select(const float* conf, const std::int32_t* drafts, float threshold, const Widths& widths,
                   std::int64_t* token, std::int32_t* m_out, cudaGraphConditionalHandle handle, cudaStream_t stream);

// The switch condition `handle` = index of *m in `widths`. Captured into the graph that owns `handle`.
void width_select(const std::int32_t* m, const Widths& widths, cudaGraphConditionalHandle handle,
                  cudaStream_t stream);

// The host's view of a speculative round: written to mapped host memory by verify_publish.
struct VerifyRecord {
  std::int32_t m;          // verify width
  std::int32_t accepted;   // accepted drafts
  std::int32_t next;       // the token after them (greedy; sampled decoding writes it separately)
  std::int32_t drafts[7];  // the drafted tokens (the verify rows 1..7)
};

// rec->m = *m, rec->accepted = *accepted, rec->next = *next (if next != nullptr), rec->drafts = token[1..7]
// (rec: mapped host memory).
void verify_publish(const std::int32_t* m, const std::int32_t* accepted, const std::int32_t* next,
                    const std::int64_t* token, VerifyRecord* rec, cudaStream_t stream);

}  // namespace ingine::model
