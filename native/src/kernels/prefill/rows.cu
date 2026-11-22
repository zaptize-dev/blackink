// Prefill per-row kernels: attention input post-processing (model._qkv), relative logits, the short
// convolutions over time, message-gate pieces, taps, and the last row's logits. Arithmetic follows the PyTorch
// reference operation by operation (separately rounded products/sums where torch runs separate kernels;
// divisions by Python scalars as multiplications by the FP32 reciprocal, as torch does).
#include "kernels/prefill/common.cuh"
#include "kernels/prefill/prefill.h"

#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace ingine::kernels::prefill {
namespace {

constexpr int H = kHidden;
constexpr float kEps = 1e-6f;
constexpr int OFF_Q = 0, OFF_REL = 6144, OFF_SINK = 6528, OFF_K = 6552, OFF_V = 7576,
              OFF_KDYN = 8600, OFF_VDYN = 8632;
static_assert(OFF_V + kKvDim == OFF_KDYN && OFF_SINK + kHeads == OFF_K);

__device__ __forceinline__ float warp_sum(float v) {
#pragma unroll
  for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
  return v;
}

// model.l2_clip factor: (threshold * reciprocal(norm + 1e-6)).clamp(max=1) (torch scalar / tensor).
__device__ __forceinline__ float clip_factor(float sumsq, float threshold) {
  return fminf((1.f / (sqrtf(sumsq) + kEps)) * threshold, 1.f);
}

__device__ __forceinline__ void load_bf16x4(const uint16_t* p, float (&f)[4]) {
  const uint2 u = *reinterpret_cast<const uint2*>(p);
  f[0] = bf_lo(u.x), f[1] = bf_hi(u.x), f[2] = bf_lo(u.y), f[3] = bf_hi(u.y);
}

__device__ __forceinline__ void store_bf16x4(uint16_t* p, const float (&f)[4]) {
  *reinterpret_cast<uint2*>(p) = make_uint2(pack_bf16x2(f[0], f[1]), pack_bf16x2(f[2], f[3]));
}

// ---- attn_prep: one CTA (8 warps) per token

__global__ void __launch_bounds__(256) attn_prep_kernel(const AttnPrepParams p) {
  const int t = blockIdx.x, warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
  const uint16_t* row = reinterpret_cast<const uint16_t*>(p.proj) + size_t(t) * p.ld;
  const int pos = p.pos0 + t;
  // model.log_scale: 1 + 0.1 * log(clamp((p + 1) / 8192, min=1))
  const float tau = 1.f + 0.1f * logf(fmaxf(float(pos + 1) * (1.f / 8192.f), 1.f));
  const float qk_clip = 22.627416997969522f, rel_clip = 8.f;

  // q: heads warp, warp + 8, warp + 16; 4 dims per lane
#pragma unroll
  for (int i = 0; i < 3; ++i) {
    const int h = warp + 8 * i, d = lane * 4;
    float x[4];
    load_bf16x4(row + OFF_Q + h * kHeadDim + d, x);
    const float4 sc = *reinterpret_cast<const float4*>(p.q_scale + h * kHeadDim + d);
    x[0] *= sc.x, x[1] *= sc.y, x[2] *= sc.z, x[3] *= sc.w;
    const float f = clip_factor(warp_sum(x[0] * x[0] + x[1] * x[1] + x[2] * x[2] + x[3] * x[3]), qk_clip);
#pragma unroll
    for (int j = 0; j < 4; ++j) {
      x[j] = round_bf16(x[j] * f);
      if (p.full) x[j] = round_bf16(x[j] * tau);
    }
    store_bf16x4(reinterpret_cast<uint16_t*>(p.q) + (size_t(t) * kHeads + h) * kHeadDim + d, x);
  }
  // rel: heads warp, warp + 8, warp + 16; lanes 0-15 one dim each (16-31 duplicate)
#pragma unroll
  for (int i = 0; i < 3; ++i) {
    const int h = warp + 8 * i, j = lane & 15;
    float x = bf2f(row[OFF_REL + h * kRel + j]) * p.r_scale[h * kRel + j];
    float ss = x * x;
#pragma unroll
    for (int o = 8; o > 0; o >>= 1) ss += __shfl_xor_sync(0xffffffffu, ss, o);
    x = round_bf16(x * clip_factor(ss, rel_clip));
    if (p.full) x = round_bf16(x * tau);
    if (lane < 16) reinterpret_cast<uint16_t*>(p.rel)[(size_t(t) * kHeads + h) * kRel + j] = f2bf_bits(x);
  }
  // keys / values of KV head `warp`, 4 channels per lane
  const int c = warp * kHeadDim + lane * 4;
#pragma unroll
  for (int kv = 0; kv < 2; ++kv) {
    const int off = kv == 0 ? OFF_K : OFF_V;
    const float* state = kv == 0 ? p.conv_k : p.conv_v;
    const float* wt = kv == 0 ? p.k_w : p.v_w;
    // ext[j] = conv input at t - 3 + j (BF16-rounded projections; the state before the chunk)
    float ext[4][4];
#pragma unroll
    for (int j = 0; j < 4; ++j) {
      const int src = t - 3 + j;
      if (src >= 0) {
        load_bf16x4(reinterpret_cast<const uint16_t*>(p.proj) + size_t(src) * p.ld + off + c, ext[j]);
      } else {
        const float4 s4 = *reinterpret_cast<const float4*>(state + (3 + src) * kKvDim + c);
        ext[j][0] = s4.x, ext[j][1] = s4.y, ext[j][2] = s4.z, ext[j][3] = s4.w;
      }
    }
    float y[4];
    if (p.full) {  // model.dynamic_conv: w = static_w[lag] + dyn[t, head, lag]; (sum + x) / 3
      const int dyn_off = kv == 0 ? OFF_KDYN : OFF_VDYN;
      float acc[4] = {0.f, 0.f, 0.f, 0.f};
#pragma unroll
      for (int lag = 0; lag < 4; ++lag) {
        const float dyn = bf2f(row[dyn_off + warp * 4 + lag]);
        const float4 sw = *reinterpret_cast<const float4*>(wt + lag * kKvDim + c);
        const float swv[4] = {sw.x, sw.y, sw.z, sw.w};
#pragma unroll
        for (int q = 0; q < 4; ++q)
          acc[q] = __fadd_rn(acc[q], __fmul_rn(__fadd_rn(swv[q], dyn), ext[3 - lag][q]));
      }
      const float inv = 1.f / 3.f;
#pragma unroll
      for (int q = 0; q < 4; ++q) y[q] = round_bf16(__fmul_rn(__fadd_rn(acc[q], ext[3][q]), inv));
    } else {  // model.static_conv: weight [C, 4] conv1d order, (sum + x) / sqrt(5)
      const float inv = 1.f / 2.2360679774997896f;
#pragma unroll
      for (int q = 0; q < 4; ++q) {
        const float4 w4 = *reinterpret_cast<const float4*>(wt + size_t(c + q) * 4);
        float acc = __fmul_rn(w4.x, ext[0][q]);
        acc = __fadd_rn(acc, __fmul_rn(w4.y, ext[1][q]));
        acc = __fadd_rn(acc, __fmul_rn(w4.z, ext[2][q]));
        acc = __fadd_rn(acc, __fmul_rn(w4.w, ext[3][q]));
        y[q] = round_bf16(__fmul_rn(__fadd_rn(acc, ext[3][q]), inv));
      }
    }
    const float ss = warp_sum(y[0] * y[0] + y[1] * y[1] + y[2] * y[2] + y[3] * y[3]);
    if (kv == 0) {  // keys: l2_clip(QK_CLIP), BF16
      const float f = clip_factor(ss, qk_clip);
#pragma unroll
      for (int q = 0; q < 4; ++q) y[q] = round_bf16(y[q] * f);
    } else {  // values: rms_norm without weight, BF16
      const float r = rsqrtf(ss * (1.f / kHeadDim) + kEps);
#pragma unroll
      for (int q = 0; q < 4; ++q) y[q] = round_bf16(y[q] * r);
    }
    void* dst = kv == 0 ? p.k_out : p.v_out;
    if (p.out_q4) {  // 4 elements -> 2 packed Q4 bytes (uint16_t)
      const uint8_t b0 = quant_q4_nibble(y[0]) | uint8_t(quant_q4_nibble(y[1]) << 4);
      const uint8_t b1 = quant_q4_nibble(y[2]) | uint8_t(quant_q4_nibble(y[3]) << 4);
      const uint16_t b = uint16_t(b0) | (uint16_t(b1) << 8);
      *reinterpret_cast<uint16_t*>(static_cast<uint8_t*>(dst) + (size_t(t) * kKvDim + c) / 2) = b;
    } else if (p.out_fp8) {  // State.full_store: saturate to +-448, E4M3 round to nearest even
      const uint32_t b = uint32_t(f2e4m3(y[0])) | (uint32_t(f2e4m3(y[1])) << 8) | (uint32_t(f2e4m3(y[2])) << 16) |
                         (uint32_t(f2e4m3(y[3])) << 24);
      *reinterpret_cast<uint32_t*>(static_cast<uint8_t*>(dst) + size_t(t) * kKvDim + c) = b;
    } else {
      store_bf16x4(static_cast<uint16_t*>(dst) + size_t(t) * kKvDim + c, y);
    }
  }
}

// ---- rel_logits: 1024 distances x 4 per thread, rel_proj columns in registers; a CTA covers 8 heads of `tpb`
// consecutive tokens (grid: token groups x 3 head groups, enough CTAs for small chunks)

__global__ void __launch_bounds__(256) rel_logits_kernel(const uint16_t* __restrict__ rel,
                                                         const float* __restrict__ rel_proj, int T, int tpb,
                                                         uint16_t* __restrict__ out) {
  const int d = threadIdx.x * 4;
  float rp[kRel][4];
#pragma unroll
  for (int j = 0; j < kRel; ++j) {
    const float4 v = *reinterpret_cast<const float4*>(rel_proj + j * kRelExtent + d);
    rp[j][0] = round_bf16(v.x), rp[j][1] = round_bf16(v.y), rp[j][2] = round_bf16(v.z), rp[j][3] = round_bf16(v.w);
  }
  const int t1 = min(T, (blockIdx.x + 1) * tpb);
  for (int t = blockIdx.x * tpb; t < t1; ++t) {
    for (int h = blockIdx.y * 8; h < blockIdx.y * 8 + 8; ++h) {
      const uint16_t* r = rel + (size_t(t) * kHeads + h) * kRel;
      float acc[4] = {0.f, 0.f, 0.f, 0.f};
#pragma unroll
      for (int j = 0; j < kRel; ++j) {
        const float x = bf2f(__ldg(r + j));
#pragma unroll
        for (int q = 0; q < 4; ++q) acc[q] = fmaf(x, rp[j][q], acc[q]);
      }
      store_bf16x4(out + (size_t(t) * kHeads + h) * kRelExtent + d, acc);
    }
  }
}

// ---- static conv over time: thread = channel, CONV_ROWS consecutive rows

constexpr int CONV_ROWS = 16;

__global__ void static_conv_kernel(const uint16_t* __restrict__ x, int ld, int T, int C, const float* __restrict__ w,
                                   const float* __restrict__ state, float* __restrict__ out) {
  const int c = blockIdx.x * blockDim.x + threadIdx.x;
  if (c >= C) return;
  const int t0 = blockIdx.y * CONV_ROWS, t1 = min(T, t0 + CONV_ROWS);
  const float4 w4 = *reinterpret_cast<const float4*>(w + size_t(c) * 4);
  auto in = [&](int src) { return src >= 0 ? bf2f(x[size_t(src) * ld + c]) : state[(3 + src) * C + c]; };
  float e0 = in(t0 - 3), e1 = in(t0 - 2), e2 = in(t0 - 1);
  const float inv = 1.f / 2.2360679774997896f;
  for (int t = t0; t < t1; ++t) {
    const float e3 = bf2f(x[size_t(t) * ld + c]);
    float acc = __fmul_rn(w4.x, e0);
    acc = __fadd_rn(acc, __fmul_rn(w4.y, e1));
    acc = __fadd_rn(acc, __fmul_rn(w4.z, e2));
    acc = __fadd_rn(acc, __fmul_rn(w4.w, e3));
    out[size_t(t) * C + c] = __fmul_rn(__fadd_rn(acc, e3), inv);
    e0 = e1, e1 = e2, e2 = e3;
  }
}

__global__ void conv_state_kernel(float* __restrict__ state, const uint16_t* __restrict__ x, int ld, int T, int C) {
  const int c = blockIdx.x * blockDim.x + threadIdx.x;
  if (c >= C) return;
  for (int i = 0; i < 3; ++i) {  // in order: reads row T + i of the old state before row i is written
    const int src = T - 3 + i;
    state[i * C + c] = src >= 0 ? bf2f(x[size_t(src) * ld + c]) : state[(3 + src) * C + c];
  }
}

// ---- message gates (H / 4 threads per row, float4 each)

constexpr int ROW_T = H / 4;  // 576
constexpr int ROW_W = ROW_T / 32;

template <int N>
__device__ __forceinline__ void block_sum(float (&v)[N]) {
  __shared__ float red[ROW_W][N];
  const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
#pragma unroll
  for (int i = 0; i < N; ++i) v[i] = warp_sum(v[i]);
  if (lane == 0)
#pragma unroll
    for (int i = 0; i < N; ++i) red[warp][i] = v[i];
  __syncthreads();
#pragma unroll
  for (int i = 0; i < N; ++i) {
    float s = 0.f;
    for (int w = 0; w < ROW_W; ++w) s += red[w][i];
    v[i] = s;
  }
}

__device__ __forceinline__ float dot4(float4 a, float4 b) { return a.x * b.x + a.y * b.y + a.z * b.z + a.w * b.w; }

__global__ void __launch_bounds__(ROW_T) gate_inputs_kernel(const GateRows rows, int T, uint16_t* __restrict__ gin) {
  const int t = blockIdx.x, g = blockIdx.y, c = threadIdx.x * 4;
  const float4 h = *reinterpret_cast<const float4*>(rows.h[g] + size_t(t) * H + c);
  const float4 m = *reinterpret_cast<const float4*>(rows.msg[g] + size_t(t) * H + c);
  float ss[2] = {dot4(h, h), dot4(m, m)};
  block_sum(ss);
  const float rh = rsqrtf(ss[0] * (1.f / H) + kEps), rm = rsqrtf(ss[1] * (1.f / H) + kEps);
  uint16_t* o = gin + (size_t(g) * T + t) * 2 * H;
  store_bf16x4(o + c, {h.x * rh, h.y * rh, h.z * rh, h.w * rh});
  store_bf16x4(o + H + c, {m.x * rm, m.y * rm, m.z * rm, m.w * rm});
}

__global__ void __launch_bounds__(ROW_T) gate_apply_kernel(const float* __restrict__ h, const float* __restrict__ msg,
                                                           const float* __restrict__ f, const float* __restrict__ gn,
                                                           const float* __restrict__ norm_w, uint16_t* __restrict__ out) {
  const size_t off = size_t(blockIdx.x) * H + threadIdx.x * 4;
  const float4 h4 = *reinterpret_cast<const float4*>(h + off), m4 = *reinterpret_cast<const float4*>(msg + off);
  const float4 f4 = *reinterpret_cast<const float4*>(f + off), g4 = *reinterpret_cast<const float4*>(gn + off);
  const float hv[4] = {h4.x, h4.y, h4.z, h4.w}, mv[4] = {m4.x, m4.y, m4.z, m4.w};
  const float fv[4] = {f4.x, f4.y, f4.z, f4.w}, gv[4] = {g4.x, g4.y, g4.z, g4.w};
  float y[4];
  float ss[1] = {0.f};
#pragma unroll
  for (int i = 0; i < 4; ++i) {  // (1 - forget) * hidden + gain * message
    y[i] = __fadd_rn(__fmul_rn(__fsub_rn(1.f, tanhf(fv[i])), hv[i]), __fmul_rn(tanhf(gv[i]), mv[i]));
    ss[0] += y[i] * y[i];
  }
  block_sum(ss);
  const float r = rsqrtf(ss[0] * (1.f / H) + kEps);
  const float4 w4 = *reinterpret_cast<const float4*>(norm_w + threadIdx.x * 4);
  store_bf16x4(out + off, {y[0] * r * w4.x, y[1] * r * w4.y, y[2] * r * w4.z, y[3] * r * w4.w});
}

// ---- misc

__global__ void widen_kernel(const int32_t* __restrict__ in, int T, int64_t* __restrict__ out) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < T) out[i] = in[i];
}

__global__ void __launch_bounds__(ROW_T) tap_kernel(const float* __restrict__ x, uint16_t* __restrict__ out,
                                                    int64_t stride) {
  const int t = blockIdx.x, c = threadIdx.x * 4;
  const float4 v = *reinterpret_cast<const float4*>(x + size_t(t) * H + c);
  store_bf16x4(out + size_t(t) * stride + c, {v.x, v.y, v.z, v.w});
}

// Warp per vocabulary row: dot in FP32 of bf16(w) (or bf16(fp8 * scale)) with x, BF16-rounded, softcapped.
template <bool FP8>
__global__ void __launch_bounds__(256) head_kernel(const uint16_t* __restrict__ x, const void* __restrict__ w,
                                                   const float* __restrict__ scale, int V, float* __restrict__ out) {
  __shared__ float xs[H];
  for (int i = threadIdx.x; i < H; i += blockDim.x) xs[i] = bf2f(x[i]);
  __syncthreads();
  const int v = blockIdx.x * 8 + (threadIdx.x >> 5), lane = threadIdx.x & 31;
  if (v >= V) return;
  float acc = 0.f;
  if constexpr (FP8) {
    const uint8_t* row = static_cast<const uint8_t*>(w) + size_t(v) * H;
    const float s = scale[v];
    for (int k = lane * 8; k < H; k += 256) {
      const uint2 u = __ldg(reinterpret_cast<const uint2*>(row + k));
      const uint32_t ws[2] = {u.x, u.y};
#pragma unroll
      for (int q = 0; q < 2; ++q) {
        const float2 a = e4m3x2_to_float2(ws[q] & 0xffff), b = e4m3x2_to_float2(ws[q] >> 16);
        acc = fmaf(round_bf16(a.x * s), xs[k + 4 * q], acc);
        acc = fmaf(round_bf16(a.y * s), xs[k + 4 * q + 1], acc);
        acc = fmaf(round_bf16(b.x * s), xs[k + 4 * q + 2], acc);
        acc = fmaf(round_bf16(b.y * s), xs[k + 4 * q + 3], acc);
      }
    }
  } else {
    const uint16_t* row = static_cast<const uint16_t*>(w) + size_t(v) * H;
    for (int k = lane * 8; k < H; k += 256) {
      const uint4 u = __ldg(reinterpret_cast<const uint4*>(row + k));
      const uint32_t ws[4] = {u.x, u.y, u.z, u.w};
#pragma unroll
      for (int q = 0; q < 4; ++q) {
        acc = fmaf(bf_lo(ws[q]), xs[k + 2 * q], acc);
        acc = fmaf(bf_hi(ws[q]), xs[k + 2 * q + 1], acc);
      }
    }
  }
  acc = warp_sum(acc);
  if (lane == 0) out[v] = tanhf(round_bf16(acc) * (1.f / 30.f)) * 30.f;
}

}  // namespace

void attn_prep(const AttnPrepParams& p, cudaStream_t stream) {
  if (p.T <= 0 || !p.proj || !p.q || !p.rel || !p.k_out || !p.v_out || !p.conv_k || !p.conv_v)
    throw std::invalid_argument("attn_prep: missing argument");
  attn_prep_kernel<<<p.T, 256, 0, stream>>>(p);
}

void rel_logits(const __nv_bfloat16* rel, const float* rel_proj, int T, __nv_bfloat16* out, cudaStream_t stream) {
  const int tpb = std::min(16, std::max(1, T / 128));
  rel_logits_kernel<<<dim3((T + tpb - 1) / tpb, kHeads / 8), 256, 0, stream>>>(
      reinterpret_cast<const uint16_t*>(rel), rel_proj, T, tpb, reinterpret_cast<uint16_t*>(out));
}

void static_conv(const __nv_bfloat16* x, int ld, int T, int C, const float* w, const float* state, float* out,
                 cudaStream_t stream) {
  const dim3 grid((C + 127) / 128, (T + CONV_ROWS - 1) / CONV_ROWS);
  static_conv_kernel<<<grid, 128, 0, stream>>>(reinterpret_cast<const uint16_t*>(x), ld, T, C, w, state, out);
}

void conv_state_update(float* state, const __nv_bfloat16* x, int ld, int T, int C, cudaStream_t stream) {
  conv_state_kernel<<<(C + 127) / 128, 128, 0, stream>>>(state, reinterpret_cast<const uint16_t*>(x), ld, T, C);
}

void gate_inputs(const GateRows& rows, int G, int T, __nv_bfloat16* gin, cudaStream_t stream) {
  gate_inputs_kernel<<<dim3(T, G), ROW_T, 0, stream>>>(rows, T, reinterpret_cast<uint16_t*>(gin));
}

void gate_apply(const float* h, const float* msg, const float* f, const float* g, const float* norm_w, int T,
                __nv_bfloat16* out, cudaStream_t stream) {
  gate_apply_kernel<<<T, ROW_T, 0, stream>>>(h, msg, f, g, norm_w, reinterpret_cast<uint16_t*>(out));
}

void widen_ids(const int32_t* in, int T, int64_t* out, cudaStream_t stream) {
  widen_kernel<<<(T + 255) / 256, 256, 0, stream>>>(in, T, out);
}

void store_tap(const float* x, int T, __nv_bfloat16* out, int64_t out_stride, cudaStream_t stream) {
  tap_kernel<<<T, ROW_T, 0, stream>>>(x, reinterpret_cast<uint16_t*>(out), out_stride);
}

void head_logits(const __nv_bfloat16* x, const void* w, const float* scale, int V, float* out, cudaStream_t stream) {
  const auto* xs = reinterpret_cast<const uint16_t*>(x);
  if (scale) head_kernel<true><<<(V + 7) / 8, 256, 0, stream>>>(xs, w, scale, V, out);
  else head_kernel<false><<<(V + 7) / 8, 256, 0, stream>>>(xs, w, scale, V, out);
}

}  // namespace ingine::kernels::prefill
