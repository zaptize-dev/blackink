// The decode runtime on a synthetic model (random weights in the real layouts; experts shared by all layers
// and routing pinned to 8 experts, so it needs well under 2 GB of GPU memory): graph replay equals the eager
// step bit for bit; the verify widths 2/4/8 reproduce consecutive width-1 steps; the speculative rollback
// (save_verify / rollback) leaves the state a shorter call would; taps and logits are finite; pipelined
// width-1 steps (chain_advance) equal explicit inputs and chain_undo restores the state bit for bit; the
// device-selected verify (switch graph, positions and accepted count on the device) equals the host-driven
// call. The Generator (plain, pipelined decoding): stop tokens, cancellation and max_new_tokens end the
// output exactly, and a follow-up request reusing the prefix generates what it generates after a request
// that never ran past its end.
#include "common/test_util.h"

#include "kernels/dense/dense.h"
#include "model/model_kernels.h"
#include "model/runtime.h"

#include "ingine/generator.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstring>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

using namespace ingine;
using namespace ingine::model;
using ingine::test::float_to_bf16;
using ingine::test::Rng;

namespace {

constexpr int kVocab = 4096;  // unembedding rows used by the synthetic model (the rest are zero; token ids stay below)
constexpr int kExperts = 8;   // experts actually allocated (the router bias pins routing to them)
constexpr int H = HIDDEN;

struct Gen {
  Rng rng{77};
  std::vector<DeviceBuffer>& st;
  cudaStream_t stream;

  void* up(const void* host, std::size_t bytes) {
    st.emplace_back(bytes);
    st.back().copy_from_host(host, bytes);
    return st.back().p;
  }
  Tensor f32(std::initializer_list<std::int32_t> shape, double sd, double mean = 0) {
    std::size_t n = 1;
    for (auto d : shape) n *= std::size_t(d);
    std::vector<float> h(n);
    for (auto& x : h) x = float(mean + rng.normal() * sd);
    return rm(up(h.data(), n * 4), DType::FP32, shape);
  }
  Tensor bf16(std::initializer_list<std::int32_t> shape, double sd) {
    std::size_t n = 1;
    for (auto d : shape) n *= std::size_t(d);
    std::vector<__nv_bfloat16> h(n);
    for (auto& x : h) x = float_to_bf16(float(rng.normal() * sd));
    return rm(up(h.data(), n * 2), DType::BF16, shape);
  }
  Dense fp8(int rows, int cols, double sd) {
    const Tensor b = bf16({rows, cols}, sd);
    st.emplace_back(std::size_t(rows) * cols);
    void* q = st.back().p;
    st.emplace_back(std::size_t(rows) * 4);
    float* s = static_cast<float*>(st.back().p);
    kernels::dense::fp8_rows(static_cast<const __nv_bfloat16*>(b.data), rows, cols, q, s, stream);
    INGINE_TEST_CUDA(cudaStreamSynchronize(stream));
    return Dense{rm(q, DType::FP8_E4M3FN, {rows, cols}), rm(s, DType::FP32, {rows})};
  }
  Tensor bytes(std::initializer_list<std::int32_t> shape, DType dt, int lo, int hi) {
    std::size_t n = 1;
    for (auto d : shape) n *= std::size_t(d);
    std::vector<std::uint8_t> h(n);
    for (auto& x : h) x = std::uint8_t(lo + int(rng.next() % std::uint64_t(hi - lo + 1)));
    return rm(up(h.data(), n), dt, shape);
  }
};

Weights synthetic_weights(cudaStream_t stream) {
  Weights w;
  w.storage.reserve(1024);
  Gen g{Rng(77), w.storage, stream};
  w.dense_fp8 = true;
  // experts shared by every layer (NVFP4); only experts 0..7 exist, the router bias selects exactly them
  Experts ex;
  ex.fmt = ExpertFormat::NVFP4;
  ex.w13 = g.bytes({kExperts, 2 * INTER, H / 2}, DType::U8, 0, 255);
  ex.w13_scale = g.bytes({kExperts, 2 * INTER, H / 16}, DType::FP8_E4M3FN, 0x30, 0x48);  // E4M3 ~ 0.5..4
  ex.w2 = g.bytes({kExperts, H, INTER / 2}, DType::U8, 0, 255);
  ex.w2_scale = g.bytes({kExperts, H, INTER / 16}, DType::FP8_E4M3FN, 0x30, 0x48);
  ex.w13_factor = g.f32({EXPERTS}, 0.0, 0.004);
  ex.w2_factor = g.f32({EXPERTS}, 0.0, 0.004);
  std::vector<float> bias(EXPERTS, 0.f);
  for (int e = 0; e < kExperts; ++e) bias[e] = 10.f;
  const Tensor router_bias = rm(g.up(bias.data(), bias.size() * 4), DType::FP32, {EXPERTS});

  const Dense wproj[2] = {g.fp8(PROJ_ROWS_SLIDING, H, 0.02), g.fp8(PROJ_ROWS_FULL, H, 0.02)};
  const Dense wo = g.fp8(H, HEADS * HEAD_DIM, 0.02);
  const Tensor ones3 = g.f32({3, H}, 0.05, 1.0), ones1 = g.f32({1, H}, 0.05, 1.0);
  for (int l = 0; l < LAYERS; ++l) {
    Layer L;
    L.idx = l;
    L.full = is_full_layer(l);
    L.loop = is_loop_layer(l);
    L.wproj = wproj[L.full];
    L.wo = wo;
    L.q_scale = g.f32({HEADS, HEAD_DIM}, 0.1, 1.0);
    L.r_scale = g.f32({HEADS, D_REL}, 0.1, 1.0);
    L.rel_proj = g.f32({D_REL, REL_EXTENT}, 0.3);
    if (L.full) {
      L.k_static = g.f32({CONV_K, KV_DIM}, 0.3);
      L.v_static = g.f32({CONV_K, KV_DIM}, 0.3);
    } else {
      L.k_conv = g.f32({KV_DIM, CONV_K}, 0.3);
      L.v_conv = g.f32({KV_DIM, CONV_K}, 0.3);
    }
    L.attn_conv = g.f32({H, CONV_K}, 0.3);
    L.mlp_conv = g.f32({H, CONV_K}, 0.3);
    L.router_bf16 = g.bf16({EXPERTS, H}, 0.02);
    L.router_bias = router_bias;
    if (l > 0) L.attn_query = g.f32({L.full ? 3 : 1, H}, 0.05);
    L.mlp_query = g.f32({1, H}, 0.05);
    L.attn_norm3 = ones3;
    L.attn_norm_mix = L.full ? ones3 : rm(ones3.data, DType::FP32, {1, H});
    L.mlp_norm1 = ones1;
    if (L.loop) {
      L.qkv_gate_down = g.bf16({3, GATE_RANK, 2 * H}, 0.02);
      L.qkv_gate_forget16 = g.bf16({3, H, GATE_RANK}, 0.1);
      L.qkv_gate_gain16 = g.bf16({3, H, GATE_RANK}, 0.1);
      L.mlp1_gate_down = g.bf16({1, GATE_RANK, 2 * H}, 0.02);
      L.mlp1_gate_forget16 = g.bf16({1, H, GATE_RANK}, 0.1);
      L.mlp1_gate_gain16 = g.bf16({1, H, GATE_RANK}, 0.1);
    }
    L.experts = ex;
    w.layers.push_back(L);
    w.expert_formats[l] = ExpertFormat::NVFP4;
  }
  w.embed = g.bf16({kVocab, H}, 0.05);
  w.embed_norm = g.f32({H}, 0.05, 1.0);
  w.final_query = g.f32({1, H}, 0.05);
  w.final_norm = g.f32({H}, 0.05, 1.0);
  w.message_queries = g.f32({LOOP_END - LOOP_START, 4, H}, 0.05);
  // the head and the prefill read every vocabulary row: rows >= kVocab are zero (logit 0), below the best of
  // the random rows, so greedy and top-k sampled tokens stay below kVocab
  const Dense used = g.fp8(kVocab, H, 0.05);
  const std::size_t q_bytes = std::size_t(UNPADDED_VOCAB) * H, s_bytes = std::size_t(UNPADDED_VOCAB) * 4;
  w.storage.emplace_back(q_bytes);
  void* q = w.storage.back().p;
  w.storage.emplace_back(s_bytes);
  void* sc = w.storage.back().p;
  INGINE_TEST_CUDA(cudaMemsetAsync(q, 0, q_bytes, stream));
  INGINE_TEST_CUDA(cudaMemsetAsync(sc, 0, s_bytes, stream));
  INGINE_TEST_CUDA(cudaMemcpyAsync(q, used.w.data, std::size_t(kVocab) * H, cudaMemcpyDeviceToDevice, stream));
  INGINE_TEST_CUDA(cudaMemcpyAsync(sc, used.scale.data, std::size_t(kVocab) * 4, cudaMemcpyDeviceToDevice, stream));
  INGINE_TEST_CUDA(cudaStreamSynchronize(stream));
  w.unembed = Dense{rm(q, DType::FP8_E4M3FN, {UNPADDED_VOCAB, H}), rm(sc, DType::FP32, {UNPADDED_VOCAB})};
  return w;
}

std::vector<float> host(const float* d, std::size_t n, cudaStream_t s) {
  std::vector<float> h(n);
  INGINE_TEST_CUDA(cudaMemcpyAsync(h.data(), d, n * 4, cudaMemcpyDeviceToHost, s));
  INGINE_TEST_CUDA(cudaStreamSynchronize(s));
  return h;
}

template <class T>
std::vector<T> host_t(const Tensor& t, cudaStream_t s) {
  std::vector<T> h(std::size_t(t.numel()));
  INGINE_TEST_CUDA(cudaMemcpyAsync(h.data(), t.data, h.size() * sizeof(T), cudaMemcpyDeviceToHost, s));
  INGINE_TEST_CUDA(cudaStreamSynchronize(s));
  return h;
}

// row `row` of width-m logits ([m, vocabulary rows] buffer, kVocab columns used)
std::vector<float> logits_row(Runtime& rt, int m, int row) {
  const std::size_t stride = std::size_t(rt.weights().unembed.rows());
  return host(static_cast<const float*>(rt.buffers(m).logits.data) + row * stride, kVocab, rt.stream());
}

double rel_l2(const std::vector<float>& a, const std::vector<float>& b) {
  double d = 0, n = 0;
  for (std::size_t i = 0; i < a.size(); ++i) d += (double(a[i]) - b[i]) * (double(a[i]) - b[i]), n += double(b[i]) * b[i];
  return std::sqrt(d / std::max(n, 1e-30));
}

bool finite(const std::vector<float>& v) {
  for (float x : v)
    if (!std::isfinite(x)) return false;
  return true;
}

bool report(bool ok, const std::string& what) {
  std::printf("%s %s\n", ok ? "PASS" : "FAIL", what.c_str());
  return ok;
}

// Every state tensor a decode step changes besides the cache rows at its own positions.
struct StateCopy {
  std::vector<float> conv_k, conv_v, conv_attn, conv_mlp;
  std::vector<std::int64_t> ring_pos;
  std::vector<std::uint16_t> k_ring, v_ring;
  bool operator==(const StateCopy&) const = default;
};

StateCopy copy_state(Runtime& rt) {
  const State& st = rt.state();
  const cudaStream_t s = rt.stream();
  return {host_t<float>(st.conv_k, s),         host_t<float>(st.conv_v, s),         host_t<float>(st.conv_attn, s),
          host_t<float>(st.conv_mlp, s),       host_t<std::int64_t>(st.ring_pos, s), host_t<std::uint16_t>(st.k_ring, s),
          host_t<std::uint16_t>(st.v_ring, s)};
}

// all m rows of width-m logits (kVocab columns each)
std::vector<float> logits_rows(Runtime& rt, int m) {
  std::vector<float> all;
  for (int r = 0; r < m; ++r) {
    const auto row = logits_row(rt, m, r);
    all.insert(all.end(), row.begin(), row.end());
  }
  return all;
}

bool run(bool kv_fp8) {
  const std::string tag = kv_fp8 ? " (fp8 kv)" : " (bf16 kv)";
  RuntimeOptions opt;
  opt.max_ctx = 2048;
  opt.spec = true;
  opt.kv_fp8 = kv_fp8;
  cudaStream_t s0;
  INGINE_TEST_CUDA(cudaStreamCreate(&s0));
  Weights w = synthetic_weights(s0);
  Runtime rt(opt, std::move(w), load_ablation({}));
  INGINE_TEST_CUDA(cudaStreamDestroy(s0));
  const cudaStream_t s = rt.stream();
  const std::int32_t toks[8] = {11, 523, 7, 4000, 19, 2048, 3, 77};
  bool ok = true;

  // 1. graph replay == eager step (bit for bit), from the same state
  reset_state(rt.state(), s);
  {
    const Runtime::Snapshot snap = rt.snapshot();
    rt.set_inputs({toks, 1}, 0);
    rt.run_step_eager(1);
    const auto eager = logits_row(rt, 1, 0);
    rt.restore(snap);
    rt.set_inputs({toks, 1}, 0);
    rt.run_step(1);
    const auto graph = logits_row(rt, 1, 0);
    ok &= report(finite(eager) && eager == graph, "graph replay equals the eager step" + tag);
  }

  // 2. six width-1 steps vs one width-2 and one width-4 call
  std::vector<std::vector<float>> seq;
  reset_state(rt.state(), s);
  for (int i = 0; i < 6; ++i) {
    rt.decode_logits(toks[i], i);
    seq.push_back(logits_row(rt, 1, 0));
  }
  const auto taps1 = host_t<__nv_bfloat16>(rt.buffers(1).taps, s);
  reset_state(rt.state(), s);
  rt.set_inputs({toks, 2}, 0);
  rt.run_step(2);
  std::vector<std::vector<float>> wide = {logits_row(rt, 2, 0), logits_row(rt, 2, 1)};
  rt.set_inputs({toks + 2, 4}, 2);
  rt.run_step(4);
  for (int i = 0; i < 4; ++i) wide.push_back(logits_row(rt, 4, i));
  double worst = 0;
  bool all_finite = true;
  for (int i = 0; i < 6; ++i) worst = std::max(worst, rel_l2(wide[i], seq[i])), all_finite &= finite(wide[i]);
  ok &= report(all_finite && worst < 0.03,
               "widths 2 + 4 reproduce six width-1 steps (worst relative L2 " + std::to_string(worst) + ")" + tag);
  {
    const auto taps4 = host_t<__nv_bfloat16>(rt.buffers(4).taps, s);
    bool f = true;
    for (const auto* t : {&taps1, &taps4})
      for (auto x : *t) f &= std::isfinite(ingine::test::bf16_to_float(x));
    ok &= report(f, "taps finite" + tag);
  }

  // 3. rollback: a width-8 verify at P = 0 rolled back to its first 3 rows against three width-1 steps
  //    (conv states and ring positions), then one more step on both
  reset_state(rt.state(), s);
  for (int i = 0; i < 3; ++i) rt.decode_logits(toks[i], i);
  const auto ref_ck = host_t<float>(rt.state().conv_k, s), ref_cm = host_t<float>(rt.state().conv_mlp, s);
  const auto ref_rp = host_t<std::int64_t>(rt.state().ring_pos, s);
  rt.decode_logits(toks[3], 3);
  const auto ref_next = logits_row(rt, 1, 0);

  reset_state(rt.state(), s);
  rt.save_verify(0, 8);
  rt.set_inputs({toks, 8}, 0);
  rt.run_step(8);
  rt.rollback(0, 3, 8);
  const auto ck = host_t<float>(rt.state().conv_k, s), cm = host_t<float>(rt.state().conv_mlp, s);
  const auto rp = host_t<std::int64_t>(rt.state().ring_pos, s);
  ok &= report(rp == ref_rp, "rollback restores the ring positions" + tag);
  ok &= report(rel_l2(ck, ref_ck) < 0.03 && rel_l2(cm, ref_cm) < 0.03,
               "rollback leaves the conv states of three steps (relative L2 " + std::to_string(rel_l2(ck, ref_ck)) +
                   ", " + std::to_string(rel_l2(cm, ref_cm)) + ")" + tag);
  rt.decode_logits(toks[3], 3);
  const auto next = logits_row(rt, 1, 0);
  ok &= report(finite(next) && rel_l2(next, ref_next) < 0.03,
               "decoding after the rollback matches (relative L2 " + std::to_string(rel_l2(next, ref_next)) + ")" + tag);

  // 4. a rollback across the ring wrap (P = 1020, slots 1020..1027 mod 1024)
  reset_state(rt.state(), s);
  const auto before = host_t<std::int64_t>(rt.state().ring_pos, s);
  rt.save_verify(1020, 8);
  rt.set_inputs({toks, 8}, 1020);
  rt.run_step(8);
  rt.rollback(1020, 2, 8);
  const auto after = host_t<std::int64_t>(rt.state().ring_pos, s);
  bool wrap_ok = true;
  for (int ring = 0; ring < NUM_SLIDING_VISITS; ++ring)
    for (int slot = 0; slot < WINDOW; ++slot) {
      const std::int64_t got = after[std::size_t(ring) * WINDOW + slot];
      const std::int64_t want = slot == 1020 ? 1020 : slot == 1021 ? 1021 : before[std::size_t(ring) * WINDOW + slot];
      wrap_ok &= got == want;
    }
  ok &= report(wrap_ok, "rollback across the ring wrap" + tag);

  // 5. pipelined width-1 steps (input token from device memory) equal explicitly set inputs bit for bit
  DeviceBuffer sampled(sizeof(std::int32_t));
  auto put_token = [&](std::int32_t t) {
    INGINE_TEST_CUDA(cudaMemcpyAsync(sampled.p, &t, sizeof t, cudaMemcpyHostToDevice, s));
  };
  {
    std::vector<std::vector<float>> explicit_steps, chained;
    reset_state(rt.state(), s);
    for (int i = 0; i < 4; ++i) {
      rt.decode_logits(toks[i], 1000 + i);
      explicit_steps.push_back(logits_row(rt, 1, 0));
    }
    reset_state(rt.state(), s);
    bool tokens_ok = true;
    for (int i = 0; i < 4; ++i) {
      put_token(toks[i]);
      rt.chain_advance(static_cast<const std::int32_t*>(sampled.p), 1000 + i);
      rt.run_step(1);
      tokens_ok &= rt.chain_token(1000 + i) == toks[i];
      chained.push_back(logits_row(rt, 1, 0));
    }
    ok &= report(tokens_ok && chained == explicit_steps, "chained width-1 steps equal explicit inputs" + tag);
  }

  // 6. chain_undo restores the state before the step bit for bit (slot 0 after the ring wrap: position 1024)
  {
    reset_state(rt.state(), s);
    for (int i = 0; i < 3; ++i) rt.decode_logits(toks[i], 1021 + i);
    const StateCopy before_step = copy_state(rt);
    put_token(toks[3]);
    rt.chain_advance(static_cast<const std::int32_t*>(sampled.p), 1024);
    rt.run_step(1);
    const bool changed = !(copy_state(rt) == before_step);
    rt.chain_undo(1024);
    ok &= report(changed && copy_state(rt) == before_step, "chain_undo restores the state bit for bit" + tag);
  }

  // 7. the device-selected verify (switch over the widths: save, decode, rollback with the width, position and
  //    accepted count read on the device) equals the host-driven call bit for bit
  {
    DeviceBuffer width(sizeof(std::int32_t)), accepted(sizeof(std::int32_t));
    const std::span<const int> vw = rt.verify_widths();
    Widths widths;
    widths.n = int(vw.size());
    std::copy(vw.begin(), vw.end(), widths.w);
    GraphExec verify = rt.capture_switch(
        [&](cudaStream_t cs, cudaGraphConditionalHandle h) {
          width_select(static_cast<const std::int32_t*>(width.p), widths, h, cs);
        },
        [&](cudaStream_t cs, int m) {
          rt.record_save_verify(m, cs);
          rt.record_step(m, cs);
          rt.record_rollback(m, static_cast<const std::int32_t*>(accepted.p), cs);
        });
    const int P = 1018;  // rows cross the ring wrap
    for (auto [m, acc] : {std::pair{8, 2}, std::pair{4, 3}, std::pair{2, 0}}) {
      auto prologue = [&] {
        reset_state(rt.state(), s);
        for (int i = 0; i < 2; ++i) rt.decode_logits(toks[7 - i], P - 2 + i);
      };
      prologue();
      rt.save_verify(P, m);
      rt.set_inputs({toks, std::size_t(m)}, P);
      rt.run_step(m);
      const auto host_logits = logits_rows(rt, m);
      rt.rollback(P, acc + 1, m);
      const StateCopy host_state = copy_state(rt);

      prologue();
      rt.set_inputs({toks, std::size_t(m)}, P);
      INGINE_TEST_CUDA(cudaMemcpyAsync(width.p, &m, 4, cudaMemcpyHostToDevice, s));
      INGINE_TEST_CUDA(cudaMemcpyAsync(accepted.p, &acc, 4, cudaMemcpyHostToDevice, s));
      verify.launch(s);
      const auto device_logits = logits_rows(rt, m);
      ok &= report(device_logits == host_logits && copy_state(rt) == host_state,
                   "device-selected verify width " + std::to_string(m) + ", commit " + std::to_string(acc + 1) +
                       " equals the host-driven call" + tag);
    }
  }

  // 8. prompt checkpoint round trip: save, decode 1100 steps (the rings wrap), restore: bit-identical state
  {
    reset_state(rt.state(), s);
    for (int i = 0; i < 3; ++i) rt.decode_logits(toks[i], i);
    const StateCopy saved_state = copy_state(rt);
    DeviceBuffer ck(rt.checkpoint_bytes());
    cudaEvent_t e0, e1, e2, e3;
    for (cudaEvent_t* e : {&e0, &e1, &e2, &e3}) INGINE_TEST_CUDA(cudaEventCreate(e));
    INGINE_TEST_CUDA(cudaEventRecord(e0, s));
    rt.save_checkpoint(ck.p);
    INGINE_TEST_CUDA(cudaEventRecord(e1, s));
    for (int i = 3; i < 1103; ++i) rt.decode_logits(toks[i % 8], i);
    INGINE_TEST_CUDA(cudaEventRecord(e2, s));
    rt.restore_checkpoint(ck.p);
    INGINE_TEST_CUDA(cudaEventRecord(e3, s));
    INGINE_TEST_CUDA(cudaEventSynchronize(e3));
    float save_ms = 0, restore_ms = 0;
    INGINE_TEST_CUDA(cudaEventElapsedTime(&save_ms, e0, e1));
    INGINE_TEST_CUDA(cudaEventElapsedTime(&restore_ms, e2, e3));
    for (cudaEvent_t e : {e0, e1, e2, e3}) INGINE_TEST_CUDA(cudaEventDestroy(e));
    char buf[160];
    std::snprintf(buf, sizeof buf, "checkpoint round trip restores the state bit for bit (%.0f MB, save %.3f ms, restore %.3f ms)",
                  double(rt.checkpoint_bytes()) / 1e6, save_ms, restore_ms);
    ok &= report(copy_state(rt) == saved_state, buf + tag);
  }

  // 8b. conversation parking: A decodes 1100 positions, parks (checkpoint + full rows [0, 1100)); B starts
  //     over and decodes 1500 positions (every full row A used and the rings overwritten); A restored decodes
  //     on with the same logits, bit for bit, as A never displaced
  {
    const int n = 1100, more = 20;
    auto run_a = [&](int from, int to, std::vector<float>* logits) {
      for (int i = from; i < to; ++i) {
        rt.decode_logits(toks[i % 8], i);
        if (logits) {
          const auto row = logits_row(rt, 1, 0);
          logits->insert(logits->end(), row.begin(), row.end());
        }
      }
    };
    reset_state(rt.state(), s);
    run_a(0, n, nullptr);
    std::vector<float> want, got;
    run_a(n, n + more, &want);
    reset_state(rt.state(), s);
    run_a(0, n, nullptr);
    DeviceBuffer ck(rt.checkpoint_bytes()), rows(rt.full_rows_bytes(n));
    rt.save_checkpoint(ck.p);
    rt.save_full_rows(rows.p, n);
    reset_state(rt.state(), s);
    for (int i = 0; i < 1500; ++i) rt.decode_logits(toks[(3 * i + 1) % 8], i);
    rt.restore_checkpoint(ck.p);
    rt.restore_full_rows(rows.p, n);
    run_a(n, n + more, &got);
    char buf[160];
    std::snprintf(buf, sizeof buf, "parked conversation restores bit for bit after another one (%.1f KB per position)",
                  double(rt.full_rows_bytes(n)) / n / 1e3);
    ok &= report(!want.empty() && got == want, buf + tag);
  }

  // 9. device width selection: verify_select (confidences -> lead -> width, drafts into token[1..7]) for
  //    leads 0, 1, 2, 4, 7; width_select of a width that is not a verify width runs no branch; verify_publish
  //    writes the host record
  {
    const std::span<const int> vw = rt.verify_widths();
    Widths widths;
    widths.n = int(vw.size());
    std::copy(vw.begin(), vw.end(), widths.w);
    DeviceBuffer conf(7 * 4), drafts(7 * 4), m_out(4), marker(4), m_in(4), acc(4), next(4);
    auto mark = [&](cudaStream_t cs, int m) { INGINE_TEST_CUDA(cudaMemsetAsync(marker.p, m, 1, cs)); };
    GraphExec select = rt.capture_switch(
        [&](cudaStream_t cs, cudaGraphConditionalHandle h) {
          verify_select(static_cast<const float*>(conf.p), static_cast<const std::int32_t*>(drafts.p), 0.8f, widths,
                        rt.tokens(), static_cast<std::int32_t*>(m_out.p), h, cs);
        },
        mark);
    GraphExec by_width = rt.capture_switch(
        [&](cudaStream_t cs, cudaGraphConditionalHandle h) {
          width_select(static_cast<const std::int32_t*>(m_in.p), widths, h, cs);
        },
        mark);
    const std::int32_t d[7] = {101, 102, 103, 104, 105, 106, 107};
    INGINE_TEST_CUDA(cudaMemcpy(drafts.p, d, sizeof d, cudaMemcpyHostToDevice));
    bool sel_ok = true;
    for (auto [lead, want] : {std::pair{0, 2}, std::pair{1, 2}, std::pair{2, 4}, std::pair{4, 8}, std::pair{7, 8}}) {
      float c[7];
      for (int i = 0; i < 7; ++i) c[i] = i < lead ? 0.9f : (i == lead ? 0.8f : 0.95f);  // 0.8 is not > 0.8
      INGINE_TEST_CUDA(cudaMemcpy(conf.p, c, sizeof c, cudaMemcpyHostToDevice));
      INGINE_TEST_CUDA(cudaMemset(marker.p, 0, 4));
      INGINE_TEST_CUDA(cudaMemset(rt.tokens(), 0, 8 * 8));
      INGINE_TEST_CUDA(cudaDeviceSynchronize());  // the uploads above are not ordered with stream s
      select.launch(s);
      INGINE_TEST_CUDA(cudaStreamSynchronize(s));
      std::int32_t m = 0, mk = 0;
      std::int64_t tk[8];
      INGINE_TEST_CUDA(cudaMemcpy(&m, m_out.p, 4, cudaMemcpyDeviceToHost));
      INGINE_TEST_CUDA(cudaMemcpy(&mk, marker.p, 4, cudaMemcpyDeviceToHost));
      INGINE_TEST_CUDA(cudaMemcpy(tk, rt.tokens(), sizeof tk, cudaMemcpyDeviceToHost));
      bool tok_ok = tk[0] == 0;
      for (int i = 0; i < 7; ++i) tok_ok &= tk[1 + i] == d[i];
      sel_ok &= m == want && (mk & 0xff) == want && tok_ok;
    }
    ok &= report(sel_ok, "verify_select picks the width of leads 0/1/2/4/7 and copies the drafts" + tag);
    bool none_ok = true;
    for (auto [m, want] : {std::pair{4, 4}, std::pair{5, 0}}) {
      INGINE_TEST_CUDA(cudaMemset(marker.p, 0, 4));
      INGINE_TEST_CUDA(cudaMemcpy(m_in.p, &m, 4, cudaMemcpyHostToDevice));
      INGINE_TEST_CUDA(cudaDeviceSynchronize());
      by_width.launch(s);
      INGINE_TEST_CUDA(cudaStreamSynchronize(s));
      std::int32_t mk = 0;
      INGINE_TEST_CUDA(cudaMemcpy(&mk, marker.p, 4, cudaMemcpyDeviceToHost));
      none_ok &= (mk & 0xff) == want;
    }
    ok &= report(none_ok, "width_select runs the branch of a verify width and none for another width" + tag);
    MappedHostBuffer rec(sizeof(VerifyRecord));
    const std::int32_t m = 4, a = 2, nx = 777;
    INGINE_TEST_CUDA(cudaMemcpy(m_in.p, &m, 4, cudaMemcpyHostToDevice));
    INGINE_TEST_CUDA(cudaMemcpy(acc.p, &a, 4, cudaMemcpyHostToDevice));
    INGINE_TEST_CUDA(cudaMemcpy(next.p, &nx, 4, cudaMemcpyHostToDevice));
    INGINE_TEST_CUDA(cudaDeviceSynchronize());
    auto* r = static_cast<VerifyRecord*>(rec.data());
    r->next = -5;
    verify_publish(static_cast<const std::int32_t*>(m_in.p), static_cast<const std::int32_t*>(acc.p), nullptr,
                   rt.tokens(), static_cast<VerifyRecord*>(rec.device_ptr()), s);
    INGINE_TEST_CUDA(cudaStreamSynchronize(s));
    bool pub_ok = r->m == 4 && r->accepted == 2 && r->next == -5;  // next untouched without a source
    for (int i = 0; i < 7; ++i) pub_ok &= r->drafts[i] == d[i];
    verify_publish(static_cast<const std::int32_t*>(m_in.p), static_cast<const std::int32_t*>(acc.p),
                   static_cast<const std::int32_t*>(next.p), rt.tokens(), static_cast<VerifyRecord*>(rec.device_ptr()), s);
    INGINE_TEST_CUDA(cudaStreamSynchronize(s));
    pub_ok &= r->next == 777;
    ok &= report(pub_ok, "verify_publish writes the width, accepted count, drafts and next token" + tag);
  }
  return ok;
}

// The Generator over the synthetic model (plain pipelined decoding, FP8 caches, 64-token prefill chunks).
bool run_generator() {
  RuntimeOptions opt;
  opt.max_ctx = 2048;
  opt.kv_fp8 = true;
  opt.prefill_chunk = 64;
  cudaStream_t s0;
  INGINE_TEST_CUDA(cudaStreamCreate(&s0));
  Weights w = synthetic_weights(s0);
  INGINE_TEST_CUDA(cudaStreamDestroy(s0));
  const auto gen = make_generator(opt, std::move(w), load_ablation({}));

  Rng rng(5);
  std::vector<std::int32_t> prompt(100);
  for (auto& t : prompt) t = std::int32_t(rng.next() % kVocab);
  struct Out {
    std::vector<std::int32_t> tokens;
    GenerateStats stats;
  };
  // cancel_after >= 0: on_tokens returns false once that many tokens arrived
  auto run_gen = [&](const std::vector<std::int32_t>& ids, int max_new, const Sampling& sampling,
                     std::vector<std::int32_t> stop, int cancel_after = -1) {
    Out o;
    o.stats = gen->generate(ids, max_new, sampling, stop, [&](std::span<const std::int32_t> t) {
      o.tokens.insert(o.tokens.end(), t.begin(), t.end());
      return cancel_after < 0 || int(o.tokens.size()) < cancel_after;
    });
    return o;
  };
  auto head = [](const std::vector<std::int32_t>& v, int n) { return std::vector<std::int32_t>(v.begin(), v.begin() + n); };

  bool ok = true;
  Sampling greedy;
  greedy.temperature = 0.f;
  Sampling sampled;
  sampled.temperature = 1.f;
  sampled.top_k = 50;
  sampled.seed = 7;
  for (const auto& [name, sp] : {std::pair{"greedy", greedy}, std::pair{"sampled", sampled}}) {
    const std::string tag = std::string(" (") + name + ")";
    const Out ref = run_gen(prompt, 48, sp, {});
    bool ref_ok = ref.tokens.size() == 48 && ref.stats.generated_tokens == 48;
    for (auto t : ref.tokens) ref_ok &= t >= 0 && t < kVocab;
    ok &= report(ref_ok, "48 tokens below the used vocabulary" + tag);
    const Out again = run_gen(prompt, 48, sp, {});
    ok &= report(again.tokens == ref.tokens, "a repeated request generates the same tokens" + tag);

    const Out capped = run_gen(prompt, 9, sp, {});
    ok &= report(capped.tokens == head(ref.tokens, 9) && capped.stats.generated_tokens == 9,
                 "max_new_tokens ends the output exactly" + tag);

    // follow-ups after an end at token k (stop token / cancel, the step for token k was already launched)
    // against a request that ended there by its budget (no step launched past it)
    Sampling follow_sp = sp;
    follow_sp.seed = 11;
    std::vector<int> ends = {0};
    for (int k = 12; k < 48 && ends.size() < 2; ++k)
      if (std::find(ref.tokens.begin(), ref.tokens.begin() + k, ref.tokens[k]) == ref.tokens.begin() + k)
        ends.push_back(k);
    ok &= report(ends.size() == 2, "a token after the 12th that did not occur before (stop candidate)" + tag);
    for (int k : ends) {
      const std::string at = " at token " + std::to_string(k) + tag;
      std::vector<std::int32_t> follow = prompt;
      follow.insert(follow.end(), ref.tokens.begin(), ref.tokens.begin() + k + 1);
      for (int i = 0; i < 6; ++i) follow.push_back(std::int32_t(rng.next() % kVocab));
      const int reuse = int(prompt.size()) + k;

      const Out budget = run_gen(prompt, k + 1, sp, {});
      const Out fresh = run_gen(follow, 24, follow_sp, {});
      ok &= report(budget.tokens == head(ref.tokens, k + 1) && fresh.stats.reused_tokens == reuse &&
                       fresh.tokens.size() == 24,
                   "reference follow-up after a budget end" + at);

      const Out stopped = run_gen(prompt, 48, sp, {ref.tokens[k]});
      const Out after_stop = run_gen(follow, 24, follow_sp, {});
      ok &= report(stopped.tokens == head(ref.tokens, k + 1) && stopped.stats.generated_tokens == k + 1,
                   "a stop token ends the output exactly" + at);
      ok &= report(after_stop.stats.reused_tokens == reuse && after_stop.tokens == fresh.tokens,
                   "the follow-up after a stop token reuses the prefix and matches" + at);

      const Out cancelled = run_gen(prompt, 48, sp, {}, k + 1);
      const Out after_cancel = run_gen(follow, 24, follow_sp, {});
      ok &= report(cancelled.tokens == head(ref.tokens, k + 1) && cancelled.stats.generated_tokens == k + 1,
                   "cancellation ends the output exactly" + at);
      ok &= report(after_cancel.stats.reused_tokens == reuse && after_cancel.tokens == fresh.tokens,
                   "the follow-up after a cancel reuses the prefix and matches" + at);
    }

    // prompt checkpoints: X prefilled the prompt P, generated G and gets P + Y (Y diverges from G); Z prefilled
    // P with no new tokens and gets P + Y through the strict-prefix path. Same chunk boundaries, so X (restored
    // checkpoint) and Z must generate identical tokens. G longer than the 1024-slot ring as well.
    std::vector<std::int32_t> other(50);
    for (auto& t : other) t = std::int32_t(rng.next() % kVocab);
    struct Case {
      int g, common;
    };
    for (const Case c : {Case{40, 0}, Case{40, 3}, Case{1100, 0}}) {
      const std::string at = " (G " + std::to_string(c.g) + " tokens, Y shares " + std::to_string(c.common) + ")" + tag;
      run_gen(other, 2, sp, {});
      const Out g = run_gen(prompt, c.g, sp, {});
      std::vector<std::int32_t> follow = prompt;
      follow.insert(follow.end(), g.tokens.begin(), g.tokens.begin() + c.common);
      follow.push_back((g.tokens[c.common] + 1) % kVocab);
      for (int i = 0; i < 5; ++i) follow.push_back(std::int32_t(rng.next() % kVocab));
      const Out x = run_gen(follow, 24, follow_sp, {});
      run_gen(other, 2, sp, {});
      const Out z0 = run_gen(prompt, 0, sp, {});
      const Out z = run_gen(follow, 24, follow_sp, {});
      ok &= report(g.tokens.size() == std::size_t(c.g) && z0.tokens.empty() && z.stats.reuse == Reuse::Extend &&
                       x.stats.reuse == Reuse::Checkpoint && x.stats.reused_tokens == int(prompt.size()) &&
                       z.stats.reused_tokens == int(prompt.size()) && x.tokens == z.tokens,
                   "a follow-up restoring the prompt checkpoint equals the strict-prefix path" + at);
    }
    // the same prompt again after a generation: the checkpoint holds its state and last logits
    run_gen(other, 2, sp, {});
    run_gen(prompt, 30, sp, {});
    const Out retry = run_gen(prompt, 48, sp, {});
    ok &= report(retry.stats.reuse == Reuse::Checkpoint && retry.stats.reused_tokens == int(prompt.size()) &&
                     retry.tokens == ref.tokens,
                 "the same prompt again restores its checkpoint and generates the same tokens" + tag);
    // a request that fails mid-generation (on_tokens throws) leaves nothing reusable: the next request starts
    // from an empty state and generates what a fresh one does
    bool threw = false;
    int count = 0;
    try {
      gen->generate(prompt, 48, sp, {}, [&](std::span<const std::int32_t>) {
        if (++count == 5) throw std::runtime_error("client gone");
        return true;
      });
    } catch (const std::runtime_error&) {
      threw = true;
    }
    const Out after_failure = run_gen(prompt, 48, sp, {});
    ok &= report(threw && after_failure.stats.reuse == Reuse::None && after_failure.tokens == ref.tokens,
                 "after a failed request the next one starts from an empty state" + tag);
  }
  return ok;
}

}  // namespace

int main() {
  if (ingine::test::skip_without_gpu()) return ingine::test::kSkip;
  std::size_t free = 0, total = 0;
  INGINE_TEST_CUDA(cudaMemGetInfo(&free, &total));
  if (free < (std::size_t(7) << 28)) {  // peak use ~1.6 GB (the full-size unembedding is 0.46 GB)
    std::printf("skip: %.2f GB of GPU memory free, need 1.75\n", double(free) / (1 << 30));
    return ingine::test::kSkip;
  }
  bool ok = run(false);
  ok &= run(true);
  ok &= run_generator();
  std::printf("%s\n", ok ? "ALL PASS" : "SOME FAILED");
  return ok ? 0 : 1;
}
