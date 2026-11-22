// Drafter smoke test with the real mtp.safetensors (~1.2 GB of GPU memory at peak) and a small synthetic
// target (embedding/unembedding of 4096 tokens): eager and graph drafts agree bit for bit, commits through
// the per-row-count graphs equal eager commits, the outputs are well formed (tokens are each row's first
// maximum, -inf outside 1024 candidates, conf = 1 / sum exp(l - max)), sampled drafts are reproducible per
// seed, and the shortlist head proposes only its ranked tokens and the noted context tokens.
//
// INGINE_BENCH=1 runs a timing instead (about 4 GB of GPU memory): the round transition (Drafter::advance:
// small commit + draft) at the full vocabulary with random weights for the target's embedding and
// unembedding, at context lengths 2K / 20K / 70K, with the L2 flushed between rounds as a verify call does.
#include "common/test_util.h"
#include "kernels/dense/dense.h"
#include "kernels/draft/shortlist.h"
#include "model/drafter.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <vector>

using namespace ingine::test;
using ingine::DeviceBuffer;
using ingine::DType;
namespace model = ingine::model;

namespace {

constexpr int H = model::HIDDEN, V = 4096, MAX_CTX = 4096;
int g_failures = 0;

void expect(bool ok, const char* what) {
  std::printf("%s %s\n", ok ? "PASS" : "FAIL", what);
  if (!ok) ++g_failures;
}

struct DraftOut {
  std::vector<int32_t> tokens;
  std::vector<float> logits, conf;
  bool operator==(const DraftOut& o) const {
    return tokens == o.tokens && conf == o.conf &&
           std::equal(logits.begin(), logits.end(), o.logits.begin(), o.logits.end(),
                      [](float a, float b) { return a == b || (std::isinf(a) && std::isinf(b) && a == b); });
  }
};

DraftOut fetch(const model::Drafter::Draft& d, cudaStream_t s) {
  INGINE_TEST_CUDA(cudaStreamSynchronize(s));
  DraftOut o{std::vector<int32_t>(7), std::vector<float>(size_t(7) * V), std::vector<float>(7)};
  INGINE_TEST_CUDA(cudaMemcpy(o.tokens.data(), d.tokens, 7 * 4, cudaMemcpyDeviceToHost));
  INGINE_TEST_CUDA(cudaMemcpy(o.logits.data(), d.logits, size_t(7) * V * 4, cudaMemcpyDeviceToHost));
  INGINE_TEST_CUDA(cudaMemcpy(o.conf.data(), d.conf, 7 * 4, cudaMemcpyDeviceToHost));
  return o;
}

bool well_formed(const DraftOut& o, bool greedy) {
  bool ok = true;
  for (int i = 0; i < 7; ++i) {
    const float* row = o.logits.data() + size_t(i) * V;
    int finite = 0, first_max = -1;
    double m = -INFINITY, s = 0;
    for (int x = 0; x < V; ++x)
      if (std::isfinite(row[x])) {
        ++finite;
        if (row[x] > m) {
          m = row[x];
          first_max = x;
        }
      }
    for (int x = 0; x < V; ++x)
      if (std::isfinite(row[x])) s += std::exp(double(row[x]) - m);
    ok &= finite == 1024;
    ok &= o.tokens[i] >= 0 && o.tokens[i] < V && std::isfinite(row[o.tokens[i]]);
    if (greedy) ok &= o.tokens[i] == first_max;
    ok &= std::fabs(o.conf[i] - 1.0 / s) <= 1e-5 / s && o.conf[i] > 0.f && o.conf[i] <= 1.f;
  }
  return ok;
}

// Timing of the round transition at production sizes (see the file comment).
int bench(const std::filesystem::path& dir, cudaStream_t stream) {
  constexpr int BV = model::UNPADDED_VOCAB, BCTX = 262144, CHUNK = 1024, ROUNDS = 300;
  // BF16 normals from a table (a full-vocabulary matrix has 461M entries)
  std::vector<__nv_bfloat16> table(65536);
  Rng r(11);
  for (auto& x : table) x = float_to_bf16(float(r.normal()));
  std::uint64_t lcg = 0x243F6A8885A308D3ull;
  auto fill = [&](std::vector<__nv_bfloat16>& v) {
    for (auto& x : v) {
      lcg = lcg * 6364136223846793005ull + 1442695040888963407ull;
      x = table[lcg >> 48];
    }
  };
  model::Weights w;
  w.storage.reserve(4);
  {
    std::vector<__nv_bfloat16> host(std::size_t(BV) * H);
    fill(host);
    auto& e = w.storage.emplace_back(host.size() * 2);
    e.copy_from_host(host.data(), host.size() * 2);
    std::vector<float> en(H, 1.f);
    auto& n = w.storage.emplace_back(en.size() * 4);
    n.copy_from_host(en.data(), en.size() * 4);
    fill(host);
    DeviceBuffer u16(host.size() * 2);
    u16.copy_from_host(host.data(), host.size() * 2);
    auto& q = w.storage.emplace_back(host.size());
    auto& sc = w.storage.emplace_back(std::size_t(BV) * 4);
    ingine::kernels::dense::fp8_rows(static_cast<const __nv_bfloat16*>(u16.p), BV, H, q.p, static_cast<float*>(sc.p),
                                     nullptr);
    INGINE_TEST_CUDA(cudaDeviceSynchronize());
    w.embed = model::rm(w.storage[0].p, DType::BF16, {BV, H});
    w.embed_norm = model::rm(w.storage[1].p, DType::FP32, {H});
    w.unembed.w = model::rm(w.storage[2].p, DType::FP8_E4M3FN, {BV, H});
    w.unembed.scale = model::rm(w.storage[3].p, DType::FP32, {BV});
    w.dense_fp8 = true;
  }
  std::vector<__nv_bfloat16> feats(std::size_t(CHUNK) * 4 * H);
  fill(feats);
  DeviceArray<__nv_bfloat16> d_feats(feats);
  const std::size_t flush_bytes = std::size_t(256) << 20;  // more than the L2
  DeviceArray<uint8_t> junk(flush_bytes);
  DeviceArray<int32_t> d_m(std::vector<int32_t>{8}), d_acc(std::vector<int32_t>{2}), d_next(std::vector<int32_t>{99});

  model::Drafter d(dir, w, BCTX, stream);
  d.capture();
  ingine::kernels::draft::RoundInputs in;
  in.nwidths = 1;
  in.widths[0] = 8;
  in.taps[0] = d_feats.get();
  in.m = d_m.get();
  in.accepted = d_acc.get();
  in.next_greedy = d_next.get();
  in.next_sampled = d_next.get();
  d.bind_round(in);
  // context tokens beyond the shortlist, as a long prompt adds (the first 4000 ids outside it)
  {
    std::vector<std::uint8_t> ranked(BV, 0);
    int taken = 0;
    for (const std::uint32_t t : ingine::kernels::draft::draft_shortlist_ranks())
      if (taken < model::Drafter::SHORTLIST_ROWS && t < std::uint32_t(BV) && !ranked[t]) ranked[t] = 1, ++taken;
    std::vector<std::int32_t> extra;
    for (int t = 0; t < BV && extra.size() < 4000; ++t)
      if (!ranked[t]) extra.push_back(t);
    d.note_tokens(extra);
  }
  int len = 0;
  for (const int target : {2000, 20000, 70000}) {
    for (; len < target; len += CHUNK) d.commit(d_feats.get(), std::min(CHUNK, target - len), len);
    len = target;
    for (const float T : {0.f, 1.f}) {
      int P = len;
      d.draft(17, P, T);
      std::vector<cudaEvent_t> ev(2 * ROUNDS);
      for (auto& e : ev) INGINE_TEST_CUDA(cudaEventCreate(&e));
      for (int i = 0; i < ROUNDS; ++i) {
        INGINE_TEST_CUDA(cudaMemsetAsync(junk.get(), i, flush_bytes, stream));
        INGINE_TEST_CUDA(cudaEventRecord(ev[2 * i], stream));
        d.advance(P);
        INGINE_TEST_CUDA(cudaEventRecord(ev[2 * i + 1], stream));
        d.advanced(3);
        P += 3;
      }
      INGINE_TEST_CUDA(cudaStreamSynchronize(stream));
      std::vector<float> ms(ROUNDS);
      for (int i = 0; i < ROUNDS; ++i) INGINE_TEST_CUDA(cudaEventElapsedTime(&ms[i], ev[2 * i], ev[2 * i + 1]));
      for (auto& e : ev) cudaEventDestroy(e);
      std::sort(ms.begin(), ms.end());
      double mean = 0;
      for (int i = ROUNDS / 10; i < ROUNDS - ROUNDS / 10; ++i) mean += ms[i];  // trimmed: the GPU is shared
      mean /= double(ROUNDS - 2 * (ROUNDS / 10));
      std::printf("BENCH context %6d T %.0f: round transition %.4f ms (median %.4f, min %.4f)\n", len, double(T), mean,
                  ms[ROUNDS / 2], ms[0]);
      std::fflush(stdout);
      // back to the context of `len` rows: the rounds committed 3 * ROUNDS rows of the same taps
      len = P;
    }
  }
  return 0;
}

}  // namespace

int main() {
  if (skip_without_gpu()) return kSkip;
  const std::filesystem::path dir = INGINE_MODEL_DIR;
  if (!std::filesystem::exists(dir / "mtp.safetensors")) {
    std::printf("skip: %s/mtp.safetensors not found\n", dir.c_str());
    return kSkip;
  }
  size_t free_b = 0, total_b = 0;
  INGINE_TEST_CUDA(cudaMemGetInfo(&free_b, &total_b));
  if (free_b < (size_t(2) << 30)) {
    std::printf("skip: %.2f GB of GPU memory free, the test needs 2 GB\n", double(free_b) / (1 << 30));
    return kSkip;
  }
  cudaStream_t stream;
  INGINE_TEST_CUDA(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
  if (const char* on = std::getenv("INGINE_BENCH"); on && *on == '1') return bench(dir, stream);

  // synthetic target: embedding, its norm, FP8-row unembedding of V tokens
  Rng r(5);
  model::Weights w;
  {
    std::vector<__nv_bfloat16> embed(size_t(V) * H), unembed(size_t(V) * H);
    for (auto& x : embed) x = float_to_bf16(float(r.normal()));
    for (auto& x : unembed) x = float_to_bf16(float(r.normal() * 0.05));
    std::vector<float> en(H);
    for (auto& x : en) x = float(r.uniform(0.5, 1.5));
    w.storage.reserve(4);  // references below stay valid
    auto& e = w.storage.emplace_back(embed.size() * 2);
    e.copy_from_host(embed.data(), embed.size() * 2);
    auto& n = w.storage.emplace_back(en.size() * 4);
    n.copy_from_host(en.data(), en.size() * 4);
    DeviceBuffer u16(unembed.size() * 2);
    u16.copy_from_host(unembed.data(), unembed.size() * 2);
    auto& q = w.storage.emplace_back(unembed.size());
    auto& sc = w.storage.emplace_back(size_t(V) * 4);
    ingine::kernels::dense::fp8_rows(static_cast<const __nv_bfloat16*>(u16.p), V, H, q.p, static_cast<float*>(sc.p),
                                     nullptr);
    INGINE_TEST_CUDA(cudaDeviceSynchronize());
    w.embed = model::rm(w.storage[0].p, DType::BF16, {V, H});
    w.embed_norm = model::rm(w.storage[1].p, DType::FP32, {H});
    w.unembed.w = model::rm(w.storage[2].p, DType::FP8_E4M3FN, {V, H});
    w.unembed.scale = model::rm(w.storage[3].p, DType::FP32, {V});
    w.dense_fp8 = true;
  }

  const int rows = 300;
  std::vector<__nv_bfloat16> feats(size_t(rows) * 4 * H);
  for (auto& x : feats) x = float_to_bf16(float(r.normal()));
  DeviceBuffer d_feats(feats.size() * 2);
  d_feats.copy_from_host(feats.data(), feats.size() * 2);
  const auto* fp = static_cast<const __nv_bfloat16*>(d_feats.p);

  // the shortlist head over the 4096-token vocabulary: the first SHORT ranked ids; the others can join only as
  // context tokens
  constexpr int SHORT = 2048;
  std::vector<std::uint8_t> ranked(V, 0);
  std::vector<std::int32_t> outside;
  {
    int taken = 0;
    for (const std::uint32_t t : ingine::kernels::draft::draft_shortlist_ranks())
      if (taken < SHORT && t < std::uint32_t(V) && !ranked[t]) ranked[t] = 1, ++taken;
    for (int t = 0; t < V; ++t)
      if (!ranked[t]) outside.push_back(t);
  }

  {
    model::Drafter d(dir, w, MAX_CTX, stream, SHORT);
    const int anchor = 17;
    // A: one eager commit of all rows; B: rows - 5 rows, then 5 rows (a commit graph once captured)
    auto commit_a = [&] { d.commit(fp, rows, 0); };
    auto commit_b = [&] {
      d.commit(fp, rows - 5, 0);
      d.commit(fp + size_t(rows - 5) * 4 * H, 5, rows - 5);
    };
    commit_a();
    expect(d.length() == rows, "length after an eager commit");
    const DraftOut eager_a = fetch(d.draft(anchor, rows, 0.f), stream);
    expect(well_formed(eager_a, true), "eager greedy draft is well formed");
    commit_b();
    const DraftOut eager_b = fetch(d.draft(anchor, rows, 0.f), stream);
    float diff = 0.f;
    for (size_t i = 0; i < eager_a.logits.size(); ++i)
      if (std::isfinite(eager_a.logits[i]) && std::isfinite(eager_b.logits[i]))
        diff = std::max(diff, std::fabs(eager_a.logits[i] - eager_b.logits[i]));
    std::printf("one-shot vs split commit (cuBLASLt shapes differ): max logit diff %.4g, tokens %s\n", diff,
                eager_a.tokens == eager_b.tokens ? "equal" : "differ");

    d.capture();
    expect(d.length() == 0, "capture resets the context");
    commit_a();
    const DraftOut graph_a = fetch(d.draft(anchor, rows, 0.f), stream);
    expect(graph_a == eager_a, "graph draft == eager draft");
    commit_b();
    const DraftOut graph_b = fetch(d.draft(anchor, rows, 0.f), stream);
    expect(graph_b == eager_b, "draft after a commit graph == draft after the eager commit");
    const DraftOut& eager = eager_a;

    // advance (commit + draft from device values) == the host-driven commit + draft
    {
      const int widths[3] = {2, 4, 8}, m = 4, acc = 2, next = 99;
      std::vector<__nv_bfloat16> taps(size_t(8) * 4 * H);
      for (auto& x : taps) x = float_to_bf16(float(r.normal()));
      DeviceArray<__nv_bfloat16> d_taps(taps);
      DeviceArray<int32_t> d_m(std::vector<int32_t>{m}), d_acc(std::vector<int32_t>{acc}),
          d_next(std::vector<int32_t>{next}), d_other(std::vector<int32_t>{5});
      commit_a();
      d.commit(d_taps.get(), acc + 1, rows);
      const DraftOut host_path = fetch(d.draft(next, rows + acc + 1, 0.f), stream);
      commit_a();
      ingine::kernels::draft::RoundInputs in;
      in.nwidths = 3;
      for (int i = 0; i < 3; ++i) {
        in.widths[i] = widths[i];
        in.taps[i] = d_taps.get();  // one buffer of 8 rows serves every width here
      }
      in.m = d_m.get();
      in.accepted = d_acc.get();
      in.next_greedy = d_next.get();
      in.next_sampled = d_other.get();
      d.bind_round(in);
      d.draft(anchor, rows, 0.f);  // the round's draft: the verify call is at its position
      const DraftOut device_path = fetch(d.advance(rows), stream);
      d.advanced(acc + 1);
      expect(device_path == host_path, "advance == host-driven commit + draft");
      expect(d.length() == rows + acc + 1, "length after advance");
      commit_b();
    }

    // checkpoint at L = rows; later commits/drafts cross the 1024-slot local window (no wrap of the 4096-slot
    // global ring), then wrap the global ring too; a restore must give the same draft as before
    {
      DeviceArray<uint8_t> ckpt(d.checkpoint_bytes());
      commit_b();
      const DraftOut ref = fetch(d.draft(anchor, rows, 0.f), stream);
      d.save_checkpoint(ckpt.get());
      for (const int extra : {1500, 5000}) {
        for (int s = rows; s < rows + extra; s += rows) d.commit(fp, rows, s);
        d.note_tokens(outside);  // context tokens added after the checkpoint are dropped by the restore
        fetch(d.draft(anchor, rows + extra, 0.f), stream);
        d.restore_checkpoint(ckpt.get());
        char what[96];
        std::snprintf(what, sizeof(what), "restored checkpoint (after %d more positions) == draft at the checkpoint",
                      extra);
        expect(d.length() == rows && fetch(d.draft(anchor, rows, 0.f), stream) == ref, what);
      }
    }

    // conversation parking: the live regions of a checkpoint (with some context tokens) copied out, another
    // conversation (other context tokens, in another order) saves its checkpoint into the same buffer, the
    // parked regions copied back and adopted with their context tokens: the same draft as at the save
    {
      DeviceArray<uint8_t> ckpt(d.checkpoint_bytes()), parked(d.checkpoint_bytes());
      const std::vector<int32_t> mine(outside.begin(), outside.begin() + outside.size() / 2),
          theirs(outside.rbegin(), outside.rend());
      auto move = [&](uint8_t* to, const uint8_t* from) {
        for (const model::Drafter::Region& r : d.checkpoint_regions(rows))
          INGINE_TEST_CUDA(cudaMemcpyAsync(to + r.offset, from + r.offset, r.bytes, cudaMemcpyDeviceToDevice, stream));
      };
      d.reset();
      commit_b();
      d.note_tokens(mine);
      const DraftOut ref = fetch(d.draft(anchor, rows, 0.f), stream);
      d.save_checkpoint(ckpt.get());
      move(parked.get(), ckpt.get());
      const std::vector<int32_t> context = d.checkpoint_context();
      d.reset();
      commit_a();
      d.note_tokens(theirs);
      d.save_checkpoint(ckpt.get());
      fetch(d.draft(anchor, rows, 0.f), stream);
      move(ckpt.get(), parked.get());
      d.adopt_checkpoint(ckpt.get(), rows, context);
      d.restore_checkpoint(ckpt.get());
      expect(context == mine && d.length() == rows && fetch(d.draft(anchor, rows, 0.f), stream) == ref,
             "adopted parked checkpoint (other context tokens meanwhile) == draft at the save");
    }

    // the shortlist head: drafts come from the ranked tokens and the noted context tokens only
    {
      auto allowed_only = [&](const DraftOut& o, const std::vector<std::uint8_t>& ok) {
        for (std::size_t i = 0; i < o.logits.size(); ++i)
          if (std::isfinite(o.logits[i]) && !ok[i % V]) return false;
        return true;
      };
      expect(allowed_only(eager_a, ranked), "drafts come from the shortlist");
      d.note_tokens(outside);
      const DraftOut noted = fetch(d.draft(anchor, rows, 0.f), stream);
      std::vector<std::uint8_t> with = ranked;
      for (const int t : outside) with[t] = 1;
      bool uses = false;
      for (std::size_t i = 0; i < noted.logits.size(); ++i) uses |= std::isfinite(noted.logits[i]) && !ranked[i % V];
      expect(well_formed(noted, true) && allowed_only(noted, with) && uses, "noted context tokens join the candidates");
      d.reset();  // drops them again
      commit_b();
      expect(fetch(d.draft(anchor, rows, 0.f), stream) == eager_b, "reset drops the context tokens");
    }

    d.set_seed(123);
    const DraftOut s1 = fetch(d.draft(anchor, rows, 0.8f), stream);
    d.set_seed(123);
    const DraftOut s2 = fetch(d.draft(anchor, rows, 0.8f), stream);
    const DraftOut s3 = fetch(d.draft(anchor, rows, 0.8f), stream);
    expect(well_formed(s1, false), "sampled draft is well formed");
    expect(s1 == s2, "sampled drafts reproduce per seed");
    expect(std::equal(s1.logits.begin(), s1.logits.begin() + V, s3.logits.begin()),
           "step 0 logits do not depend on the noise");
    std::printf("greedy drafts:");
    for (int t : eager.tokens) std::printf(" %d", t);
    std::printf("\nsampled drafts:");
    for (int t : s1.tokens) std::printf(" %d", t);
    std::printf(" / next draw:");
    for (int t : s3.tokens) std::printf(" %d", t);
    std::printf("\n");
  }
  INGINE_TEST_CUDA(cudaStreamDestroy(stream));
  std::printf("%s: %d failure(s)\n", g_failures ? "FAILED" : "OK", g_failures);
  return g_failures ? 1 : 0;
}
