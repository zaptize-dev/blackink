// Prompt checkpoints on the full model with speculative decoding (DSpark drafter + verify graphs): a follow-up
// that restores the checkpoint of its prompt prefix generates exactly what the strict-prefix path generates
// (same prefill chunk boundaries), greedy and sampled, with the generation between them longer than the
// 1024-slot rings and crossing drafter context buckets. Needs the model (~22 GB of GPU memory).
#include "common/test_util.h"
#include "core/arena.h"
#include "model/runtime.h"

#include "ingine/generator.h"

#include <cstdio>
#include <filesystem>
#include <string>
#include <vector>

using namespace ingine::model;

int main() {
  if (ingine::test::skip_without_gpu()) return ingine::test::kSkip;
  const std::filesystem::path dir = INGINE_MODEL_DIR;
  if (!std::filesystem::exists(dir / "mtp.safetensors")) {
    std::printf("skip: no model at %s\n", dir.c_str());
    return ingine::test::kSkip;
  }
  RuntimeOptions opt;
  opt.model_dir = dir;
  opt.max_ctx = 8192;  // drafter buckets 1024, 2048, 4096, 8192
  opt.dense_fp8 = opt.kv_fp8 = opt.spec = true;
  opt.prefill_chunk = 512;
  opt.park_mb = 0;  // the checkpoint tests below see no parked conversations
  auto gen = make_generator(opt);

  ingine::test::Rng rng(3);
  // a prompt of ordinary token ids (no special tokens), 900 long: generation crosses the 1024 bucket
  auto random_ids = [&](int n) {
    std::vector<std::int32_t> v(static_cast<std::size_t>(n));
    for (auto& t : v) t = std::int32_t(1000 + rng.next() % 50000);
    return v;
  };
  const std::vector<std::int32_t> prompt = random_ids(900), other = random_ids(300);
  struct Out {
    std::vector<std::int32_t> tokens;
    GenerateStats stats;
  };
  auto run = [&](const std::vector<std::int32_t>& ids, int max_new, const Sampling& sp) {
    Out o;
    o.stats = gen->generate(ids, max_new, sp, {}, [&](std::span<const std::int32_t> t) {
      o.tokens.insert(o.tokens.end(), t.begin(), t.end());
      return true;
    });
    return o;
  };
  bool ok = true;
  Sampling greedy;
  greedy.temperature = 0.f;
  Sampling sampled;
  sampled.temperature = 1.f;
  sampled.seed = 5;
  for (const auto& [name, sp] : {std::pair{"greedy", greedy}, std::pair{"sampled", sampled}}) {
    Sampling follow_sp = sp;
    follow_sp.seed = 9;
    for (const int g_len : {200, 1300}) {
      run(other, 4, sp);
      const Out g = run(prompt, g_len, sp);
      std::vector<std::int32_t> follow = prompt;
      follow.insert(follow.end(), g.tokens.begin(), g.tokens.begin() + 2);
      const std::vector<std::int32_t> y = random_ids(40);
      follow.insert(follow.end(), y.begin(), y.end());
      const Out x = run(follow, 64, follow_sp);
      run(other, 4, sp);
      run(prompt, 0, sp);
      const Out z = run(follow, 64, follow_sp);
      const bool good = int(g.tokens.size()) == g_len && x.stats.reuse == Reuse::Checkpoint &&
                        z.stats.reuse == Reuse::Extend && x.stats.reused_tokens == int(prompt.size()) &&
                        z.stats.reused_tokens == int(prompt.size()) && x.tokens == z.tokens;
      std::printf("%s checkpoint restore equals the strict-prefix path with spec (%s, G %d tokens): %zu tokens\n",
                  good ? "PASS" : "FAIL", name, g_len, x.tokens.size());
      ok &= good;
    }
  }
  gen.reset();

  // Conversation parking: A (prompt, generate), B (unrelated prompt), A's follow-up, B's follow-up. The
  // follow-ups restore their parked conversation (only the new tokens prefilled) and generate what they
  // generate on a generator that never saw the other conversation; the follow-ups re-render the answer
  // differently from the generated tokens (the Checkpoint case). With room for one conversation (~200 MB
  // each), C after B evicts A; with room for none nothing is parked; the outputs stay the same.
  const std::vector<std::int32_t> a = random_ids(1500), b = random_ids(700), c = random_ids(400);
  auto follow_up = [&](const std::vector<std::int32_t>& p, const Out& o) {
    std::vector<std::int32_t> f = p;
    f.insert(f.end(), o.tokens.begin(), o.tokens.begin() + 2);
    const std::vector<std::int32_t> y = random_ids(40);
    f.insert(f.end(), y.begin(), y.end());
    return f;
  };
  for (const bool spec : {false, true}) {
    opt.spec = spec;
    const char* tag = spec ? "with spec" : "without spec";
    // the reference: each conversation alone
    opt.park_mb = 0;
    gen = make_generator(opt);
    const Out ra = run(a, 300, greedy), rb = run(b, 300, greedy);
    const std::vector<std::int32_t> fa = follow_up(a, ra), fb = follow_up(b, rb);
    run(a, 300, greedy);
    const Out rfa = run(fa, 64, greedy);
    run(b, 300, greedy);
    const Out rfb = run(fb, 64, greedy);
    // a follow-up without its conversation prefills everything (other chunk boundaries than the restore path)
    run(c, 8, greedy);
    const Out nfa = run(fa, 64, greedy);
    run(c, 8, greedy);
    const Out nfb = run(fb, 64, greedy);
    gen.reset();
    for (const int mb : {2048, 300, 40}) {
      opt.park_mb = mb;
      gen = make_generator(opt);
      const Out oa = run(a, 300, greedy), ob = run(b, 300, greedy);
      Out ofa, ofb;
      if (mb == 300) {
        run(c, 8, greedy);  // parks B, evicting A
        ofb = run(fb, 64, greedy);
        ofa = run(fa, 64, greedy);
      } else {
        ofa = run(fa, 64, greedy);
        ofb = run(fb, 64, greedy);
      }
      gen.reset();
      const Reuse want_a = mb == 2048 ? Reuse::Parked : Reuse::None;
      const Reuse want_b = mb == 40 ? Reuse::None : Reuse::Parked;
      const bool same = ofa.tokens == (want_a == Reuse::Parked ? rfa : nfa).tokens &&
                        ofb.tokens == (want_b == Reuse::Parked ? rfb : nfb).tokens;
      const bool good = oa.tokens == ra.tokens && ob.tokens == rb.tokens && same && ofa.stats.reuse == want_a &&
                        ofb.stats.reuse == want_b &&
                        (want_a != Reuse::Parked || ofa.stats.reused_tokens == int(a.size())) &&
                        (want_b != Reuse::Parked || ofb.stats.reused_tokens == int(b.size()));
      std::printf("%s interleaved conversations, %d MiB parking %s: follow-ups reuse %s %d / %s %d, outputs %s\n",
                  good ? "PASS" : "FAIL", mb, tag, to_string(ofa.stats.reuse), ofa.stats.reused_tokens,
                  to_string(ofb.stats.reuse), ofb.stats.reused_tokens, same ? "equal" : "DIFFER");
      ok &= good;
    }
    // memory pressure: a device allocation inside a request (plain decoding's forced-token buffer) fails once
    // while A and B are parked: they are released, the allocation retried, the request completes unchanged; B,
    // released, is prefilled again
    if (!spec) {
      opt.park_mb = 2048;
      gen = make_generator(opt);
      run(a, 300, greedy);
      run(b, 300, greedy);
      run(fa, 64, greedy);  // parks B, restores A (still parked too)
      Sampling forced = greedy;
      forced.forced = rfa.tokens;
      ingine::fail_next_allocations(1);
      const Out ofa = run(fa, 64, forced);  // the checkpoint of fa: nothing to park before the allocation
      const Out ofb = run(fb, 64, greedy);
      gen.reset();
      const bool good = ofa.stats.reuse == Reuse::Checkpoint && ofa.tokens == rfa.tokens &&
                        ofb.stats.reuse == Reuse::None && ofb.tokens == nfb.tokens;
      std::printf("%s a failed allocation releases the parked conversations and the request completes (reuse %s, "
                  "then %s)\n",
                  good ? "PASS" : "FAIL", to_string(ofa.stats.reuse), to_string(ofb.stats.reuse));
      ok &= good;
    }
  }
  std::printf("%s\n", ok ? "ALL PASS" : "SOME FAILED");
  return ok ? 0 : 1;
}
