// Bitwise lock of the weight-streaming and attention decode kernels for every call width M = 1..8 (the dense
// mix / message_gates / router locks are in test_dense.cpp): proj, proj_conv and head (BF16 and FP8 rows),
// moe_up and moe_down (NVFP4 and MXFP8, experts shared between rows), and sliding-window attention. A verify
// call of M rows must keep giving the bits it gave when these hashes were recorded: on fixed inputs every
// output byte (activations, conv states and histories, cache rows, ring positions) is hashed and compared
// with kLock. A changed hash means the arithmetic changed (order of a sum, an FMA contraction, a rounding
// point). Either undo that, or, when the change is deliberate and qualified against the model
// (ingine-validate), record the new hashes this test prints.
#include "kernels/oracle.h"

#include "kernels/attn/attn.h"
#include "kernels/dense/dense.h"
#include "kernels/moe/moe.h"

#include <cuda_runtime.h>

#include <numeric>

using namespace ingine::test;
namespace attn = ingine::kernels::attn;
namespace dense = ingine::kernels::dense;
namespace moe = ingine::kernels::moe;

namespace {

constexpr int H = 2304, I = 1152, TOPK = 8;
cudaStream_t g_stream = nullptr;

void sync() { INGINE_TEST_CUDA(cudaStreamSynchronize(g_stream)); }

struct LockGen {  // integer generator, exact float conversion: the inputs do not depend on libm
  uint64_t s;
  uint64_t next() {
    s += 0x9e3779b97f4a7c15ull;
    uint64_t z = s;
    z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ull;
    z = (z ^ (z >> 27)) * 0x94d049bb133111ebull;
    return z ^ (z >> 31);
  }
  float unit() { return float(int32_t(next() >> 40) - (1 << 23)) * (1.f / float(1 << 23)); }  // [-1, 1)
  std::vector<float> vec(size_t n, float scale, float offset = 0.f) {
    std::vector<float> v(n);
    for (auto& x : v) x = unit() * scale + offset;
    return v;
  }
  std::vector<float> vec_bf16(size_t n, float scale, float offset = 0.f) {
    std::vector<float> v = vec(n, scale, offset);
    for (auto& x : v) x = round_bf16(x);
    return v;
  }
  std::vector<uint8_t> bytes(size_t n) {
    std::vector<uint8_t> v(n);
    for (size_t i = 0; i < n; i += 8) {
      const uint64_t r = next();
      std::memcpy(&v[i], &r, std::min<size_t>(8, n - i));
    }
    return v;
  }
};

struct Fnv {
  uint64_t h = 1469598103934665603ull;
  template <class T>
  void add(const std::vector<T>& v) {
    const unsigned char* b = reinterpret_cast<const unsigned char*>(v.data());
    for (size_t i = 0; i < v.size() * sizeof(T); ++i) h = (h ^ b[i]) * 1099511628211ull;
  }
};

// ---- dense GEMVs --------------------------------------------------------------------------------------------

// A weight matrix [rows, K] as BF16 rows and as E4M3 rows with per-row scales.
struct DenseWeight {
  int rows, K;
  DeviceArray<__nv_bfloat16> bf16;
  DeviceArray<uint8_t> fp8;
  DeviceArray<float> scale;
  DenseWeight(LockGen& g, int rows_, int K_)
      : rows(rows_), K(K_), bf16(to_bf16(g.vec_bf16(size_t(rows_) * K_, 0.05f))), fp8(codes(g, size_t(rows_) * K_)),
        scale(g.vec(size_t(rows_), 2e-4f, 4e-4f)) {}
  static std::vector<uint8_t> codes(LockGen& g, size_t n) {
    std::vector<uint8_t> c = g.bytes(n);
    for (auto& b : c)
      if ((b & 0x7f) == 0x7f) b ^= 1;  // no NaN
    return c;
  }
  dense::Matrix matrix(bool f8) const {
    return f8 ? dense::Matrix{fp8.get(), scale.get(), rows} : dense::Matrix{bf16.get(), nullptr, rows};
  }
};

// proj segment tables of the decode step: (row_begin, row_end, input)
std::vector<int> segments(bool full, bool single) {
  const int begin[9] = {0, 3072, 6144, 6528, 6552, 7576, 8600, 8632, 8664};
  const int input[8] = {0, 0, 0, 0, 1, 2, 1, 2};
  std::vector<int> t;
  for (int s = 0; s < (full ? 8 : 6); ++s) {
    t.push_back(begin[s]);
    t.push_back(begin[s + 1]);
    t.push_back(single ? 0 : input[s]);
  }
  return t;
}

uint64_t lock_proj(LockGen& g, const DenseWeight& w, int M) {
  Fnv f;
  DeviceArray<__nv_bfloat16> xin(to_bf16(g.vec_bf16(size_t(3) * M * H, 1.5f)));
  for (int f8 = 0; f8 < 2; ++f8)
    for (int shape = 0; shape < 3; ++shape) {  // sliding single input, sliding three inputs, full
      const bool full = shape == 2;
      const std::vector<int> seg = segments(full, shape == 0);
      const int R = full ? 8664 : 8600;
      DeviceArray<int> dseg(seg);
      DeviceArray<float> out(size_t(M) * R);
      dense::Matrix m = w.matrix(f8);
      m.rows = R;
      dense::proj(xin.get(), M, m, dseg.get(), int(seg.size() / 3), out.get(), true, g_stream);
      sync();
      f.add(out.download());
    }
  return f.h;
}

uint64_t lock_proj_conv(LockGen& g, const DenseWeight& w, int M) {
  Fnv f;
  DeviceArray<__nv_bfloat16> x(to_bf16(g.vec_bf16(size_t(M) * 3072, 1.5f)));
  DeviceArray<float> cw(g.vec(size_t(H) * 4, 0.5f));
  const std::vector<float> st0 = g.vec_bf16(size_t(3) * H, 0.3f);
  for (int f8 = 0; f8 < 2; ++f8) {
    DeviceArray<float> st(st0), out(size_t(M) * H), hist(size_t(M) * H);
    dense::proj_conv(x.get(), M, w.matrix(f8), cw.get(), st.get(), out.get(), hist.get(), true, g_stream);
    sync();
    f.add(out.download()), f.add(hist.download()), f.add(st.download());
  }
  return f.h;
}

uint64_t lock_head(LockGen& g, const DenseWeight& w, int M) {
  Fnv f;
  DeviceArray<__nv_bfloat16> x(to_bf16(g.vec_bf16(size_t(M) * H, 1.5f)));
  for (int f8 = 0; f8 < 2; ++f8) {
    dense::Matrix m = w.matrix(f8);
    m.rows = 5003;  // a last block with fewer than 16 rows
    DeviceArray<float> out(size_t(M) * m.rows);
    dense::head(x.get(), M, m, out.get(), true, g_stream);
    sync();
    f.add(out.download());
  }
  return f.h;
}

// ---- routed experts -----------------------------------------------------------------------------------------

constexpr int E = 40;  // experts allocated (routing stays below E)

struct ExpertWeight {  // [E, rows, cols] in checkpoint encoding
  DeviceArray<uint8_t> codes, scales;
  DeviceArray<float> factor;
  ExpertWeight(LockGen& g, int rows, int cols, bool nv)
      : codes(make_codes(g, size_t(E) * rows * cols / (nv ? 2 : 1), nv)),
        scales(make_scales(g, size_t(E) * rows * cols / (nv ? 16 : 32), nv)), factor(g.vec(128, 1e-4f, 3e-4f)) {}
  static std::vector<uint8_t> make_codes(LockGen& g, size_t n, bool nv) {
    std::vector<uint8_t> c = g.bytes(n);
    if (!nv)
      for (auto& b : c)
        if ((b & 0x7f) == 0x7f) b ^= 1;  // no NaN
    return c;
  }
  static std::vector<uint8_t> make_scales(LockGen& g, size_t n, bool nv) {
    std::vector<uint8_t> s = g.bytes(n);
    for (auto& b : s) b = nv ? uint8_t(0x48 + b % 0x30) /* E4M3 1..54 */ : uint8_t(112 + b % 6) /* E8M0 */;
    return s;
  }
};

// 8 distinct experts per row; a row keeps 3 of the previous row's experts (real adjacent rows share ~3)
std::vector<int> routing(LockGen& g, int M) {
  std::vector<int> idx, prev;
  for (int m = 0; m < M; ++m) {
    std::vector<int> cur;
    for (int k = 0; k < 3 && m > 0; ++k) {
      const int e = prev[g.next() % TOPK];
      if (std::find(cur.begin(), cur.end(), e) == cur.end()) cur.push_back(e);
    }
    while (int(cur.size()) < TOPK) {
      const int e = int(g.next() % E);
      if (std::find(cur.begin(), cur.end(), e) == cur.end()) cur.push_back(e);
    }
    for (int i = TOPK - 1; i > 0; --i) std::swap(cur[i], cur[g.next() % (i + 1)]);
    idx.insert(idx.end(), cur.begin(), cur.end());
    prev = cur;
  }
  return idx;
}

uint64_t lock_moe(LockGen& g, const ExpertWeight& w13, const ExpertWeight& w2, bool nv, int M) {
  Fnv f;
  const moe::Format fmt = nv ? moe::Format::NVFP4 : moe::Format::MXFP8;
  DeviceArray<int> idx(routing(g, M));
  DeviceArray<__nv_bfloat16> x(to_bf16(g.vec_bf16(size_t(M) * H, 1.5f))), act(size_t(M) * TOPK * I);
  moe::moe_up(x.get(), M, idx.get(), w13.codes.get(), w13.scales.get(), nv ? w13.factor.get() : nullptr, fmt, act.get(),
              true, g_stream);
  sync();
  f.add(act.download());
  DeviceArray<__nv_bfloat16> act2(to_bf16(g.vec_bf16(size_t(M) * TOPK * I, 0.75f)));
  DeviceArray<float> wts(g.vec(size_t(M) * TOPK, 0.9f, 1.1f)), cw(g.vec(size_t(H) * 4, 0.5f));
  DeviceArray<float> st(g.vec_bf16(size_t(3) * H, 0.3f)), out(size_t(M) * H), hist(size_t(M) * H);
  moe::moe_down(act2.get(), M, idx.get(), wts.get(), w2.codes.get(), w2.scales.get(), nv ? w2.factor.get() : nullptr,
                fmt, cw.get(), st.get(), out.get(), hist.get(), true, g_stream);
  sync();
  f.add(out.download()), f.add(hist.download()), f.add(st.download());
  return f.h;
}

// ---- sliding-window attention -------------------------------------------------------------------------------

constexpr int HQ = 24, D = 128, DREL = 16, EXT = 1024, WIN = 1024, C = 8 * D, R_SLIDING = 8600;
constexpr int OFF_REL = 6144, OFF_SINK = 6528, OFF_K = 6552;

// Ring states: 0 = partially filled (p0 keys in order), 1 = wrapped, 2 = wrapped with stale entries at
// positions >= p0 (rows a rejected verify call left behind).
uint64_t lock_attn_sliding(LockGen& g, int M) {
  Fnv f;
  DeviceArray<float> qs(g.vec_bf16(size_t(HQ) * D, 0.3f, 1.f)), rs(g.vec_bf16(size_t(HQ) * DREL, 0.3f, 1.f));
  DeviceArray<float> rp(g.vec_bf16(size_t(DREL) * EXT, 0.5f)), kc(g.vec_bf16(size_t(4) * C, 0.3f)),
      vc(g.vec_bf16(size_t(4) * C, 0.3f));
  DeviceArray<uint8_t> ws(attn::workspace_bytes(8));
  struct Ring {
    int p0, kind;
  };
  for (const Ring ring : {Ring{300, 0}, Ring{1021, 0}, Ring{5000, 1}, Ring{70003, 2}}) {
    const int p0 = ring.p0;
    std::vector<float> proj = g.vec_bf16(size_t(M) * R_SLIDING, 1.f);
    for (int m = 0; m < M; ++m) {
      for (int r = OFF_REL; r < OFF_SINK; ++r) proj[size_t(m) * R_SLIDING + r] *= 3.f;
      for (int r = OFF_SINK; r < OFF_K; ++r) proj[size_t(m) * R_SLIDING + r] *= 2.f;
    }
    std::vector<int64_t> kpos(WIN, -1);
    if (ring.kind == 0) {
      for (int s = 0; s < p0; ++s) kpos[s] = s;
    } else {
      for (int s = 0; s < WIN; ++s) kpos[s] = p0 - 1 - ((p0 - 1 - s) % WIN + WIN) % WIN;
      if (ring.kind == 2)
        for (int i = 0; i < 6; ++i) kpos[(p0 + i) % WIN] = p0 + i;
    }
    DeviceArray<float> dproj(proj), ck(g.vec_bf16(size_t(3) * C, 1.f)), cv(g.vec_bf16(size_t(3) * C, 1.f));
    DeviceArray<__nv_bfloat16> k(to_bf16(g.vec_bf16(size_t(WIN) * C, 1.6f))), v(to_bf16(g.vec_bf16(size_t(WIN) * C, 1.7f)));
    DeviceArray<int64_t> dkpos(kpos), dpos(std::vector<int64_t>{p0});
    DeviceArray<__nv_bfloat16> out(size_t(M) * HQ * D);
    attn::DecodeArgs a;
    a.proj = dproj.get();
    a.M = M;
    a.q_scale = qs.get();
    a.r_scale = rs.get();
    a.rel_proj = rp.get();
    a.k_conv = kc.get();
    a.v_conv = vc.get();
    a.conv_k_state = ck.get();
    a.conv_v_state = cv.get();
    a.k_cache = k.get();
    a.v_cache = v.get();
    a.cache_fp8 = false;
    a.cache_rows = WIN;
    a.kpos = dkpos.get();
    a.pos = dpos.get();
    a.window = WIN;
    a.log_scaling = false;
    a.workspace = ws.get();
    a.workspace_size = ws.count;
    a.out = out.get();
    attn::attn_decode(a, g_stream);
    sync();
    f.add(out.download()), f.add(ck.download()), f.add(cv.download()), f.add(k.download()), f.add(v.download());
    f.add(dkpos.download());
  }
  return f.h;
}

// [M - 1][proj, proj_conv, head, moe nvfp4, moe mxfp8, sliding attention]
constexpr int kLocks = 6;
constexpr const char* kNames[kLocks] = {"proj", "proj_conv", "head", "moe nvfp4", "moe mxfp8", "attn sliding"};
constexpr uint64_t kLock[8][kLocks] = {
    {0xeaaffb36e3fef998ull, 0x790e36f18198ffb7ull, 0xb70817049b1d2fa3ull, 0xadb48f6d3cd579e9ull, 0x1434a3aac7348d9dull, 0x2a8f3dcf4b8c8d5bull},
    {0x6afd97ec2ce8db9full, 0x73631448743ab916ull, 0xbe6e8620ec2d5c91ull, 0x2a7fc99d95ff23baull, 0x9b208d3fb3498c1bull, 0xef4b0972e5ea3244ull},
    {0x441529aab368d641ull, 0x955503a16b61b392ull, 0x2647ab757a753f8full, 0x8497967345a92287ull, 0xd2168560902406fdull, 0xe4cbaf0c2d0c7bc2ull},
    {0x2165f0ff57d80238ull, 0x3b764a0b58871673ull, 0x52cece1223da1753ull, 0x13678cfce17b592bull, 0xbb4b2b7b74284f36ull, 0xfc4ee3f74f7f7970ull},
    {0xc7e8b065cd281619ull, 0x8def6144b8d1a6ddull, 0xf728fa27f51a6addull, 0xf8f5d56151931c89ull, 0xd2b386c363d4d43dull, 0x3b08c7121c1f7f30ull},
    {0xa916f12be963aee1ull, 0x24ff175b36d98597ull, 0xff957e3c72a1220eull, 0x8400a007229514c9ull, 0xfd6d7163fa91de54ull, 0xc5c4d8fd1dd6de9eull},
    {0xf940f814b1aeab52ull, 0x5acd1dc5543cd309ull, 0x3bb8e61b7b123ee1ull, 0xfa4d0561c1ae8319ull, 0x2406078539f39eabull, 0x60c2fcfa9e617660ull},
    // attn sliding differs from 32489d5: that M = 8 build left 2 of 6 bias adds unfused; pinned to fmaf
    {0x96acab9db3688209ull, 0x0860e87ad1181b6bull, 0xed4e557d3be4bc9full, 0x40873edc05b60326ull, 0x1ead80b992b2cc6aull, 0xeb11175f858ef93eull},
};

}  // namespace

int main() {
  if (skip_without_gpu()) return kSkip;
  INGINE_TEST_CUDA(cudaStreamCreateWithFlags(&g_stream, cudaStreamNonBlocking));
  bool ok = true;
  uint64_t got[8][kLocks];
  {
    LockGen gw{0x10c50000ull};
    const DenseWeight wproj(gw, 8664, H), wo(gw, H, 3072);
    for (int M = 1; M <= 8; ++M) {
      LockGen g{0x10c51000ull + uint64_t(M)};
      got[M - 1][0] = lock_proj(g, wproj, M);
      got[M - 1][1] = lock_proj_conv(g, wo, M);
      got[M - 1][2] = lock_head(g, wproj, M);
    }
  }
  for (int nv = 1; nv >= 0; --nv) {
    LockGen gw{0x10c52000ull + uint64_t(nv)};
    const ExpertWeight w13(gw, 2 * I, H, nv), w2(gw, H, I, nv);
    for (int M = 1; M <= 8; ++M) {
      LockGen g{0x10c53000ull + uint64_t(8 * nv + M)};
      got[M - 1][nv ? 3 : 4] = lock_moe(g, w13, w2, nv, M);
    }
  }
  for (int M = 1; M <= 8; ++M) {
    LockGen g{0x10c54000ull + uint64_t(M)};
    got[M - 1][5] = lock_attn_sliding(g, M);
    LockGen g2{0x10c54000ull + uint64_t(M)};
    if (lock_attn_sliding(g2, M) != got[M - 1][5]) {
      std::printf("FAIL attn sliding M=%d: two runs differ\n", M);
      ok = false;
    }
  }
  for (int M = 1; M <= 8; ++M) {
    bool same = true;
    std::string diff;
    for (int i = 0; i < kLocks; ++i)
      if (got[M - 1][i] != kLock[M - 1][i]) {
        same = false;
        diff += std::string(diff.empty() ? " differs: " : ", ") + kNames[i];
      }
    std::printf("%s bit lock M=%d: {", same ? "PASS" : "FAIL", M);
    for (int i = 0; i < kLocks; ++i)
      std::printf("%s0x%016llxull", i ? ", " : "", static_cast<unsigned long long>(got[M - 1][i]));
    std::printf("},%s\n", diff.c_str());
    ok = ok && same;
  }
  INGINE_TEST_CUDA(cudaStreamDestroy(g_stream));
  std::printf("%s\n", ok ? "ALL PASS" : "SOME FAILED");
  return ok ? 0 : 1;
}
