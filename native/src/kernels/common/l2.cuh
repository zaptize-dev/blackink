// L2 cache policies for device loads, copies and stores (PTX cache hints).
//
// A decode step streams about 3.8 GB of weights through the L2 (about 96 MB on the RTX 5090: re-reading a
// buffer runs at L2 speed up to 95 MB and at DRAM speed from 111 MB), so with the default policy everything
// read once per token has been evicted by its next use and comes from DRAM again. A hint sets the eviction
// class of the L2 lines an access fills:
// * Evict::last for small data re-read every token: its lines go last, so it stays resident through the
//   streams, as long as all evict-last data together fits. Measured (3 MB pieces, each re-read once per
//   token, 148 MB of stream between two pieces): a 24..72 MB set reads at the hot-cache time, a 96 MB set at
//   twice the time it takes without any hint. Keep the evict-last data of a model well below the L2 size; it
//   also stays there when the kernels that marked it no longer run. Residency is not a gain by itself: with
//   the gates' and router's weights (70 MB) marked evict-last they stayed in the L2, and the decode step was
//   about 2% slower (their fetch already overlaps the previous kernel; everything else had 25 MB left).
//   Without hints, data re-read after a large stream only stays if it is small (24 MiB did, 40 MiB did not).
// * Evict::first for big streams read once per token: their lines are the first victims, which protects
//   everything else (activations, state, the evict-last set).
// * Evict::normal is the default class; Evict::none issues the access without a hint.
// The policy is a 64-bit operand made once per thread by policy<E>() and passed to each access.
// discard_line drops dead lines without a write-back; prefetch requests a line without waiting.
//
// Requires the CUDA 13.4 ptxas (13.1 left the hint operand of some fills uninitialized); older toolkits and
// INGINE_L2_HINTS=0 compile every access without hints.
#pragma once

#include <cuda_runtime.h>

#include <cstdint>

#ifndef INGINE_L2_HINTS
#if __CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 4)
#define INGINE_L2_HINTS 1
#else
#define INGINE_L2_HINTS 0
#endif
#endif

namespace ingine::kernels::l2 {

enum class Evict { none, first, normal, last };

template <Evict E>
constexpr bool hinted = INGINE_L2_HINTS && E != Evict::none;

// The access policy of class E for whole lines (0 and unused for Evict::none).
template <Evict E>
__device__ __forceinline__ std::uint64_t policy() {
  unsigned long long p = 0;
  if constexpr (hinted<E>) {
    if constexpr (E == Evict::first) asm("createpolicy.fractional.L2::evict_first.b64 %0, 1.0;" : "=l"(p));
    else if constexpr (E == Evict::last) asm("createpolicy.fractional.L2::evict_last.b64 %0, 1.0;" : "=l"(p));
    else asm("createpolicy.fractional.L2::evict_normal.b64 %0, 1.0;" : "=l"(p));
  }
  return p;
}

// 16-byte load of read-only data (weights, constants) on the non-coherent path without L1 allocation.
// `volatile`: the load is issued where it is written (ahead of a later griddepcontrol.wait with
// pdl_wait_fenced).
template <Evict E>
__device__ __forceinline__ uint4 ld_stream(const void* p, std::uint64_t pol) {
  uint4 r;
  if constexpr (hinted<E>) {
    asm volatile("ld.global.nc.L1::no_allocate.L2::cache_hint.v4.u32 {%0,%1,%2,%3}, [%4], %5;"
                 : "=r"(r.x), "=r"(r.y), "=r"(r.z), "=r"(r.w)
                 : "l"(p), "l"(pol));
  } else {
    asm volatile("ld.global.nc.L1::no_allocate.v4.u32 {%0,%1,%2,%3}, [%4];"
                 : "=r"(r.x), "=r"(r.y), "=r"(r.z), "=r"(r.w)
                 : "l"(p));
  }
  return r;
}

// 16-byte asynchronous global -> shared copy (L2 only); complete with cp.async.commit_group / wait.
template <Evict E>
__device__ __forceinline__ void cp_async16(void* smem, const void* gmem, std::uint64_t pol) {
  const unsigned dst = static_cast<unsigned>(__cvta_generic_to_shared(smem));
  if constexpr (hinted<E>) {
    asm volatile("cp.async.cg.shared.global.L2::cache_hint [%0], [%1], 16, %2;" ::"r"(dst), "l"(gmem), "l"(pol)
                 : "memory");
  } else {
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16;" ::"r"(dst), "l"(gmem) : "memory");
  }
}

// 16-byte store whose L2 line gets class E (e.g. Evict::last for a result read after a large stream).
template <Evict E>
__device__ __forceinline__ void st16(void* p, uint4 v, std::uint64_t pol) {
  if constexpr (hinted<E>) {
    asm volatile("st.global.L2::cache_hint.v4.b32 [%0], {%1,%2,%3,%4}, %5;" ::"l"(p), "r"(v.x), "r"(v.y), "r"(v.z),
                 "r"(v.w), "l"(pol)
                 : "memory");
  } else {
    asm volatile("st.global.v4.b32 [%0], {%1,%2,%3,%4};" ::"l"(p), "r"(v.x), "r"(v.y), "r"(v.z), "r"(v.w) : "memory");
  }
}

// Invalidates the 128-byte L2 line at `line` without writing it back. Only for lines whose data is dead:
// every later read of the line must follow a new write.
__device__ __forceinline__ void discard_line(const void* line) {
#if INGINE_L2_HINTS
  asm volatile("discard.global.L2 [%0], 128;" ::"l"(line) : "memory");
#else
  static_cast<void>(line);
#endif
}

// Requests the L2 line holding `p` without waiting for it (Evict::last or Evict::normal class; any other E
// requests it with the default class).
template <Evict E>
__device__ __forceinline__ void prefetch(const void* p) {
  if constexpr (hinted<E> && E == Evict::last) asm volatile("prefetch.global.L2::evict_last [%0];" ::"l"(p));
  else if constexpr (hinted<E> && E == Evict::normal) asm volatile("prefetch.global.L2::evict_normal [%0];" ::"l"(p));
  else asm volatile("prefetch.global.L2 [%0];" ::"l"(p));
}

}  // namespace ingine::kernels::l2
