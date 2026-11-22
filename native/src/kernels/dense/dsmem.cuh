// One-way distributed shared memory exchange for the cluster kernels (router, message_gates): st.async stores
// into another CTA's shared memory that complete bytes on that CTA's mbarrier. The receiver waits on its own
// mbarrier only (no cluster barrier round trip: ~0.25 us instead of ~0.55 us for a release/acquire cluster
// barrier after remote stores on the RTX 5090). Incoming DSMEM traffic is slow (a store every ~18 clock
// cycles per receiving SM, whatever its width), so exchanges carry a few 16-byte stores per sender.
// Protocol: thread 0 initialises the barrier and announces the expected bytes (mb_init, fence_mbarrier_init,
// mb_expect) before the CTA's relaxed cluster arrive; senders issue their first st.async after the matching
// cluster wait; every barrier completes one phase (parity 0) per launch.
#pragma once

#include <cstdint>

namespace ingine::kernels::dense {

__device__ __forceinline__ uint32_t smem_u32(const void* p) {
  return static_cast<uint32_t>(__cvta_generic_to_shared(p));
}
// address `a` (this CTA's shared memory) in CTA `rank` of the cluster
__device__ __forceinline__ uint32_t remote(uint32_t a, int rank) {
  uint32_t o;
  asm volatile("mapa.shared::cluster.u32 %0, %1, %2;" : "=r"(o) : "r"(a), "r"(rank));
  return o;
}
// 4 floats (16 bytes, the widest st.async) to `addr` (4-byte aligned). Delivery cost is per store (~18 clock
// cycles each on the receiving SM), so exchanges pack 4 values into each one.
__device__ __forceinline__ void st_async4(uint32_t addr, float4 v, uint32_t mb) {
  asm volatile("st.async.shared::cluster.mbarrier::complete_tx::bytes.v4.b32 [%0], {%1, %2, %3, %4}, [%5];" ::"r"(addr),
               "r"(__float_as_uint(v.x)), "r"(__float_as_uint(v.y)), "r"(__float_as_uint(v.z)),
               "r"(__float_as_uint(v.w)), "r"(mb)
               : "memory");
}
__device__ __forceinline__ void mb_init(uint32_t mb) {
  asm volatile("mbarrier.init.shared::cta.b64 [%0], 1;" ::"r"(mb) : "memory");
}
// makes the barrier initialisations visible to the cluster (pairs with the cluster wait)
__device__ __forceinline__ void fence_mbarrier_init() {
  asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
}
// the single arrival of the phase, expecting `bytes` of st.async data
__device__ __forceinline__ void mb_expect(uint32_t mb, uint32_t bytes) {
  asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;" ::"r"(mb), "r"(bytes) : "memory");
}
// waits for phase 0 to complete; the received data is then visible to the calling thread
__device__ __forceinline__ void mb_wait(uint32_t mb) {
  asm volatile(
      "{\n .reg .pred p;\n W: mbarrier.try_wait.parity.acquire.cluster.shared::cta.b64 p, [%0], 0;\n @!p bra W;\n}" ::"r"(mb)
      : "memory");
}

}  // namespace ingine::kernels::dense
