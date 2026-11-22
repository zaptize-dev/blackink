// Launch helpers of the `dense` kernels: split cluster barriers, programmatic dependent launch (PDL),
// cluster / PDL kernel launch. Ported from engine/csrc/dense/cluster.cuh (torch stream -> cudaStream_t).
#pragma once

#include "core/device.h"

#include <cuda_runtime.h>

#include <stdexcept>
#include <string>
#include <utility>

namespace ingine::kernels::dense {

// Host-side argument check (replaces TORCH_CHECK).
inline void require(bool ok, const char* what) {
  if (!ok) throw std::invalid_argument(std::string("dense: ") + what);
}

// Split cluster barrier. A relaxed arrive at kernel start + wait before the first remote shared-memory
// store guarantees every CTA of the cluster is running; release/acquire pairs publish remote stores.
__device__ __forceinline__ void cluster_arrive_relaxed() {
  asm volatile("barrier.cluster.arrive.relaxed.aligned;" ::: "memory");
}
__device__ __forceinline__ void cluster_arrive() { asm volatile("barrier.cluster.arrive.release.aligned;" ::: "memory"); }
__device__ __forceinline__ void cluster_wait() { asm volatile("barrier.cluster.wait.acquire.aligned;" ::: "memory"); }

// Programmatic dependent launch. Every dense kernel loads only weights/constants before pdl_wait() and
// every activation/state input, and every write, after it; pdl_wait() returns once the preceding grid has
// completed and its writes are visible. pdl_trigger() lets the next kernel of the stream start its own
// prologue (it still waits for this grid's completion before reading our outputs). Both are no-ops when
// the kernel was not launched with the PDL attribute.
__device__ __forceinline__ void pdl_wait() { asm volatile("griddepcontrol.wait;" ::: "memory"); }
__device__ __forceinline__ void pdl_trigger() { asm volatile("griddepcontrol.launch_dependents;" ::: "memory"); }
// griddepcontrol.wait that keeps the loads issued before it ahead of it. ptxas sinks independent loads
// below a plain griddepcontrol.wait (legal: the wait only orders later accesses), even below a
// __syncthreads, which silently turns the "weights in flight before the wait" prefetch into loads issued
// after the predecessor completed; it does not move them into a loop whose trip count it cannot see.
// `zero` must be 0 at run time and opaque to the compiler (a kernel argument).
__device__ __forceinline__ void pdl_wait_fenced(uint32_t zero) {
#pragma unroll 1
  for (uint32_t i = 0; i <= zero; ++i) asm volatile("griddepcontrol.wait;" ::: "memory");
}

// Launches `kernel` on `stream` (graph capturable): as clusters of `cl` CTAs if cl > 1, with
// the programmatic-stream-serialization attribute if `pdl` (captured as a programmatic graph edge).
// Cluster sizes above 8 need the non-portable opt-in; shared memory above 48 KB the dynamic-size opt-in.
template <class... KArgs, class... Args>
void launch_ex(void (*kernel)(KArgs...), dim3 grid, int block, int smem, int cl, bool pdl, cudaStream_t stream,
               Args&&... args) {
  if (cl > 8) CUDA_CHECK(cudaFuncSetAttribute(kernel, cudaFuncAttributeNonPortableClusterSizeAllowed, 1));
  // static + dynamic shared memory above 48 KB needs the opt-in, whatever the split
  if (smem > 0) CUDA_CHECK(cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem));
  cudaLaunchConfig_t cfg = {};
  cfg.gridDim = grid;
  cfg.blockDim = dim3(block);
  cfg.dynamicSmemBytes = smem;
  cfg.stream = stream;
  cudaLaunchAttribute attr[2];
  int n = 0;
  if (cl > 1) {
    attr[n].id = cudaLaunchAttributeClusterDimension;
    attr[n].val.clusterDim.x = cl;
    attr[n].val.clusterDim.y = 1;
    attr[n].val.clusterDim.z = 1;
    ++n;
  }
  if (pdl) {
    attr[n].id = cudaLaunchAttributeProgrammaticStreamSerialization;
    attr[n].val.programmaticStreamSerializationAllowed = 1;
    ++n;
  }
  cfg.attrs = attr;
  cfg.numAttrs = n;
  CUDA_CHECK(cudaLaunchKernelEx(&cfg, kernel, std::forward<Args>(args)...));
}

}  // namespace ingine::kernels::dense
