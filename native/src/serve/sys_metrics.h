// Lightweight system & GPU telemetry (NVML via dlopen + /proc/stat + /proc/meminfo) for /api/metrics.
#pragma once

#include <cstdint>
#include <string>

namespace ingine::serve {

struct SysSnapshot {
  bool gpu_available = false;
  std::string gpu_name = "NVIDIA GPU";
  int gpu_util_pct = 0;
  int mem_util_pct = 0;
  double vram_used_gb = 0.0;
  double vram_total_gb = 0.0;
  int gpu_temp_c = 0;
  double power_w = 0.0;
  double power_limit_w = 0.0;
  int pcie_gen = 0;
  int pcie_width = 0;
  double pcie_tx_mbs = 0.0;
  double pcie_rx_mbs = 0.0;

  double cpu_util_pct = 0.0;
  int cpu_threads = 0;
  double ram_used_gb = 0.0;
  double ram_total_gb = 0.0;
};

SysSnapshot sample_sys_metrics();

}  // namespace ingine::serve
