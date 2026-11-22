#include "serve/sys_metrics.h"

#include <dlfcn.h>

#include <algorithm>
#include <cstdint>
#include <fstream>
#include <mutex>
#include <sstream>
#include <string>
#include <thread>

namespace ingine::serve {
namespace {

struct NvmlUtil {
  unsigned int gpu;
  unsigned int memory;
};

struct NvmlMem {
  unsigned long long total;
  unsigned long long free;
  unsigned long long used;
};

struct NvmlApi {
  using InitFn = int (*)();
  using ShutdownFn = int (*)();
  using GetHandleFn = int (*)(unsigned int, void**);
  using GetNameFn = int (*)(void*, char*, unsigned int);
  using GetUtilFn = int (*)(void*, NvmlUtil*);
  using GetMemFn = int (*)(void*, NvmlMem*);
  using GetTempFn = int (*)(void*, int, unsigned int*);
  using GetPowerFn = int (*)(void*, unsigned int*);
  using GetPcieGenFn = int (*)(void*, unsigned int*);
  using GetPcieWidthFn = int (*)(void*, unsigned int*);
  using GetPcieThruFn = int (*)(void*, int, unsigned int*);

  void* lib = nullptr;
  void* dev = nullptr;
  bool ok = false;
  std::string name = "NVIDIA GPU";

  GetUtilFn get_util = nullptr;
  GetMemFn get_mem = nullptr;
  GetTempFn get_temp = nullptr;
  GetPowerFn get_power = nullptr;
  GetPowerFn get_power_limit = nullptr;
  GetPcieGenFn get_pcie_gen = nullptr;
  GetPcieWidthFn get_pcie_width = nullptr;
  GetPcieThruFn get_pcie_thru = nullptr;

  NvmlApi() {
    lib = dlopen("libnvidia-ml.so.1", RTLD_LAZY);
    if (!lib) lib = dlopen("libnvidia-ml.so", RTLD_LAZY);
    if (!lib) return;

    auto init = reinterpret_cast<InitFn>(dlsym(lib, "nvmlInit_v2"));
    if (!init) init = reinterpret_cast<InitFn>(dlsym(lib, "nvmlInit"));
    auto get_handle = reinterpret_cast<GetHandleFn>(dlsym(lib, "nvmlDeviceGetHandleByIndex_v2"));
    if (!get_handle) get_handle = reinterpret_cast<GetHandleFn>(dlsym(lib, "nvmlDeviceGetHandleByIndex"));
    if (!init || !get_handle || init() != 0 || get_handle(0, &dev) != 0) return;

    auto get_name = reinterpret_cast<GetNameFn>(dlsym(lib, "nvmlDeviceGetName"));
    if (get_name) {
      char buf[128]{};
      if (get_name(dev, buf, sizeof(buf)) == 0) name = buf;
    }
    get_util = reinterpret_cast<GetUtilFn>(dlsym(lib, "nvmlDeviceGetUtilizationRates"));
    get_mem = reinterpret_cast<GetMemFn>(dlsym(lib, "nvmlDeviceGetMemoryInfo"));
    get_temp = reinterpret_cast<GetTempFn>(dlsym(lib, "nvmlDeviceGetTemperature"));
    get_power = reinterpret_cast<GetPowerFn>(dlsym(lib, "nvmlDeviceGetPowerUsage"));
    get_power_limit = reinterpret_cast<GetPowerFn>(dlsym(lib, "nvmlDeviceGetEnforcedPowerLimit"));
    get_pcie_gen = reinterpret_cast<GetPcieGenFn>(dlsym(lib, "nvmlDeviceGetCurrPcieLinkGeneration"));
    get_pcie_width = reinterpret_cast<GetPcieWidthFn>(dlsym(lib, "nvmlDeviceGetCurrPcieLinkWidth"));
    get_pcie_thru = reinterpret_cast<GetPcieThruFn>(dlsym(lib, "nvmlDeviceGetPcieThroughput"));
    ok = true;
  }
};

struct CpuSample {
  uint64_t idle = 0;
  uint64_t total = 0;
  bool valid = false;
};

double read_cpu_pct(CpuSample& prev) {
  std::ifstream in("/proc/stat");
  std::string line;
  if (!std::getline(in, line)) return 0.0;
  std::istringstream ss(line);
  std::string label;
  ss >> label;
  if (label != "cpu") return 0.0;
  uint64_t user = 0, nice = 0, sys = 0, idle = 0, iowait = 0, irq = 0, softirq = 0, steal = 0;
  ss >> user >> nice >> sys >> idle >> iowait >> irq >> softirq >> steal;
  const uint64_t cur_idle = idle + iowait;
  const uint64_t cur_total = user + nice + sys + cur_idle + irq + softirq + steal;
  double pct = 0.0;
  if (prev.valid && cur_total > prev.total) {
    const double dt = double(cur_total - prev.total);
    const double di = double(cur_idle - prev.idle);
    pct = std::clamp(100.0 * (1.0 - di / dt), 0.0, 100.0);
  }
  prev.idle = cur_idle;
  prev.total = cur_total;
  prev.valid = true;
  return pct;
}

void read_ram_gb(double& used_gb, double& total_gb) {
  std::ifstream in("/proc/meminfo");
  std::string key;
  uint64_t val_kb = 0;
  std::string unit;
  uint64_t mem_total_kb = 0, mem_avail_kb = 0;
  while (in >> key >> val_kb >> unit) {
    if (key == "MemTotal:") mem_total_kb = val_kb;
    else if (key == "MemAvailable:") mem_avail_kb = val_kb;
    if (mem_total_kb && mem_avail_kb) break;
  }
  if (mem_total_kb > 0) {
    total_gb = double(mem_total_kb) / (1024.0 * 1024.0);
    used_gb = double(mem_total_kb - std::min(mem_total_kb, mem_avail_kb)) / (1024.0 * 1024.0);
  }
}

}  // namespace

SysSnapshot sample_sys_metrics() {
  static std::mutex mu;
  static NvmlApi nvml;
  static CpuSample prev_cpu;

  std::lock_guard lock(mu);
  SysSnapshot s;
  s.cpu_threads = int(std::thread::hardware_concurrency());
  s.cpu_util_pct = read_cpu_pct(prev_cpu);
  read_ram_gb(s.ram_used_gb, s.ram_total_gb);

  if (nvml.ok) {
    s.gpu_available = true;
    s.gpu_name = nvml.name;
    if (nvml.get_util) {
      NvmlUtil u{};
      if (nvml.get_util(nvml.dev, &u) == 0) {
        s.gpu_util_pct = int(u.gpu);
        s.mem_util_pct = int(u.memory);
      }
    }
    if (nvml.get_mem) {
      NvmlMem m{};
      if (nvml.get_mem(nvml.dev, &m) == 0) {
        constexpr double kGiB = 1024.0 * 1024.0 * 1024.0;
        s.vram_used_gb = double(m.used) / kGiB;
        s.vram_total_gb = double(m.total) / kGiB;
      }
    }
    if (nvml.get_temp) {
      unsigned int t = 0;
      if (nvml.get_temp(nvml.dev, 0, &t) == 0) s.gpu_temp_c = int(t);
    }
    if (nvml.get_power) {
      unsigned int mw = 0;
      if (nvml.get_power(nvml.dev, &mw) == 0) s.power_w = double(mw) / 1000.0;
    }
    if (nvml.get_power_limit) {
      unsigned int mw = 0;
      if (nvml.get_power_limit(nvml.dev, &mw) == 0) s.power_limit_w = double(mw) / 1000.0;
    }
    if (nvml.get_pcie_gen) {
      unsigned int gen = 0;
      if (nvml.get_pcie_gen(nvml.dev, &gen) == 0) s.pcie_gen = int(gen);
    }
    if (nvml.get_pcie_width) {
      unsigned int w = 0;
      if (nvml.get_pcie_width(nvml.dev, &w) == 0) s.pcie_width = int(w);
    }
    if (nvml.get_pcie_thru) {
      unsigned int tx = 0, rx = 0;
      if (nvml.get_pcie_thru(nvml.dev, 0, &tx) == 0) s.pcie_tx_mbs = double(tx) / 1024.0;
      if (nvml.get_pcie_thru(nvml.dev, 1, &rx) == 0) s.pcie_rx_mbs = double(rx) / 1024.0;
    }
  }
  return s;
}

}  // namespace ingine::serve
