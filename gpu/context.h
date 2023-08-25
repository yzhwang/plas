#ifndef PLAS_GPU_CONTEXT_H_
#define PLAS_GPU_CONTEXT_H_

#include <vector>
#include <memory>
#include <cassert>
#include <exception>
#include "udt.h"
#include "util.h"
#include "launch_params.h"
#include <chrono>

namespace plas {
namespace gpu {

enum memory_space_t {
  memory_space_device = 0,
  memory_space_host = 1
};

inline std::string device_prop_string(cudaDeviceProp prop) {
  int ordinal;
  cudaGetDevice(&ordinal);

  size_t freeMem, totalMem;
  cudaError_t result = cudaMemGetInfo(&freeMem, &totalMem);
  if (cudaSuccess != result) throw cuda_exception_t(result);

  double memBandwidth =
      (prop.memoryClockRate * 1000.0) * (prop.memoryBusWidth / 8 * 2) / 1.0e9;

  std::string s = detail::stringprintf(
      "%s : %8.3lf Mhz   (Ordinal %d)\n"
      "%d SMs enabled. Compute Capability sm_%d%d\n"
      "FreeMem: %6dMB   TotalMem: %6dMB   %2d-bit pointers.\n"
      "Mem Clock: %8.3lf Mhz x %d bits   (%5.1lf GB/s)\n"
      "ECC %s\n\n",
      prop.name, prop.clockRate / 1000.0, ordinal, prop.multiProcessorCount,
      prop.major, prop.minor, (int)(freeMem / (1 << 20)),
      (int)(totalMem / (1 << 20)), 8 * sizeof(int*),
      prop.memoryClockRate / 1000.0, prop.memoryBusWidth, memBandwidth,
      prop.ECCEnabled ? "Enabled" : "Disabled");
  return s;
}

////////////////////////////////////////////////////////////////////////////////
// context_t
// Derive context_t to add support for streams and a custom allocator.

struct context_t {
  context_t() = default;
  std::chrono::time_point<std::chrono::high_resolution_clock> _cpu_time;

  // Disable copy ctor and assignment operator. We don't want to let the
  // user copy only a slice.
  context_t(const context_t& rhs) = delete;
  context_t& operator=(const context_t& rhs) = delete;

  virtual const cudaDeviceProp& props() const = 0;
  virtual int ptx_version() const = 0;
  virtual cudaStream_t stream() = 0;

  // Alloc GPU memory.
  virtual void* alloc(size_t size, memory_space_t space) = 0;
  virtual void free(void* p, memory_space_t space) = 0;

  // cudaStreamSynchronize or cudaDeviceSynchronize for stream 0.
  virtual void synchronize() = 0;

  virtual cudaEvent_t event() = 0;
  virtual void timer_begin() = 0;
  virtual double timer_end() = 0;

  void cpu_timer_begin() {
    _cpu_time = std::chrono::high_resolution_clock::now();
  }

  // return time in msec
  double cpu_timer_end() {
    auto end = std::chrono::high_resolution_clock::now();
    return std::chrono::duration_cast<std::chrono::milliseconds>(end - _cpu_time).count();
  }
};

////////////////////////////////////////////////////////////////////////////////
// standard_context_t is a trivial implementation of context_t. Users can
// derive this type to provide a custom allocator.

class standard_context_t : public context_t {
 protected:
  cudaDeviceProp _props;
  int _ptx_version;
  cudaStream_t _stream;

  cudaEvent_t _timer[2];
  cudaEvent_t _event;

  // Making this a template argument means we won't generate an instance
  // of dummy_k for each translation unit.
  template <int dummy_arg = 0>
  void init() {
    cudaFuncAttributes attr;
    cudaError_t result = cudaFuncGetAttributes(&attr, dummy_k<0>);
    if (cudaSuccess != result) throw cuda_exception_t(result);
    _ptx_version = attr.ptxVersion;

    int ord;
    cudaGetDevice(&ord);
    cudaGetDeviceProperties(&_props, ord);

    cudaEventCreate(&_timer[0]);
    cudaEventCreate(&_timer[1]);
    cudaEventCreate(&_event);
  }

 public:
  explicit standard_context_t(bool print_prop = true, cudaStream_t stream_ = 0)
      : context_t(), _stream(stream_) {
    init();
    if (print_prop) {
      printf("%s\n", device_prop_string(_props).c_str());
    }
  }
  ~standard_context_t() {
    cudaEventDestroy(_timer[0]);
    cudaEventDestroy(_timer[1]);
    cudaEventDestroy(_event);
  }

  virtual const cudaDeviceProp& props() const { return _props; }
  virtual int ptx_version() const { return _ptx_version; }
  virtual cudaStream_t stream() { return _stream; }

  // Alloc GPU memory.
  virtual void* alloc(size_t size, memory_space_t space) {
    void* p = nullptr;
    if (size) {
      cudaError_t result = (memory_space_device == space)
                               ? cudaMalloc(&p, size)
                               : cudaMallocHost(&p, size);
      if (cudaSuccess != result) throw cuda_exception_t(result);
    }
    return p;
  }

  virtual void free(void* p, memory_space_t space) {
    if (p) {
      cudaError_t result =
          (memory_space_device == space) ? cudaFree(p) : cudaFreeHost(p);
      if (cudaSuccess != result) throw cuda_exception_t(result);
    }
  }

  virtual void synchronize() {
    cudaError_t result =
        _stream ? cudaStreamSynchronize(_stream) : cudaDeviceSynchronize();
    if (cudaSuccess != result) throw cuda_exception_t(result);
  }

  virtual cudaEvent_t event() { return _event; }
  virtual void timer_begin() { cudaEventRecord(_timer[0], _stream); }
  virtual double timer_end() {
    cudaEventRecord(_timer[1], _stream);
    cudaEventSynchronize(_timer[1]);
    float ms;
    cudaEventElapsedTime(&ms, _timer[0], _timer[1]);
    return ms;
  }
};

class context_singleton {
 public:
  static standard_context_t& get_instance(cudaStream_t stream_ = 0) {
    static standard_context_t instance_(false, stream_);
    return instance_;
  }
};

}  // namespace gpu
}  // namespace plas

#endif
