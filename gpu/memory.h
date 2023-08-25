#ifndef PLAS_GPU_MEMORY_H_
#define PLAS_GPU_MEMORY_H_

#include "kernel_launch.h"
#include "context.h"

// Memory related class and functions for running plas_gpu without any
// DL framework.

namespace plas {
namespace gpu {

////////////////////////////////////////////////////////////////////////////////
// mem_t

template <typename type_t>
class mem_t {
  context_t* _context;
  type_t* _pointer;
  size_t _size;
  memory_space_t _space;

 public:
  void swap(mem_t& rhs) {
    std::swap(_context, rhs._context);
    std::swap(_pointer, rhs._pointer);
    std::swap(_size, rhs._size);
    std::swap(_space, rhs._space);
  }

  mem_t()
      : _context(nullptr),
        _pointer(nullptr),
        _size(0),
        _space(memory_space_device) {}
  mem_t& operator=(const mem_t& rhs) = delete;
  mem_t(const mem_t& rhs) = delete;

  mem_t(size_t size, context_t& context,
        memory_space_t space = memory_space_device)
      : _context(&context), _pointer(nullptr), _size(size), _space(space) {
    _pointer = (type_t*)context.alloc(sizeof(type_t) * size, space);
  }

  mem_t(mem_t&& rhs) : mem_t() { swap(rhs); }
  mem_t& operator=(mem_t&& rhs) {
    swap(rhs);
    return *this;
  }

  ~mem_t() {
    if (_context && _pointer) _context->free(_pointer, _space);
    _pointer = nullptr;
    _size = 0;
  }

  context_t& context() { return *_context; }
  size_t size() const { return _size; }
  type_t* data() const { return _pointer; }
  memory_space_t space() const { return _space; }

  // Return a deep copy of this container.
  mem_t clone() {
    mem_t cloned(size(), context(), space());
    if (memory_space_device)
      dtod(cloned.data(), data(), size());
    else
      htoh(cloned.data(), data(), size());
    return cloned;
  }
};

////////////////////////////////////////////////////////////////////////////////
// Memory functions on raw pointers.

template <typename type_t>
cudaError_t htoh(type_t* dest, const type_t* source, size_t count) {
  if (count) memcpy(dest, source, sizeof(type_t) * count);
  return cudaSuccess;
}

template <typename type_t>
cudaError_t dtoh(type_t* dest, const type_t* source, size_t count) {
  cudaError_t result = count ? cudaMemcpy(dest, source, sizeof(type_t) * count,
                                          cudaMemcpyDeviceToHost)
                             : cudaSuccess;
  return result;
}

template <typename type_t>
cudaError_t htod(type_t* dest, const type_t* source, size_t count) {
  cudaError_t result = count ? cudaMemcpy(dest, source, sizeof(type_t) * count,
                                          cudaMemcpyHostToDevice)
                             : cudaSuccess;
  return result;
}

template <typename type_t>
cudaError_t dtod(type_t* dest, const type_t* source, size_t count) {
  cudaError_t result = count ? cudaMemcpy(dest, source, sizeof(type_t) * count,
                                          cudaMemcpyDeviceToDevice)
                             : cudaSuccess;
  return result;
}

template <typename type_t>
cudaError_t dtoh(std::vector<type_t>& dest, const type_t* source,
                 size_t count) {
  dest.resize(count);
  return dtoh(dest.data(), source, count);
}

template <typename type_t>
cudaError_t htod(type_t* dest, const std::vector<type_t>& source) {
  return htod(dest, source.data(), source.size());
}

////////////////////////////////////////////////////////////////////////////////
// Memory functions on mem_t.

template <typename type_t>
mem_t<type_t> to_mem(const std::vector<type_t>& data, context_t& context) {
  mem_t<type_t> mem(data.size(), context);
  cudaError_t result = htod(mem.data(), data);
  if (cudaSuccess != result) throw cuda_exception_t(result);
  return mem;
}

template <typename type_t>
std::vector<type_t> from_mem(const mem_t<type_t>& mem) {
  std::vector<type_t> host;
  cudaError_t result = dtoh(host, mem.data(), mem.size());
  if (cudaSuccess != result) throw cuda_exception_t(result);
  return host;
}

template <typename type_t, typename func_t>
mem_t<type_t> fill_function(func_t f, size_t count, context_t& context) {
  mem_t<type_t> mem(count, context);
  type_t* p = mem.data();
  transform([=] PLAS_DEVICE(int index) { p[index] = f(index); }, count,
            context);
  return mem;
}

template <typename type_t>
mem_t<type_t> fill(type_t value, size_t count, context_t& context) {
  // We'd prefer to call fill_function and pass a lambda that returns value,
  // but that can create tokens that are too long for VS2013.
  mem_t<type_t> mem(count, context);
  type_t* p = mem.data();
  transform([=] PLAS_DEVICE(int index) { p[index] = value; }, count, context);
  return mem;
}

template <typename it_t>
auto copy_to_mem(it_t input, size_t count, context_t& context)
    -> mem_t<typename std::iterator_traits<it_t>::value_type> {
  typedef typename std::iterator_traits<it_t>::value_type type_t;
  mem_t<type_t> mem(count, context);
  type_t* p = mem.data();
  transform([=] PLAS_DEVICE(int index) { p[index] = input[index]; }, count,
            context);
  return mem;
}

inline std::mt19937& get_mt19937() {
  static std::mt19937 mt19937;
  return mt19937;
}

mem_t<int> inline fill_random(int a, int b, size_t count, bool sorted,
                              context_t& context) {
  std::uniform_int_distribution<int> d(a, b);
  std::vector<int> data(count);

  for (int& i : data) i = d(get_mt19937());
  if (sorted) std::sort(data.begin(), data.end());

  return to_mem(data, context);
}

mem_t<float> inline fill_random(float a, float b, size_t count, bool sorted,
                                context_t& context) {
  std::uniform_real_distribution<float> d(a, b);
  std::vector<float> data(count);

  for (float& i : data) i = d(get_mt19937());
  if (sorted) std::sort(data.begin(), data.end());

  return to_mem(data, context);
}

mem_t<double> inline fill_random(double a, double b, size_t count, bool sorted,
                                context_t& context) {
  std::uniform_real_distribution<double> d(a, b);
  std::vector<double> data(count);

  for (double& i : data) i = d(get_mt19937());
  if (sorted) std::sort(data.begin(), data.end());

  return to_mem(data, context);
}

}  // namespace gpu
}  // namespace plas

#endif
