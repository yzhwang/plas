#ifndef PLAS_GPU_UDT_H_
#define PLAS_GPU_UDT_H_

#include "meta.h"

namespace plas {
namespace gpu {

struct cuda_exception_t : std::exception {
  cudaError_t result;

  explicit cuda_exception_t(cudaError_t result_) : result(result_) {}
  virtual const char* what() const noexcept {
    return cudaGetErrorString(result);
  }
};

template <typename type_t, int size>
struct array_t {
  type_t data[size];  // NOLINT

  PLAS_HOST_DEVICE type_t operator[](int i) const { return data[i]; }
  PLAS_HOST_DEVICE type_t& operator[](int i) { return data[i]; }

  array_t() = default;
  array_t(const array_t&) = default;
  array_t& operator=(const array_t&) = default;

  // Fill the array with x.
  PLAS_HOST_DEVICE array_t(type_t x) {
    iterate<size>([&](int i) { data[i] = x; });
  }
};

template <typename type_t>
struct array_t<type_t, 0> {
  PLAS_HOST_DEVICE type_t operator[](int i) const { return type_t(); }
  PLAS_HOST_DEVICE type_t& operator[](int i) { return *(type_t*)nullptr; }
};

// Reduce on components of array_t.
template <typename type_t, int size, typename op_t = plus_t<type_t> >
PLAS_HOST_DEVICE type_t reduce(array_t<type_t, size> x, op_t op = op_t()) {
  type_t a;
  iterate<size>([&](int i) { a = i ? op(a, x[i]) : x[i]; });
  return a;
}

// Call the operator component-wise on all components.
template <typename type_t, int size, typename op_t>
PLAS_HOST_DEVICE array_t<type_t, size> combine(array_t<type_t, size> x,
                                               array_t<type_t, size> y,
                                               op_t op) {
  array_t<type_t, size> z;
  iterate<size>([&](int i) { z[i] = op(x[i], y[i]); });
  return z;
}

template <typename type_t, int size>
PLAS_HOST_DEVICE array_t<type_t, size> operator+(array_t<type_t, size> a,
                                                 array_t<type_t, size> b) {
  return combine(a, b, plus_t<type_t>());
}

template <typename type_t, int size>
PLAS_HOST_DEVICE array_t<type_t, size> operator-(array_t<type_t, size> a,
                                                 array_t<type_t, size> b) {
  return combine(a, b, minus_t<type_t>());
}

template <typename key_t, typename val_t, int size>
struct kv_array_t {
  array_t<key_t, size> keys;
  array_t<val_t, size> vals;
};

enum bounds_t {
  bounds_lower,
  bounds_upper
};

struct PLAS_ALIGN(8) range_t {
  int begin, end;
  PLAS_HOST_DEVICE int size() const { return end - begin; }
  PLAS_HOST_DEVICE int count() const { return size(); }
  PLAS_HOST_DEVICE bool valid() const { return end > begin; }
};

PLAS_HOST_DEVICE range_t get_tile(int cta, int nv, int count) {
  return range_t{nv * cta, min(count, nv * (cta + 1))};
}

struct PLAS_ALIGN(16) merge_range_t {
  int a_begin, a_end, b_begin, b_end;

  PLAS_HOST_DEVICE int a_count() const { return a_end - a_begin; }
  PLAS_HOST_DEVICE int b_count() const { return b_end - b_begin; }
  PLAS_HOST_DEVICE int total() const { return a_count() + b_count(); }

  PLAS_HOST_DEVICE range_t a_range() const {
    return range_t{a_begin, a_end};
  }
  PLAS_HOST_DEVICE range_t b_range() const {
    return range_t{b_begin, b_end};
  }

  PLAS_HOST_DEVICE merge_range_t to_local() const {
    return merge_range_t{0, a_count(), a_count(), total()};
  }

  // Partition from mp to the end.
  PLAS_HOST_DEVICE merge_range_t partition(int mp0, int diag) const {
    return merge_range_t{a_begin + mp0, a_end, b_begin + diag - mp0, b_end};
  }

  // Partition from mp0 to mp1.
  PLAS_HOST_DEVICE merge_range_t
  partition(int mp0, int diag0, int mp1, int diag1) const {
    return merge_range_t{a_begin + mp0,         a_begin + mp1,
                         b_begin + diag0 - mp0, b_begin + diag1 - mp1};
  }

  PLAS_HOST_DEVICE bool a_valid() const { return a_begin < a_end; }
  PLAS_HOST_DEVICE bool b_valid() const { return b_begin < b_end; }
};

template <typename type_t, int size>
struct merge_pair_t {
  array_t<type_t, size> keys;
  array_t<int, size> indices;
};

}  // namespace gpu
}  // namespace plas

#endif
