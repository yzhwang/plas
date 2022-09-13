#ifndef PLAS_GPU_META_H_
#define PLAS_GPU_META_H_

#include <typeinfo>
#include <type_traits>
#include <iterator>
#include <cassert>
#include <cfloat>
#include <cstdint>

#ifdef __CUDACC__

#ifndef PLAS_HOST_DEVICE
#define PLAS_HOST_DEVICE __forceinline__ __device__ __host__
#endif

#ifndef PLAS_DEVICE
#define PLAS_DEVICE __device__
#endif

#ifndef PLAS_LAMBDA
#define PLAS_LAMBDA __device__ __host__
#endif

#else  // #ifndef __CUDACC__
#define PLAS_HOST_DEVICE

#endif  // #ifdef __CUDACC__

namespace plas {
namespace gpu {

template <bool B, class T = void>
using enable_if_t = typename std::enable_if<B, T>::type;

enum {
  warp_size = 32
};

////////////////////////////////////////////////////////////////////////////////
// Basic utility functions

PLAS_HOST_DEVICE constexpr bool is_pow2(int x) { return 0 == (x & (x - 1)); }

PLAS_HOST_DEVICE constexpr int div_up(int x, int y) { return (x + y - 1) / y; }

PLAS_HOST_DEVICE constexpr int64_t div_up(int64_t x, int64_t y) {
  return (x + y - 1) / y;
}

PLAS_HOST_DEVICE constexpr size_t div_up(size_t x, size_t y) {
  return (x + y - 1) / y;
}

PLAS_HOST_DEVICE constexpr int s_log2(int x, int p = 0) {
  return x > 1 ? s_log2(x / 2) + 1 : p;
}

PLAS_HOST_DEVICE constexpr size_t s_log2(size_t x, size_t p = 0) {
  return x > 1 ? s_log2(x / 2) + 1 : p;
}

template <typename real_t>
PLAS_HOST_DEVICE constexpr real_t min(real_t a, real_t b) {
  return (b < a) ? b : a;
}

template <typename real_t>
PLAS_HOST_DEVICE constexpr real_t max(real_t a, real_t b) {
  return (a < b) ? b : a;
}

#define PLAS_ALIGN(x) __attribute__((aligned(x)))

struct empty_t {};

template <typename... args_t>
PLAS_HOST_DEVICE void swallow(args_t...) {}

template <typename... base_v>
struct inherit_t;

template <typename base_t, typename... base_v>
struct inherit_t<base_t,
                 base_v...> : base_t::template rebind<inherit_t<base_v...> > {};

template <typename base_t>
struct inherit_t<base_t> : base_t {};

////////////////////////////////////////////////////////////////////////////////
// Conditional typedefs.

// Typedef type_a if type_a is not empty_t.
// Otherwise typedef type_b.
template <typename type_a, typename type_b>
struct conditional_typedef_t {
  typedef typename std::conditional<!std::is_same<type_a, empty_t>::value,
                                    type_a, type_b>::type type_t;
};

////////////////////////////////////////////////////////////////////////////////
// Code to treat __restrict__ as a CV qualifier.

template <typename arg_t>
struct is_restrict {
  enum {
    value = false
  };
};
template <typename arg_t>
struct is_restrict<arg_t __restrict__> {
  enum {
    value = true
  };
};

// Add __restrict__ only to pointers.
template <typename arg_t>
struct add_restrict {
  typedef arg_t type;
};
template <typename arg_t>
struct add_restrict<arg_t*> {
  typedef arg_t* __restrict__ type;
};

template <typename arg_t>
struct remove_restrict {
  typedef arg_t type;
};
template <typename arg_t>
struct remove_restrict<arg_t __restrict__> {
  typedef arg_t type;
};

template <typename arg_t>
PLAS_HOST_DEVICE typename add_restrict<arg_t>::type make_restrict(arg_t x) {
  typename add_restrict<arg_t>::type y = x;
  return y;
}

// read-only data cache load function __ldg() intrinsic is supported
// on GPUs with cc >= 3.5. Usually 2-3x faster than accessing through
// raw pointer.

namespace detail {
template <typename it_t,
          typename type_t = typename std::iterator_traits<it_t>::value_type,
          bool use_ldg =
              std::is_pointer<it_t>::value&& std::is_arithmetic<type_t>::value>
struct ldg_load_t {
  PLAS_HOST_DEVICE static type_t load(it_t it) { return *it; }
};

#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 350

template <typename it_t, typename type_t>
struct ldg_load_t<it_t, type_t, true> {
  PLAS_HOST_DEVICE static type_t load(it_t it) { return __ldg(it); }
};

#endif

}  // namespace detail

template <typename it_t>
PLAS_HOST_DEVICE typename std::iterator_traits<it_t>::value_type ldg(it_t it) {
  return detail::ldg_load_t<it_t>::load(it);
}

////////////////////////////////////////////////////////////////////////////////
// Device-side comparison operators.

template <typename type_t>
struct less_t : public std::binary_function<type_t, type_t, bool> {
  PLAS_HOST_DEVICE bool operator()(type_t a, type_t b) const { return a < b; }
};
template <typename type_t>
struct less_equal_t : public std::binary_function<type_t, type_t, bool> {
  PLAS_HOST_DEVICE bool operator()(type_t a, type_t b) const { return a <= b; }
};
template <typename type_t>
struct greater_t : public std::binary_function<type_t, type_t, bool> {
  PLAS_HOST_DEVICE bool operator()(type_t a, type_t b) const { return a > b; }
};
template <typename type_t>
struct greater_equal_t : public std::binary_function<type_t, type_t, bool> {
  PLAS_HOST_DEVICE bool operator()(type_t a, type_t b) const { return a >= b; }
};
template <typename type_t>
struct equal_to_t : public std::binary_function<type_t, type_t, bool> {
  PLAS_HOST_DEVICE bool operator()(type_t a, type_t b) const { return a == b; }
};
template <typename type_t>
struct not_equal_to_t : public std::binary_function<type_t, type_t, bool> {
  PLAS_HOST_DEVICE bool operator()(type_t a, type_t b) const { return a != b; }
};

////////////////////////////////////////////////////////////////////////////////
// Device-side arithmetic operators.

template <typename type_t>
struct plus_t : public std::binary_function<type_t, type_t, type_t> {
  PLAS_HOST_DEVICE type_t operator()(type_t a, type_t b) const { return a + b; }
};

template <typename type_t>
struct minus_t : public std::binary_function<type_t, type_t, type_t> {
  PLAS_HOST_DEVICE type_t operator()(type_t a, type_t b) const { return a - b; }
};

template <typename type_t>
struct multiplies_t : public std::binary_function<type_t, type_t, type_t> {
  PLAS_HOST_DEVICE type_t operator()(type_t a, type_t b) const { return a * b; }
};

template <typename type_t>
struct maximum_t : public std::binary_function<type_t, type_t, type_t> {
  PLAS_HOST_DEVICE type_t operator()(type_t a, type_t b) const {
    return max(a, b);
  }
};

template <typename type_t>
struct minimum_t : public std::binary_function<type_t, type_t, type_t> {
  PLAS_HOST_DEVICE type_t operator()(type_t a, type_t b) const {
    return min(a, b);
  }
};

////////////////////////////////////////////////////////////////////////////////
// iterator_t and const_iterator_t are base classes for customized iterators.

template <typename outer_t, typename int_t, typename value_type>
struct iterator_t : public std::iterator_traits<const value_type*> {
  iterator_t() = default;
  PLAS_HOST_DEVICE iterator_t(int_t i) : index(i) {}

  PLAS_HOST_DEVICE outer_t operator+(int_t diff) const {
    outer_t next = *static_cast<const outer_t*>(this);
    next += diff;
    return next;
  }
  PLAS_HOST_DEVICE outer_t operator-(int_t diff) const {
    outer_t next = *static_cast<const outer_t*>(this);
    next -= diff;
    return next;
  }
  PLAS_HOST_DEVICE outer_t& operator+=(int_t diff) {
    index += diff;
    return *static_cast<outer_t*>(this);
  }
  PLAS_HOST_DEVICE outer_t& operator-=(int_t diff) {
    index -= diff;
    return *static_cast<outer_t*>(this);
  }

  int_t index;
};

template <typename outer_t, typename int_t, typename value_type>
struct const_iterator_t : public iterator_t<outer_t, int_t, value_type> {
  typedef iterator_t<outer_t, int_t, value_type> base_t;

  const_iterator_t() = default;
  PLAS_HOST_DEVICE const_iterator_t(int_t i) : base_t(i) {}

  // operator[] and operator* are tagged as DEVICE-ONLY.  This is to ensure
  // compatibility with lambda capture in CUDA 7.5, which does not support
  // marking a lambda as __host__ __device__.
  // We hope to relax this when a future CUDA fixes this problem.
  PLAS_HOST_DEVICE value_type operator[](int_t diff) const {
    return static_cast<const outer_t&>(*this)(base_t::index + diff);
  }
  PLAS_HOST_DEVICE value_type operator*() const { return (*this)[0]; }
};

////////////////////////////////////////////////////////////////////////////////
// discard_iterator_t is a store iterator that discards its input.

template <typename value_type>
struct discard_iterator_t
    : iterator_t<discard_iterator_t<value_type>, int, value_type> {

  struct assign_t {
    PLAS_HOST_DEVICE value_type operator=(value_type v) { return value_type(); }
  };

  PLAS_HOST_DEVICE assign_t operator[](int index) const { return assign_t(); }
  PLAS_HOST_DEVICE assign_t operator*() const { return assign_t(); }
};

////////////////////////////////////////////////////////////////////////////////
// Template unrolled looping construct.

template <int i, int count, bool valid = (i < count)>
struct iterate_t {
#pragma nv_exec_check_disable
  template <typename func_t>
  PLAS_HOST_DEVICE static void eval(func_t f) {
    f(i);
    iterate_t<i + 1, count>::eval(f);
  }
};
template <int i, int count>
struct iterate_t<i, count, false> {
  template <typename func_t>
  PLAS_HOST_DEVICE static void eval(func_t f) {}
};

template <int begin, int end, typename func_t>
PLAS_HOST_DEVICE void iterate(func_t f) {
  iterate_t<begin, end>::eval(f);
}

template <int count, typename func_t>
PLAS_HOST_DEVICE void iterate(func_t f) {
  iterate<0, count>(f);
}

template <int count, typename type_t, typename op_t = plus_t<type_t> >
PLAS_HOST_DEVICE type_t reduce(const type_t (&x)[count], op_t op = op_t()) {
  type_t y;
  iterate<count>([&](int i) { y = i ? op(y, x[i]) : x[i]; });
  return y;
}

template <int count, typename type_t>
PLAS_HOST_DEVICE void fill(type_t (&x)[count], type_t val) {
  iterate<count>([&](int i) { x[i] = val; });
}

#ifdef __CUDACC__

// Invoke unconditionally.
template <int nt, int vt, typename func_t>
PLAS_DEVICE void strided_iterate(func_t f, int tid) {
  iterate<vt>([=](int i) { f(i, nt * i + tid); });
}

// Check range.
template <int nt, int vt, int vt0 = vt, typename func_t>
PLAS_DEVICE void strided_iterate(func_t f, int tid, int count) {
  // Unroll the first vt0 elements of each thread.
  if (vt0 > 1 && count >= nt * vt0) {
    strided_iterate<nt, vt0>(f, tid);
  } else {
    iterate<vt0>([=](int i) {
      int j = nt * i + tid;
      if (j < count) f(i, j);
    });
  }

  // the remaining vt- vt0 values
  iterate<vt0, vt>([=](int i) {
    int j = nt * i + tid;
    if (j < count) f(i, j);
  });
}

template <int vt, typename func_t>
PLAS_DEVICE void thread_iterate(func_t f, int tid) {
  iterate<vt>([=](int i) { f(i, vt * tid + i); });
}

#endif  // ifdef __CUDACC__

}  // namespace gpu
}  // namespace plas

#endif  // ifndef PLAS_GPU_META_H_
