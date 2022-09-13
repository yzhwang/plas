#ifndef PLAS_GPU_ANY_ALL_KERNEL_H_
#define PLAS_GPU_ANY_ALL_KERNEL_H_

#include <vector>
#include "memory.h"
#include "kernel_launch.h"

namespace plas {
namespace gpu {

// return true if all items in the range [0, n) evaluate to true
template <typename launch_arg_t = empty_t, typename predicate_it>
bool all(
  const uint32_t n, const predicate_it pred, context_t& context) {
  typedef typename conditional_typedef_t<
      launch_arg_t, launch_params_t<256, 7> >::type_t launch_t;

  int nv = launch_t::nv(context);
  int num_ctas = div_up(n, nv);
  mem_t<uint32_t> preds(1, context);
  uint32_t* preds_data = preds.data();
  auto k = [=] PLAS_DEVICE(int tid, int cta) {
    typedef typename launch_t::sm_ptx params_t;
    enum {
      nt = params_t::nt,
      vt = params_t::vt,
      nv = nt * vt
    };
    preds_data[0] = 1u;
    bool p_i = true;
    const int count = (cta * nv + tid * vt >= n ? 0 :
      (((cta * nv + (tid + 1) * vt) > n) ?
      n - (cta * nv + tid * vt) : vt));
    iterate<vt>([&](int i) {
      p_i = (i < count) ? p_i && pred[i] : true;
      });
      bool p = __syncthreads_and(p_i);
      if (p == false) {
        preds_data[0] = 0u;
      }
  };
  cta_launch<launch_t>(k, num_ctas, context);
  std::vector<uint32_t> preds_vec = from_mem(preds);
  return (preds_vec[0] != 0u);
}

// return true if all items in the range [0, n) evaluate to true
template <typename launch_arg_t = empty_t, typename predicate_it>
bool any(
  const uint32_t n, const predicate_it pred, context_t& context) {
  typedef typename conditional_typedef_t<
      launch_arg_t, launch_params_t<256, 7> >::type_t launch_t;

  int nv = launch_t::nv(context);
  int num_ctas = div_up(n, nv);
  mem_t<uint32_t> preds(1, context);
  uint32_t* preds_data = preds.data();
  auto k = [=] PLAS_DEVICE(int tid, int cta) {
    typedef typename launch_t::sm_ptx params_t;
    enum {
      nt = params_t::nt,
      vt = params_t::vt,
      nv = nt * vt
    };

    preds_data[0] = 0u;
    bool p_i = false;
    const int count = (cta * nv + tid * vt >= n ? 0 :
      (((cta * nv + (tid + 1) * vt) > n) ?
      n - (cta * nv + tid * vt) : vt));
    iterate<vt>([&](int i) {
      p_i = i < count ? p_i || pred[i] : false;
      });
      bool p = __syncthreads_or(p_i);
      if (p == true) {
        preds_data[0] = 1u;
      }
  };
  cta_launch<launch_t>(k, num_ctas, context);
  std::vector<uint32_t> preds_vec = from_mem(preds);
  return (preds_vec[0] != 0u);
}

template <typename it_lhs, typename it_rhs>
struct is_sorted_iterator {
  // constructor
  PLAS_HOST_DEVICE
  is_sorted_iterator(const it_lhs _it1, const it_rhs _it2) : it1(_it1), it2(_it2) {}

  // dereference operator
  PLAS_HOST_DEVICE
  bool operator[] (const uint32_t i) const { return it1[i] <= it2[i]; }

  const it_lhs it1;
  const it_rhs it2;
};

template <typename iterator>
bool is_sorted(
  const uint32_t n,
  const iterator values, context_t& context) {
  return all(n-1, is_sorted_iterator<iterator, iterator>(values, values+1), context);
}

template <typename it_lhs, typename it_rhs>
struct is_unsorted_iterator {
  // constructor
  PLAS_HOST_DEVICE
  is_unsorted_iterator(const it_lhs _it1, const it_rhs _it2) : it1(_it1), it2(_it2) {}

  // dereference operator
  PLAS_HOST_DEVICE
  bool operator[] (const uint32_t i) const { return it1[i] > it2[i]; }

  const it_lhs it1;
  const it_rhs it2;
};

template <typename iterator>
bool is_unsorted(
  const uint32_t n,
  const iterator values, context_t& context) {
  return any(n-1, is_unsorted_iterator<iterator, iterator>(values, values+1), context);
}

}  // namespace gpu
}  // namespace plas

#endif
