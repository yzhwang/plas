#ifndef PLAS_GPU_LOOKUP_QUERY_KERNEL_H_
#define PLAS_GPU_LOOKUP_QUERY_KERNEL_H_

#include "search_cta.h"
#include "scan_kernel.h"
#include "memory.h"
#include "kernel_launch.h"

#include <cub/cub.cuh>

namespace plas {
namespace gpu {
// Lookup query in unsorted data, return indices of query in data.
// no duplicates supported for now.
template <typename launch_arg_t = empty_t, typename query_keys_it,
          typename data_keys_it, typename indices_it>
void lookup_query(query_keys_it query, int query_len, data_keys_it data,
  int data_len, indices_it indices, context_t& context) {
  typedef typename conditional_typedef_t<
      launch_arg_t, launch_params_t<256, 7> >::type_t launch_t;

  int nv = launch_t::nv(context);
  // divide data array into tiles each has size nv
  int num_ctas = div_up(data_len, nv);

  typedef typename std::iterator_traits<query_keys_it>::value_type type_t;
  typedef typename std::iterator_traits<indices_it>::value_type ind_t;

  auto k = [=] PLAS_DEVICE(int tid, int cta) {
    typedef typename launch_t::sm_ptx params_t;
    enum {
      nt = params_t::nt,
      vt = params_t::vt,
      nv = nt * vt
    };

    typedef cub::BlockRadixSort<type_t, nt, vt, ind_t> BlockRadixSort;

    __shared__ struct {
      type_t keys[nv];  // NOLINT
      ind_t indices[nv];  // NOLINT
      typename BlockRadixSort::TempStorage temp_storage;
    } shared;

    // load from data to register
    int start = cta * nv;
    int end = (cta + 1) * nv < data_len ? (cta+1) * nv : data_len;
    array_t<type_t, vt> data_key;
    array_t<ind_t, vt> data_ind;
    iterate<vt>([&](int i) {
      ind_t local_start = start + tid * vt;
      data_ind[i] = (local_start + i < end) ? local_start + i : data_len;
      data_key[i] = (local_start + i < end) ? ldg(data + local_start + i) : (type_t)-1;
    });

    // cub radix_sort data chunk
    BlockRadixSort(shared.temp_storage).Sort(data_key.data, data_ind.data);

    // put sorted kv pair of data into shared memory.
    reg_to_shared_thread<nt, vt, type_t, nv>(data_key, tid, shared.keys);
    reg_to_shared_thread<nt, vt, ind_t, nv>(data_ind, tid, shared.indices);
    __syncthreads();

    // load query and search in sorted data
    int num_query_tiles = div_up(query_len, nv);
    for (int i = 0; i < num_query_tiles; ++i) {
      int query_start = i * nv + tid * vt;
      int query_end = (query_start + vt < query_len) ? query_start + vt : query_len;
      type_t q;
      int count = nv;
      iterate<vt>([&](int j) {
        if (query_start + j < query_end) {
          q = ldg(query + query_start + j);
          int idx = binary_search(shared.keys, count, q);
          if (idx >= 0) {
            ind_t data_index = shared.indices[idx];
            indices[query_start+j] = data_index;
          }
        }
      });
    }
  };
  cta_launch<launch_t>(k, num_ctas, context);
}

}  // namespace gpu
}  // namespace plas

#endif

