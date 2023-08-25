#ifndef PLAS_GPU_INTERSECTION_KERNEL_H_
#define PLAS_GPU_INTERSECTION_KERNEL_H_

#include "search_cta.h"
#include "scan_kernel.h"
#include "memory.h"
#include "kernel_launch.h"

// TODO(slashwang): for unsorted data and query, need two things:
// 1) is_sorted kernel. use __synchthreads_and().
// 2) load query (smaller) to shmem several rounds.
// each round for each item in the large data array, do binary search
// combine results from all rounds. compact the indices to the output.

namespace plas {
namespace gpu {

// Keys-only intersection.
template <bool duplicates, typename launch_arg_t = empty_t, typename a_keys_it,
          typename b_keys_it, typename c_indices_it, typename comp_t>
int intersect_indices(a_keys_it a_keys, int a_count, b_keys_it b_keys, int b_count,
              c_indices_it c_indices, comp_t comp, context_t& context) {
  typedef typename conditional_typedef_t<
      launch_arg_t, launch_box_t<arch_35_cta<128, 15> > >::type_t launch_t;

  typedef typename std::iterator_traits<a_keys_it>::value_type type_t;
  typedef typename std::iterator_traits<c_indices_it>::value_type indices_type_t;

  mem_t<int> partitions = balanced_path_partitions<duplicates>(
      a_keys, a_count, b_keys, b_count, launch_t::nv(context), comp, context);
  int* mp_data = partitions.data();

  int num_ctas = launch_t::cta_dim(context).num_ctas(a_count + b_count);
  std::vector<int> partitions_vec = from_mem(partitions);

  mem_t<int> counts(num_ctas + 1, context);
  int* counts_data = counts.data();
  // intersection will first put all a_count in keys_temp
  // then compact to final total according to intersect flags.
  mem_t<indices_type_t> indices_temp(launch_t::nv(context) * num_ctas, context);
  indices_type_t* indices_temp_data = indices_temp.data();

  auto k = [=] PLAS_DEVICE(int tid, int cta) {
    typedef typename launch_t::sm_ptx params_t;
    enum {
      nt = params_t::nt,
      vt = params_t::vt,
      nv = nt * vt
    };
    typedef cta_scan_t<nt, type_t> scan_t;

    __shared__ union {
      struct {
        type_t keys[nt * (vt + 1)];  // NOLINT
        indices_type_t indices[nt * (vt + 1)];  // NOLINT
      } values;
      typename scan_t::storage_t scan;
    } shared;

    // Load the range for this CTA and merge the values into register.
    int mp0 = mp_data[cta + 0];
    int mp1 = mp_data[cta + 1];
    bool extended;
    merge_range_t range1;
    merge_range_t range2;
    merge_pair_t<type_t, vt> intersected;
    compute_intersect_ranges(a_count, b_count, cta, nv, mp0, mp1, extended,
                             range1, range2);

    int commit = cta_intersect_from_mem<nt, vt, duplicates>(
        a_keys, b_keys, range1, range2, extended, intersected, tid, mp0, comp,
        shared.values.keys);
    __syncthreads();

    int output_count = popc(commit);

    int global_start = nv * cta;

    scan_result_t<type_t> scan = scan_t().scan(tid, output_count, shared.scan);
    type_t start = scan.scan;

    iterate<vt>([&](int i) {
      if ((1 << i) & commit) {
        shared.values.indices[start++] = intersected.indices[i];
      }
    });
    __syncthreads();

    // save shared indices to global
    shared_to_mem<nt, vt>(shared.values.indices, tid, scan.reduction,
                          indices_temp_data + global_start, /*sync=*/true);
    if (!tid) counts_data[cta] = scan.reduction;
  };
  cta_transform<launch_t>(k, a_count + b_count, context);

  // Scan Block Counts
  int total;
  mem_t<int> scanned_counts(num_ctas + 1, context);
  int* scanned_counts_data = scanned_counts.data();
  scan(counts_data, num_ctas + 1, scanned_counts_data, context);
  std::vector<int> scanned_counts_h = from_mem(scanned_counts);
  total = scanned_counts_h[num_ctas];

  int nv = launch_t::nv(context);
  auto compact_indices = [=] PLAS_DEVICE(int index) {
    int seg = index / nv;
    int idx_start = ldg(scanned_counts_data + seg);
    int neighbor_length = ldg(scanned_counts_data + seg + 1) - idx_start;
    int local_offset = index % nv;
    if (local_offset >= neighbor_length) return;
    int src_index = idx_start + local_offset;
    c_indices[src_index] = indices_temp_data[index];
  };
  transform(compact_indices, num_ctas * nv, context);

  return total;
}

}  // namespace gpu
}  // namespace plas

#endif
