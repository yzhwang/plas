#ifndef PLAS_GPU_REDUCE_KERNEL_H_
#define PLAS_GPU_REDUCE_KERNEL_H_

#include "reduce_cta.h"
#include "memory.h"
#include "kernel_launch.h"

namespace plas {
namespace gpu {

/**
 * @brief reduce segments of feature vectors when segments are on average large.
 *
 * @tparam launch_arg_t: launch arguments for specific GPU compute capability;
 * @tparam input_it:     type of input;
 * @tparam output_it:    type of output;
 * @tparam index_it:     type of index;
 * @tparam op_t:         type of operator;
 * @tparam shmem_dim:    length of feature vector that holds in shared memory.
 *
 * @param input:       feature vectors to be gathered and reduced (vdata);
 * @param seg_offsets: offset of each segment, seg_offsets[seg_count] stores
 *                     edge count;
 * @param seg_count:   number of segments;
 * @param indices:     vertex indices of each neighbor node;
 * @param hidden_dim:  hidden dimension size of the feature vector;
 * @param output:      feature vectors to store reduction results (udata);
 * @param op:          operator for reduce (plus_t, minimum_t, maximum_t, etc.);
 * @param context:     reference to CUDA context singleton instance.
 */
template <typename launch_arg_t = empty_t, typename input_it,
          typename output_it, typename index_it, typename op_t,
          int shmem_dim = 64>
void seg_vector_reduce_large(input_it input, index_it seg_offsets,
                             int seg_count, index_it indices, int hidden_dim,
                             output_it output, op_t op, context_t& context) {
  assert((hidden_dim % shmem_dim == 0));

  typedef typename conditional_typedef_t<
      launch_arg_t, launch_params_t<512, 4> >::type_t launch_t;

  typedef typename std::iterator_traits<input_it>::value_type type_t;

  // Each CTA will do reduce a segment to a single feature vector.
  // For cases where hidden_dim > 64, it takes more CTAs to finish.
  int num_ctas = seg_count * hidden_dim / shmem_dim;

  auto k = [=] PLAS_DEVICE(int tid, int cta) {
    typedef typename launch_t::sm_ptx params_t;
    enum {
      nt = params_t::nt,
      vt = params_t::vt
    };

    // claim a 4K element shmem region to store elements to reduce
    // elements are 64-length vector with at most 64 of them.
    typedef cta_vector_reduce_t<nt, vt, shmem_dim, type_t> vec_reduce_t;
    __shared__ typename vec_reduce_t::storage_t shared_reduce;

    int seg = cta % seg_count;
    int seg_start = seg_offsets[seg];
    int seg_end = seg_offsets[seg + 1];
    int seg_length = seg_end - seg_start;
    int load_vt = nt / shmem_dim;
    int round = div_up(seg_length, load_vt);
    int feat_vec_offset = cta / seg_count;
    int idx, base_idx, data_idx, shmem_idx;

    // load from mem to shmem
    #pragma unroll
    for (int i = 0; i < round; ++i) {
      idx = seg_start + i * load_vt + tid / shmem_dim;
      if (idx >= seg_end) break;
      base_idx = (nullptr == indices) ? idx : ldg(indices + idx);
      // row-major to row-major
      data_idx =
          base_idx * hidden_dim + feat_vec_offset * shmem_dim + tid % shmem_dim;
      shmem_idx = (i * load_vt + tid / shmem_dim) * shmem_dim + tid % shmem_dim;
      shared_reduce.data[shmem_idx] = ldg(input + data_idx);
    }
    // only when neighbor numbers is larger than 64, this part will be run.
    idx += load_vt;
    while (idx < seg_end && idx >= 64) {
      int new_idx = (nullptr == indices) ? idx : ldg(indices + idx);
      int new_data_idx =
          new_idx * hidden_dim + feat_vec_offset * shmem_dim + tid % shmem_dim;

      shared_reduce.data[shmem_idx] =
          op(shared_reduce.data[shmem_idx], ldg(input + new_data_idx));
      idx += 64;
    }
    __syncthreads();

    // reduce and write back multiple-rounds
    int output_idx = seg * hidden_dim + feat_vec_offset * shmem_dim;
    vec_reduce_t().reduce(tid, output + output_idx, shared_reduce, seg_length,
                          op);
  };
  cta_launch<launch_t>(k, num_ctas, context);
}

/**
 * @brief reduce segments of feature vectors when segments are on average small
 *        (here when segments are < 16 we call them small).
 *
 * @tparam input_it:     type of input;
 * @tparam output_it:    type of output;
 * @tparam index_it:     type of index;
 * @tparam op_t:         type of operator;
 * @tparam shmem_dim:    length of feature vector that holds in shared memory.
 *
 * @param input:       feature vectors to be gathered and reduced (vdata);
 * @param seg_offsets: offset of each segment, seg_offsets[seg_count] stores
 *                     edge count;
 * @param seg_count:   number of segments;
 * @param indices:     vertex indices of each neighbor node;
 * @param hidden_dim:  hidden dimension size of the feature vector;
 * @param output:      feature vectors to store reduction results (udata);
 * @param op:          operator for reduce (plus_t, minimum_t, maximum_t, etc.);
 * @param context:     reference to CUDA context singleton instance.
 */
template <typename input_it, typename output_it, typename index_it,
          typename op_t, int shmem_dim = 64>
void seg_vector_reduce_small(input_it input, index_it seg_offsets,
                             int seg_count, index_it indices, int hidden_dim,
                             output_it output, op_t op, context_t& context) {
  auto seg_reduce = [=] PLAS_DEVICE(int index) {
    int seg = index / hidden_dim;
    int idx_start = ldg(seg_offsets + seg);
    int neighbor_length = ldg(seg_offsets + seg + 1) - idx_start;
    int local_offset = index % hidden_dim;
    float local_reduce = 0;
    output[index] = 0;
    for (int i = idx_start; i < idx_start + neighbor_length; ++i) {
      int idx = (nullptr == indices) ? i : ldg(indices + i);
      float data = ldg(input + idx * hidden_dim + local_offset);
      local_reduce = local_reduce + data;
    }
    output[index] += local_reduce;
  };
  transform(seg_reduce, seg_count * hidden_dim, context);
}

}  // namespace gpu
}  // namespace plas

#endif
