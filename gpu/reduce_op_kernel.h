#ifndef PLAS_GPU_REDUCE_OP_KERNEL_H_
#define PLAS_GPU_REDUCE_OP_KERNEL_H_

#include "reduce_cta.h"
#include "memory.h"
#include "kernel_launch.h"

namespace plas {
namespace gpu {

enum bcast_type {
  ebcast,
  vbcast,
  nobcast
};

/**
 * @brief reduce segments of feature vectors when segments are on average large.
 *
 * @tparam launch_arg_t: launch arguments for specific GPU compute capability;
 * @tparam input_it:     type of input;
 * @tparam output_it:    type of output;
 * @tparam index_it:     type of index;
 * @tparam reduce_op_t:  type of reduce operator;
 * @tparam ewise_op_t:   type of ewise operator;
 * @tparam bcast:        bcast type, can be ebcast or vbcast.
 * @tparam shmem_dim:    length of feature vector that holds in shared memory.
 *
 * @param input:       feature vectors to be gathered and reduced (vdata);
 * @param input2:      feature vectors/scalars to be operated on (edata);
 * @param seg_offsets: offset of each segment, seg_offsets[seg_count] stores
 *                     edge count;
 * @param seg_count:   number of segments;
 * @param indices:     vertex indices of each neighbor node;
 * @param hidden_dim:  hidden dimension size of the feature vector;
 * @param output:      feature vectors to store reduction results (udata);
 * @param reduce_op:   operator for reduce (plus_t, minimum_t, maximum_t, etc.);
 * @param ewise_op:     operator for ewise op;
 * @param context:     reference to CUDA context singleton instance.
 */
template <typename launch_arg_t = empty_t, typename input_it,
          typename output_it, typename index_it, typename reduce_op_t,
          typename ewise_op_t, bcast_type bcast = nobcast, int shmem_dim = 64>
void seg_vector_reduce_op_large(input_it input, input_it input2,
                                index_it seg_offsets, int seg_count,
                                index_it indices, int hidden_dim,
                                output_it output, reduce_op_t reduce_op,
                                ewise_op_t ewise_op, context_t& context) {
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
    int seg_start = ldg(seg_offsets + seg);
    int seg_end = ldg(seg_offsets + seg + 1);
    int seg_length = seg_end - seg_start;
    int load_vt = nt / shmem_dim;
    int round = div_up(seg_length, load_vt);
    int feat_vec_offset = cta / seg_count;
    int idx, base_idx, data1_idx, data2_idx, shmem_idx;

    // load from mem to shmem
    #pragma unroll
    for (int i = 0; i < round; ++i) {
      idx = seg_start + i * load_vt + tid / shmem_dim;
      if (idx >= seg_end) break;
      base_idx = ldg(indices + idx);
      // row-major to row-major
      data1_idx = (bcast == vbcast)
                      ? base_idx
                      : base_idx * hidden_dim + feat_vec_offset * shmem_dim +
                            tid % shmem_dim;
      data2_idx = (bcast == ebcast) ? idx : idx * hidden_dim +
                                                feat_vec_offset * shmem_dim +
                                                tid % shmem_dim;
      shmem_idx = (i * load_vt + tid / shmem_dim) * shmem_dim + tid % shmem_dim;
      type_t data1 = ldg(input + data1_idx);
      type_t data2 = ldg(input2 + data2_idx);
      shared_reduce.data[shmem_idx] = ewise_op(data1, data2);
    }
    // only when neighbor numbers is larger than 64, this part will be run.
    idx += load_vt;
    while (idx < seg_end && idx >= 64) {
      int new_idx = ldg(indices + idx);
      int new_data1_idx =
          (bcast == vbcast) ? new_idx : new_idx * hidden_dim +
                                            feat_vec_offset * shmem_dim +
                                            tid % shmem_dim;
      int new_data2_idx = (bcast == ebcast)
                              ? idx
                              : idx * hidden_dim + feat_vec_offset * shmem_dim +
                                    tid % shmem_dim;
      type_t data1 = ldg(input + new_data1_idx);
      type_t data2 = ldg(input2 + new_data2_idx);
      type_t data = ewise_op(data1, data2);
      shared_reduce.data[shmem_idx] =
          reduce_op(shared_reduce.data[shmem_idx], data);
      idx += 64;
    }
    __syncthreads();

    // reduce and write back multiple-rounds
    int output_idx = seg * hidden_dim + feat_vec_offset * shmem_dim;
    vec_reduce_t().reduce(tid, output + output_idx, shared_reduce, seg_length,
                          reduce_op);
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
 * @tparam reduce_op_t:  type of reduce operator;
 * @tparam ewise_op_t:   type of ewise operator;
 * @tparam bcast:        whether bcast input2's scalar. If not then assume
 *                       input2's feature vec has the same shape as input1.
 *                       false by default.
 *
 * @param input:       feature vectors to be gathered and reduced (vdata);
 * @param input2:      feature vectors/scalars to be operated on (edata);
 * @param seg_offsets: offset of each segment, seg_offsets[seg_count] stores
 *                     edge count;
 * @param seg_count:   number of segments;
 * @param indices:     vertex indices of each neighbor node;
 * @param hidden_dim:  hidden dimension size of the feature vector;
 * @param output:      feature vectors to store reduction results (udata);
 * @param reduce_op:   operator for reduce (plus_t, minimum_t, maximum_t, etc.);
 * @param ewise_op:    operator for ewise op;
 * @param context:     reference to CUDA context singleton instance.
 */
template <typename input_it, typename output_it, typename index_it,
          typename reduce_op_t, typename ewise_op_t, bcast_type bcast = nobcast>
void seg_vector_reduce_op_small(input_it input, input_it input2,
                                index_it seg_offsets, int seg_count,
                                index_it indices, int hidden_dim, int bcast_dim,
                                output_it output, reduce_op_t reduce_op,
                                ewise_op_t ewise_op, context_t& context) {
  typedef typename std::iterator_traits<input_it>::value_type type_t;
  auto seg_reduce = [=] PLAS_DEVICE(int index) {
    int seg = index / hidden_dim;
    int idx_start = ldg(seg_offsets + seg);
    int neighbor_length = ldg(seg_offsets + seg + 1) - idx_start;
    int local_offset = index % hidden_dim;
    int hidden_dim2 = hidden_dim / bcast_dim;
    int dim_offset = local_offset / hidden_dim2;
    type_t local_reduce = 0;
    output[index] = 0;
    for (int i = idx_start; i < idx_start + neighbor_length; ++i) {
      int idx = ldg(indices + i);
      type_t data1 = (bcast == vbcast)
                         ? ldg(input + idx * bcast_dim + dim_offset)
                         : ldg(input + idx * hidden_dim + local_offset);
      type_t data2 = (bcast == ebcast)
                         ? ldg(input2 + i * bcast_dim + dim_offset)
                         : ldg(input2 + i * hidden_dim + local_offset);
      type_t data = ewise_op(data1, data2);
      local_reduce = reduce_op(local_reduce, data);
    }
    output[index] = reduce_op(output[index], local_reduce);
  };
  transform(seg_reduce, seg_count * hidden_dim, context);
}

/**
 * @brief compute egrad for seg vector reduce v * e to u.
 *
 * @tparam data_it:     type of data;
 * @tparam index_it:     type of index;
 * @tparam reduce_op_t:  type of reduce operator;
 * @tparam ewise_op_t:   type of ewise operator;
 *
 * @param input_grad:  computed grad of u;
 * @param input_eidx:  edge indices;
 *
 * seg_offsets, seg_count, and indices store the transposed CSR.
 * @param seg_offsets: offset of each segment, seg_offsets[seg_count] stores
 *                     edge count;
 * @param seg_count:   number of segments;
 * @param edge_count:  number of edges;
 * @param indices:     vertex indices of each neighbor node;
 *
 * @param hidden_dim:   hidden dimension size of the feature vector;
 * @param vdata:        feature vector values for neighbors in the forward pass;
 * @param output_egrad: egrad vector to output;
 * @param reduce_op:   operator for reduce (plus_t, minimum_t, maximum_t, etc.);
 * @param ewise_op:    operator for ewise op;
 * @param context:     reference to CUDA context singleton instance.
 */
template <typename input_it, typename output_it, typename const_index_it, typename index_it,
typename reduce_op_t,
          typename ewise_op_t, bcast_type bcast = nobcast>
void seg_vector_reduce_op_grad_e_small(
    input_it input_grad, const_index_it input_eidx, index_it seg_offsets,
    int seg_count, int edge_count, index_it indices, int hidden_dim, int bcast_dim,
    input_it vdata, output_it output_egrad, reduce_op_t reduce_op,
    ewise_op_t ewise_op, context_t& context) {
  typedef typename std::iterator_traits<input_it>::value_type type_t;
  typedef typename std::iterator_traits<index_it>::value_type index_t;

  auto egrad_init = [=] PLAS_DEVICE(int index) {
    // TODO(slashwang): should be initialized with identity of reduce_op.
    output_egrad[index] = 0;
  };

  if (bcast == ebcast) {
    transform(egrad_init, edge_count, context);

    auto seg_reduce_egrad_ebcast = [=] PLAS_DEVICE(int index) {
      int idx_start = ldg(seg_offsets + index);
      int neighbor_length = ldg(seg_offsets + index + 1) - idx_start;
      for (int i = idx_start; i < idx_start + neighbor_length; ++i) {
        index_t ii_e = ldg(input_eidx + i);
        index_t ii_u = ldg(indices + i);
        int hidden_dim2 = hidden_dim / bcast_dim;
        for (int kk = 0; kk < bcast_dim; ++kk) {
          for (int k = 0; k < hidden_dim2; ++k) {
            type_t grad_data = ldg(input_grad + ii_u * hidden_dim + kk*hidden_dim2 + k);
            type_t v_data = ldg(vdata + index * hidden_dim + kk*hidden_dim2 + k);
            type_t data = ewise_op(grad_data, v_data);
            output_egrad[ii_e*bcast_dim + kk] = reduce_op(output_egrad[ii_e*bcast_dim+kk], data);
          }
        }
      }
    };
    transform(seg_reduce_egrad_ebcast, seg_count, context);
  } else if (bcast == vbcast) {
    transform(egrad_init, edge_count * hidden_dim, context);

    auto seg_reduce_egrad_vbcast = [=] PLAS_DEVICE(int index) {
      int seg = index / hidden_dim;
      int idx_start = ldg(seg_offsets + seg);
      int neighbor_length = ldg(seg_offsets + seg + 1) - idx_start;
      int local_offset = index % hidden_dim;
      int hidden_dim2 = hidden_dim / bcast_dim;
      int dim_offset = local_offset / hidden_dim2;
      type_t v_data = ldg(vdata + seg * bcast_dim + dim_offset);
      for (int i = idx_start; i < idx_start + neighbor_length; ++i) {
        index_t ii_e = ldg(input_eidx + i);
        index_t ii_u = ldg(indices + i);
        type_t grad_data = ldg(input_grad + ii_u * hidden_dim + local_offset);
        type_t data = ewise_op(grad_data, v_data);
        output_egrad[ii_e * hidden_dim + local_offset] =
            reduce_op(output_egrad[ii_e * hidden_dim + local_offset], data);
      }
    };
    transform(seg_reduce_egrad_vbcast, seg_count * hidden_dim, context);
  } else {
    transform(egrad_init, edge_count * hidden_dim, context);

    auto seg_reduce_egrad_nobcast = [=] PLAS_DEVICE(int index) {
      int seg = index / hidden_dim;
      int idx_start = ldg(seg_offsets + seg);
      int neighbor_length = ldg(seg_offsets + seg + 1) - idx_start;
      int local_offset = index % hidden_dim;
      type_t v_data = ldg(vdata + index);
      for (int i = idx_start; i < idx_start + neighbor_length; ++i) {
        index_t ii_e = ldg(input_eidx + i);
        index_t ii_u = ldg(indices + i);
        type_t grad_data = ldg(input_grad + ii_u * hidden_dim + local_offset);
        type_t data = ewise_op(grad_data, v_data);
        output_egrad[ii_e * hidden_dim + local_offset] =
            reduce_op(output_egrad[ii_e * hidden_dim + local_offset], data);
      }
    };
    transform(seg_reduce_egrad_nobcast, seg_count * hidden_dim, context);
  }
}

/**
 * @brief compute vgrad for seg vector reduce v * e to u.
 *
 * @tparam data_it:     type of data;
 * @tparam index_it:     type of index;
 * @tparam reduce_op_t:  type of reduce operator;
 * @tparam ewise_op_t:   type of ewise operator;
 *
 * @param input_grad:  computed grad of u;
 * @param input_eidx:  edge indices;
 *
 * seg_offsets, seg_count, and indices store the transposed CSR.
 * @param seg_offsets: offset of each segment, seg_offsets[seg_count] stores
 *                     edge count;
 * @param seg_count:   number of segments;
 * @param edge_count:  number of edges;
 * @param indices:     vertex indices of each neighbor node;
 *
 * @param hidden_dim:   hidden dimension size of the feature vector;
 * @param edata:        feature vector values for neighbors in the forward pass;
 * @param output_vgrad: egrad vector to output;
 * @param reduce_op:   operator for reduce (plus_t, minimum_t, maximum_t, etc.);
 * @param ewise_op:    operator for ewise op;
 * @param context:     reference to CUDA context singleton instance.
 */
template <typename input_it, typename output_it, typename const_index_it,
          typename index_it, typename reduce_op_t, typename ewise_op_t,
          bcast_type bcast = nobcast>
void seg_vector_reduce_op_grad_v_small(
    input_it input_grad, const_index_it input_eidx, index_it seg_offsets,
    int seg_count, int edge_count, index_it indices, int hidden_dim, int bcast_dim,
    input_it edata, output_it output_vgrad, reduce_op_t reduce_op,
    ewise_op_t ewise_op, context_t& context) {
  typedef typename std::iterator_traits<input_it>::value_type type_t;
  typedef typename std::iterator_traits<index_it>::value_type index_t;

  auto vgrad_init = [=] PLAS_DEVICE(int index) {
    // TODO(slashwang): should be initialized with identity of reduce_op.
    output_vgrad[index] = 0;
  };

  if (bcast == vbcast) {
    transform(vgrad_init, seg_count, context);
    auto seg_reduce_vgrad_vbcast = [=] PLAS_DEVICE(int index) {
      int idx_start = ldg(seg_offsets + index);
      int neighbor_length = ldg(seg_offsets + index + 1) - idx_start;
      // TODO(slashwang): should be initialized with identity of reduce_op.
      for (int i = idx_start; i < idx_start + neighbor_length; ++i) {
        index_t ii_e = ldg(input_eidx + i);
        index_t ii_u = ldg(indices + i);
        int hidden_dim2 = hidden_dim / bcast_dim;
        for (int kk = 0; kk < bcast_dim; ++kk) {
          for (int k = 0; k < hidden_dim2; ++k) {
            type_t grad_data = ldg(input_grad + ii_u * hidden_dim + kk*hidden_dim2 + k);
            type_t e_data = ldg(edata + ii_e * hidden_dim + kk*hidden_dim2 + k);
            type_t data = ewise_op(grad_data, e_data);
            output_vgrad[index*bcast_dim + kk] = reduce_op(output_vgrad[index*bcast_dim+kk], data);
          }
        }
      }
    };
    transform(seg_reduce_vgrad_vbcast, seg_count, context);
  } else if (bcast == ebcast) {
    transform(vgrad_init, seg_count * hidden_dim, context);

    auto seg_reduce_vgrad_ebcast = [=] PLAS_DEVICE(int index) {
      int seg = index / hidden_dim;
      int idx_start = ldg(seg_offsets + seg);
      int neighbor_length = ldg(seg_offsets + seg + 1) - idx_start;
      int local_offset = index % hidden_dim;
      int hidden_dim2 = hidden_dim / bcast_dim;
      int dim_offset = local_offset / hidden_dim2;
      // TODO(slashwang): should be initialized with identity of reduce_op.
      for (int i = idx_start; i < idx_start + neighbor_length; ++i) {
        index_t ii_e = ldg(input_eidx + i);
        index_t ii_u = ldg(indices + i);
        type_t grad_data = ldg(input_grad + ii_u * hidden_dim + local_offset);
        type_t e_data = ldg(edata + ii_e * bcast_dim + dim_offset);
        type_t data = ewise_op(grad_data, e_data);
        output_vgrad[index] = reduce_op(output_vgrad[index], data);
      }
    };
    transform(seg_reduce_vgrad_ebcast, seg_count * hidden_dim, context);
  } else {
    transform(vgrad_init, seg_count * hidden_dim, context);

    auto seg_reduce_vgrad_nobcast = [=] PLAS_DEVICE(int index) {
      int seg = index / hidden_dim;
      int idx_start = ldg(seg_offsets + seg);
      int neighbor_length = ldg(seg_offsets + seg + 1) - idx_start;
      int local_offset = index % hidden_dim;
      // TODO(slashwang): should be initialized with identity of reduce_op.
      for (int i = idx_start; i < idx_start + neighbor_length; ++i) {
        index_t ii_e = ldg(input_eidx + i);
        index_t ii_u = ldg(indices + i);
        type_t grad_data = ldg(input_grad + ii_u * hidden_dim + local_offset);
        type_t e_data = ldg(edata + ii_e * hidden_dim + local_offset);
        type_t data = ewise_op(grad_data, e_data);
        output_vgrad[index] = reduce_op(output_vgrad[index], data);
      }
    };
    transform(seg_reduce_vgrad_nobcast, seg_count * hidden_dim, context);
  }
}

}  // namespace gpu
}  // namespace plas

#endif
