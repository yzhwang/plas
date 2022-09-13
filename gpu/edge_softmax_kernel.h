#ifndef PLAS_GPU_EDGE_SOFTMAX_KERNEL_H_
#define PLAS_GPU_EDGE_SOFTMAX_KERNEL_H_

#include <limits>

#include "memory.h"
#include "kernel_launch.h"

namespace plas {
namespace gpu {

/**
 * @brief compute edge softmax when segments are on average small
 *        (here when segments are < 16 we call them small).
 *
 * @tparam input_it:     type of input;
 * @tparam output_it:    type of output;
 * @tparam index_it:     type of index;
 *
 * @param input:       feature vectors to be gathered and reduced (vdata);
 * @param seg_offsets: offset of each segment, seg_offsets[seg_count] stores
 *                     edge count;
 * @param seg_count:   number of segments;
 * @param hidden_dim:  hidden dimension size of the feature vector;
 * @param output:      feature vectors to store reduction results (udata);
 * @param context:     reference to CUDA context singleton instance.
 */
template <typename input_it, typename output_it, typename index_it>
void edge_softmax_small(input_it input, index_it seg_offsets, int seg_count,
                        int hidden_dim, output_it output, context_t& context) {
  typedef typename std::iterator_traits<input_it>::value_type type_t;
  auto edge_softmax = [=] PLAS_DEVICE(int index) {
    int seg = index / hidden_dim;
    int idx_start = ldg(seg_offsets + seg);
    int neighbor_length = ldg(seg_offsets + seg + 1) - idx_start;
    int local_offset = index % hidden_dim;
    type_t local_max = std::numeric_limits<type_t>::lowest();
    for (int i = idx_start; i < idx_start + neighbor_length; ++i) {
      type_t data = ldg(input + i * hidden_dim + local_offset);
      local_max = max(local_max, data);
    }
    type_t local_sum = 0;
    for (int i = idx_start; i < idx_start + neighbor_length; ++i) {
      type_t data = ldg(input + i * hidden_dim + local_offset);
      type_t exp_data = exp(data - local_max);
      local_sum = local_sum + exp_data;
      output[i * hidden_dim + local_offset] = exp_data;
    }
    for (int i = idx_start; i < idx_start + neighbor_length; ++i) {
      output[i * hidden_dim + local_offset] /= local_sum;
    }
  };
  transform(edge_softmax, seg_count * hidden_dim, context);
}

/**
 * @brief compute edge softmax gradient when segments are on average small
 *        (here when segments are < 16 we call them small).
 *
 * @tparam input_it:     type of input;
 * @tparam output_it:    type of output;
 * @tparam index_it:     type of index;
 *
 * @param input:       feature vectors to be gathered and reduced (vdata);
 * @param input_grad:  grad of input
 * @param seg_offsets: offset of each segment, seg_offsets[seg_count] stores
 *                     edge count;
 * @param seg_count:   number of segments;
 * @param hidden_dim:  hidden dimension size of the feature vector;
 * @param output:      feature vectors to store reduction results (udata);
 * @param context:     reference to CUDA context singleton instance.
 */
template <typename input_it, typename output_it, typename index_it>
void edge_softmax_bkw_small(input_it input, input_it input_grad,
                            index_it seg_offsets, int seg_count, int hidden_dim,
                            output_it output, context_t& context) {
  typedef typename std::iterator_traits<input_it>::value_type type_t;
  auto edge_softmax_bkw = [=] PLAS_DEVICE(int index) {
    int seg = index / hidden_dim;
    int idx_start = ldg(seg_offsets + seg);
    int neighbor_length = ldg(seg_offsets + seg + 1) - idx_start;
    int local_offset = index % hidden_dim;
    type_t accum = 0.0f;
    for (int i = idx_start; i < idx_start + neighbor_length; ++i) {
      type_t data = ldg(input + i * hidden_dim + local_offset);
      type_t data_grad = ldg(input_grad + i * hidden_dim + local_offset);
      accum -= data * data_grad;
    }
    for (int i = idx_start; i < idx_start + neighbor_length; ++i) {
      type_t data = ldg(input + i * hidden_dim + local_offset);
      type_t data_grad = ldg(input_grad + i * hidden_dim + local_offset);
      output[i * hidden_dim + local_offset] = accum * data + data_grad * data;
    }
  };
  transform(edge_softmax_bkw, seg_count * hidden_dim, context);
}

}  // namespace gpu
}  // namespace plas

#endif
