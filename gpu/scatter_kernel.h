#ifndef PLAS_GPU_SCATTER_KERNEL_H_
#define PLAS_GPU_SCATTER_KERNEL_H_

#include "memory.h"
#include "kernel_launch.h"

namespace plas {
namespace gpu {

/**
 * @brief scatter to segments of feature vectors.
 *
 * @tparam launch_arg_t: launch arguments for specific GPU compute capability;
 * @tparam input_it:     type of input;
 * @tparam output_it:    type of output;
 * @tparam index_it:     type of index;
 * @tparam shmem_dim:    length of feature vector that holds in shared memory.
 *
 * @param input:       feature vectors to be scattered (udata);
 * @param seg_offsets: offset of each segment, seg_offsets[seg_count] stores
 *                     edge count;
 * @param seg_count:   number of segments;
 * @param indices:     vertex indices of each neighbor node;
 * @param hidden_dim:  hidden dimension size of the feature vector;
 * @param output:      feature vectors to store scattered results (vdata);
 * @param context:     reference to CUDA context singleton instance.
 */
template <typename launch_arg_t = empty_t, typename input_it,
          typename output_it, typename index_it, int shmem_dim = 64>
void seg_vector_scatter_small(input_it input, index_it seg_offsets,
                              int seg_count, int hidden_dim, output_it output,
                              context_t& context) {
  auto seg_scatter = [=] PLAS_DEVICE(int index) {
    int seg = index / hidden_dim;
    int idx_start = ldg(seg_offsets + seg);
    int neighbor_length = ldg(seg_offsets + seg + 1) - idx_start;
    int local_offset = index % hidden_dim;
    float origin_data = ldg(input + index);
    for (int i = idx_start; i < idx_start + neighbor_length; ++i) {
      output[i * hidden_dim + local_offset] = origin_data;
    }
  };
  transform(seg_scatter, seg_count * hidden_dim, context);
}

}  // namespace gpu
}  // namespace plas

#endif
