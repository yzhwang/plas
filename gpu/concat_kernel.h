#ifndef PLAS_GPU_CONCAT_KERNEL_H_
#define PLAS_GPU_CONCAT_KERNEL_H_

#include "memory.h"
#include "loadstore.h"
#include "kernel_launch.h"

namespace plas {
namespace gpu {
template <typename launch_arg_t = empty_t, typename input_it,
          typename output_it>
void slice(input_it input, int count, output_it output, int* d_input_dims_ptr,
  int* d_input_beginning_indices_ptr, int* d_output_sizes_ptr,
  int input_tensor_rank, context_t& context) {

  typedef typename conditional_typedef_t<
      launch_arg_t, launch_params_t<128, 7> >::type_t launch_t;
  typedef typename std::iterator_traits<input_it>::value_type type_t;
  int num_ctas = launch_t::cta_dim(context).num_ctas(count);

  auto k = [=] PLAS_DEVICE(int tid, int cta) {
    typedef typename launch_t::sm_ptx params_t;
    enum {
      nt = params_t::nt,
      vt = params_t::vt,
      nv = nt * vt
    };

    __shared__ int r_input_dims[8];
    __shared__ int r_input_beginning_indices[8];
    __shared__ int r_output_sizes[8];

    if (tid % nv == 0) {
      for (int i = 0; i < input_tensor_rank; ++i) {
        r_input_dims[i] = plas::gpu::ldg(d_input_dims_ptr+i);
        r_input_beginning_indices[i] = plas::gpu::ldg(d_input_beginning_indices_ptr+i);
        r_output_sizes[i] = plas::gpu::ldg(d_output_sizes_ptr+i);
      }
    }
    __syncthreads();

    // Load the tile's data into register.
    range_t tile = get_tile(cta, nv, count);
    array_t<type_t, vt> x;

    strided_iterate<nt, vt>([&](int i, int j) {
      int input_idx = 0;
      int input_stride = 1;
      int output_stride = 1;
      for (int idx = input_tensor_rank-1; idx >= 0; --idx) {
        int input_coord = r_input_beginning_indices[idx] + (tile.begin + j) / output_stride % r_output_sizes[idx];
        input_idx += input_coord * input_stride;
        input_stride *= r_input_dims[idx];
        output_stride *= r_output_sizes[idx];
      }
      x[i] = plas::gpu::ldg(input+input_idx);
		}, tid, tile.count());
    // Store the tile's data into global memory.
    reg_to_mem_strided<nt, vt>(x, tid, tile.count(), output + tile.begin);
  };
  cta_launch<launch_t>(k, num_ctas, context);
}

}  // namespace gpu
}  // namespace plas

#endif
