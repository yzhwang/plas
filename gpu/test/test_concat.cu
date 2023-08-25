#include "../kernel_launch.h"
#include "../memory.h"
#include "../loadstore.h"
#include <iostream>
#include <vector>
#include <algorithm>
#include "../CLI11.hpp"

#include <thrust/copy.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/generate.h>
#include <thrust/random/linear_congruential_engine.h>
#include <thrust/random/uniform_int_distribution.h>

bool verifyConcat(thrust::device_vector<int>& output,
        thrust::device_vector<int>& input_1,
        thrust::device_vector<int>& input_2,
        std::vector<int>& output_dims,
        std::vector<int>& input_dims,
        int concat_dim,
        int input_tensor_rank,
        int total_input_size,
        int total_output_size)
{
    // Copy the matrices to the host
    thrust::host_vector<int> input_1_host(total_input_size);
    thrust::host_vector<int> input_2_host(total_input_size);
    thrust::host_vector<int> output_host(total_output_size);

    thrust::copy(input_1.begin(), input_1.end(), input_1_host.begin());
    thrust::copy(input_2.begin(), input_2.end(), input_2_host.begin());
    thrust::copy(output.begin(), output.end(), output_host.begin());
   
    for (int idx = 0; idx < total_output_size; ++idx) {
      int input_idx = 0;
      int output_idx = 0;
      int input_stride = 1;
      int output_stride = 1;
      int which_input = 0;
      for (int dim_idx = input_tensor_rank-1; dim_idx >= 0; --dim_idx) {
        int output_coord, input_coord;
        input_coord = output_coord = idx / output_stride % input_dims[dim_idx];
        input_idx += input_coord * input_stride;
        output_idx += output_coord * output_stride;
        input_stride *= input_dims[dim_idx];
        output_stride *= output_dims[dim_idx];
        if (dim_idx == concat_dim) {
          which_input = input_coord / input_dims[dim_idx];
        }
      }
      int input_val = which_input == 0 ? input_1_host[input_idx] : input_2_host[input_idx];
      if (output_host[output_idx] != input_val) {
          return false;
      }
    }
    return true;
}

int main(int argc, char** argv) {
  plas::gpu::standard_context_t context;

  CLI::App app{"CUDA Concat Benchmark"};
  int input_tensor_rank = 4;
  std::vector<int> input_dims = {64, 1600, 1600, 3};
  std::vector<int> output_dims;
  int concat_dim = 2;

  app.add_option("-d, --input_dim", input_tensor_rank, "Number of dimensions of the input tensor");
  app.add_option("-i,--input_sizes", input_dims, "Dim sizes of input tensor");
  app.add_option("-c,--concat_dim", concat_dim, "Concat dimension");

  CLI11_PARSE(app, argc, argv);

  // Prepare input data
  // lambda function to compute the total size of a tensor
  auto tensor_size = [&](const std::vector<int>& size) {
      return thrust::reduce(size.begin(), size.end(), 1, thrust::multiplies<int>());
  };

  int total_input_size = tensor_size(input_dims);
  int total_output_size = total_input_size * 2;

  for (int i = 0; i < input_tensor_rank; ++i) {
      if (i == concat_dim) {
          output_dims.push_back(input_dims[i] * 2);
      } else {
          output_dims.push_back(input_dims[i]);
      }
  }

  // Initialize input and output
  thrust::device_vector<int> input_1(total_input_size);
  thrust::device_vector<int> input_2(total_input_size);
  thrust::device_vector<int> output(total_output_size);
  plas::gpu::GetRandomIntDeviceArray(input_1, 1, 10, total_input_size);
  plas::gpu::GetRandomIntDeviceArray(input_2, 10, 20, total_input_size);

  // Copy input_dims beginning_of_indices and output_sizes to device arrays
  thrust::device_vector<int> d_input_dims(input_dims.begin(), input_dims.end());
  thrust::device_vector<int> d_output_dims(output_dims.begin(), output_dims.end());

  std::cout << "input tensor rank: " << input_tensor_rank << std::endl;
  std::cout << "total input tensor size: " << total_input_size << std::endl;
  std::cout << "total output tensor size: " << total_output_size << std::endl;
  for (int i = 0; i < input_tensor_rank; ++i) {
      std::cout << "input dims: " << input_dims[i] << std::endl;
  }
  for (int i = 0; i < input_tensor_rank; ++i) {
      std::cout << "output sizes: " << output_dims[i] << std::endl;
  }

  // Launch the kernel
  int* inputs_1 = thrust::raw_pointer_cast(input_1.data());
  int* inputs_2 = thrust::raw_pointer_cast(input_2.data());
  int* outputs = thrust::raw_pointer_cast(output.data());
  int* d_input_dims_ptr = thrust::raw_pointer_cast(d_input_dims.data());
  int* d_output_dims_ptr = thrust::raw_pointer_cast(d_output_dims.data());


  typedef plas::gpu::launch_box_t<plas::gpu::arch_80_cta<512, 7>> launch_t;

  auto concat_kernel = [=] PLAS_DEVICE(int index) {
    typedef typename launch_t::sm_ptx params_t;
    enum {
      nt = params_t::nt,
      vt = params_t::vt,
      nv = nt * vt
    };

    __shared__ int s_vals[nv+1];

    __shared__ int r_input_dims[8];
    __shared__ int r_output_dims[8];

    if (index % nv == 0) {
      for (int i = 0; i < input_tensor_rank; ++i) {
        r_input_dims[i] = plas::gpu::ldg(d_input_dims_ptr+i);
        r_output_dims[i] = plas::gpu::ldg(d_output_dims_ptr+i);
      }
    }
    __syncthreads();

    int input_idx = 0;
    int output_idx = 0;
    int input_stride = 1;
    int output_stride = 1;
    int which_input = 0;
    for (int dim_idx = input_tensor_rank-1; dim_idx >= 0; --dim_idx) {
      int output_coord, input_coord;
      input_coord = output_coord = index / output_stride % r_input_dims[dim_idx];
      input_idx += input_coord * input_stride;
      output_idx += output_coord * output_stride;
      input_stride *= r_input_dims[dim_idx];
      output_stride *= r_output_dims[dim_idx];
      if (dim_idx == concat_dim) {
        which_input = input_coord / r_input_dims[dim_idx];
      }
    }
    int s_idx = index % nv;
    if (which_input == 0) {
      s_vals[s_idx] = plas::gpu::ldg(inputs_1+input_idx);
    } else {
      s_vals[s_idx] = plas::gpu::ldg(inputs_2+input_idx);
    }
    __syncthreads();
    outputs[index] = s_vals[s_idx];
  };

  context.timer_begin();
  for (int i = 0; i < 10; ++i) {
    plas::gpu::transform<launch_t>(concat_kernel, total_output_size, context);
  }
  double elapsed = context.timer_end();
  std::cout << "GPU time: " << elapsed / 10 << "msec." << std::endl;

  std::cout << "Memory bandwidth: " << total_output_size * sizeof(int) * 2 * 10 / elapsed / 1e6 << "GB/s." << std::endl;

  // Verify results.
  bool result = verifyConcat(output, input_1, input_2, output_dims, input_dims, concat_dim, input_tensor_rank, total_input_size, total_output_size);
    if (result)
        std::cout << "Verification succeeded" << std::endl;
    else
        std::cout << "Verification failed" << std::endl;

  return 0;
}
