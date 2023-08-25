#include "../slice_kernel.h"
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

bool verifySlice(thrust::device_vector<int>& output, thrust::device_vector<int>& input,
        std::vector<int>& begin,
        std::vector<int>& size,
        std::vector<int>& input_dims,
        int input_tensor_rank,
        int total_input_size,
        int total_output_size)
{
    // Copy the matrices to the host
    thrust::host_vector<int> input_host(total_input_size);
    thrust::host_vector<int> output_host(total_output_size);

    thrust::copy(input.begin(), input.end(), input_host.begin());
    thrust::copy(output.begin(), output.end(), output_host.begin());
   
    for (int idx = 0; idx < total_output_size; ++idx) {
        int input_idx = 0;
        int output_idx = 0;
        int input_stride = 1;
        int output_stride = 1;
        for (int i = input_tensor_rank-1; i >= 0; --i) {
            int output_coord = idx / output_stride % size[i];
            int input_coord = begin[i] + output_coord;
            input_idx += input_coord * input_stride;
            output_idx += output_coord * output_stride;
            input_stride *= input_dims[i];
            output_stride *= size[i];
        }
        if (output_host[output_idx] != input_host[input_idx]) {
            return false;
        }
    }
    return true;
}

int main(int argc, char** argv) {
  plas::gpu::standard_context_t context;

  CLI::App app{"CUDA Slice Benchmark"};
  int input_tensor_rank = 4;
  std::vector<int> input_dims = {64, 1600, 1600, 3};
  std::vector<int> input_beginning_indices = {0, 0, 0, 0};
  std::vector<int> output_sizes = {64, 1600, 1600, 3};

  app.add_option("-d, --input_dim", input_tensor_rank, "Number of dimensions of the input tensor");
  app.add_option("-i,--input_sizes", input_dims, "Dim sizes of input tensor");
  app.add_option("-j,--input_begins", input_beginning_indices, "begin indices to slice of input tensor");
  app.add_option("-k,--output_sizes", output_sizes, "Dim sizes of output tensor");

  CLI11_PARSE(app, argc, argv);

  // Prepare input data
  // lambda function to compute the total size of a tensor
  auto tensor_size = [&](const std::vector<int>& size) {
      return thrust::reduce(size.begin(), size.end(), 1, thrust::multiplies<int>());
  };

  int total_input_size = tensor_size(input_dims);
  int total_output_size = tensor_size(output_sizes);

  // Initialize input and output
  thrust::device_vector<int> input(total_input_size);
  thrust::device_vector<int> output(total_output_size);
  plas::gpu::GetRandomIntDeviceArray(input, 1, 10, total_input_size);

  // Copy input_dims beginning_of_indices and output_sizes to device arrays
  thrust::device_vector<int> d_input_dims(input_dims.begin(), input_dims.end());
  thrust::device_vector<int> d_input_beginning_indices(input_beginning_indices.begin(), input_beginning_indices.end());
  thrust::device_vector<int> d_output_sizes(output_sizes.begin(), output_sizes.end());

  std::cout << "input tensor rank: " << input_tensor_rank << std::endl;
  std::cout << "total input tensor size: " << total_input_size << std::endl;
  std::cout << "total output tensor size: " << total_output_size << std::endl;
  for (int i = 0; i < input_tensor_rank; ++i) {
      std::cout << "input dims: " << input_dims[i] << std::endl;
  }
  for (int i = 0; i < input_tensor_rank; ++i) {
      std::cout << "input begins: " << input_beginning_indices[i] << std::endl;
  }
  for (int i = 0; i < input_tensor_rank; ++i) {
      std::cout << "output sizes: " << output_sizes[i] << std::endl;
  }

  // Launch the kernel
  int* inputs = thrust::raw_pointer_cast(input.data());
  int* outputs = thrust::raw_pointer_cast(output.data());
  int* d_input_dims_ptr = thrust::raw_pointer_cast(d_input_dims.data());
  int* d_input_beginning_indices_ptr = thrust::raw_pointer_cast(d_input_beginning_indices.data());
  int* d_output_sizes_ptr = thrust::raw_pointer_cast(d_output_sizes.data());

  context.timer_begin();
  for (int i = 0; i < 10; ++i) {
    plas::gpu::slice(inputs, total_output_size, outputs, d_input_dims_ptr,
    d_input_beginning_indices_ptr, d_output_sizes_ptr, input_tensor_rank,
    context);
  }
  double elapsed = context.timer_end();
  std::cout << "GPU time: " << elapsed / 10 << "msec." << std::endl;

  std::cout << "Memory bandwidth: " << total_output_size * sizeof(int) * 2 * 10 / elapsed / 1e6 << "GB/s." << std::endl;

  // Verify results.
  bool result = verifySlice(output, input, input_beginning_indices, output_sizes, input_dims, input_tensor_rank, total_input_size, total_output_size);
  if (result)
    std::cout << "Verification succeeded" << std::endl;
  else
    std::cout << "Verification failed" << std::endl;

  return 0;
}
