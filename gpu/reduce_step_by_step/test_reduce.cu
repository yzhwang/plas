#include "reduce.h"
#include "../memory.h"
#include <iostream>
#include <vector>
#include <algorithm>
#include "../CLI11.hpp"

#include <cub/device/device_reduce.cuh>

template <typename input_it, typename output_it>
void reduce_sum_kernel(input_it input, int count, output_it output, plas::gpu::standard_context_t& context, int op) {
  // Define the function object for the bound kernel function
    std::function<void(input_it, int, output_it, plas::gpu::standard_context_t&)> bound_kernel;
    // Check the value of command line option "p"
    switch (op) {
        default:
        case 0:
            bound_kernel = std::bind(&plas::gpu::reduce_baseline<input_it, output_it>,
			    std::placeholders::_1,
			    std::placeholders::_2,
			    std::placeholders::_3,
                            std::placeholders::_4);
            break;
        case 1:
	    bound_kernel = std::bind(&plas::gpu::reduce_interleave<input_it, output_it>,
			    std::placeholders::_1,
			    std::placeholders::_2,
			    std::placeholders::_3,
			    std::placeholders::_4);
	    break;
	case 2:
	    bound_kernel = std::bind(&plas::gpu::reduce_bank_conflict_free<input_it, output_it>,
			    std::placeholders::_1,
			    std::placeholders::_2,
			    std::placeholders::_3,
			    std::placeholders::_4);
	    break;
	case 3:
	    bound_kernel = std::bind(&plas::gpu::reduce_idle_thread<input_it, output_it>,
			    std::placeholders::_1,
			    std::placeholders::_2,
			    std::placeholders::_3,
			    std::placeholders::_4);
            break;
	case 4:
	    bound_kernel = std::bind(&plas::gpu::reduce_warp_1<input_it, output_it>,
			    std::placeholders::_1,
			    std::placeholders::_2,
			    std::placeholders::_3,
			    std::placeholders::_4);
  case 5:
	    bound_kernel = std::bind(&plas::gpu::reduce_warp_multi_add<input_it, output_it>,
			    std::placeholders::_1,
			    std::placeholders::_2,
			    std::placeholders::_3,
			    std::placeholders::_4);
            break;
	case 6:
	    bound_kernel = std::bind(&plas::gpu::reduce_final<input_it, output_it>,
			    std::placeholders::_1,
			    std::placeholders::_2,
			    std::placeholders::_3,
			    std::placeholders::_4);
            break;
    }
    bound_kernel(input, count, output, context);
}

int main(int argc, char** argv) {
  plas::gpu::standard_context_t context;
  
  CLI::App app{"CUDA Reduce Benchmark"};
  int input_tensor_rank = 3;
  std::vector<int> input_dims = {1024, 1024, 32};
  int option = 0;
  bool cub_test = false;

  app.add_option("-d, --input_dim", input_tensor_rank, "Number of dimensions of the input tensor");
  app.add_option("-i,--input_sizes", input_dims, "Dim sizes of input tensor");
  app.add_option("-p, --option", option, "type of kernel used for test");
  app.add_option("-u, --cub", cub_test, "whether to verify results with CPU code");


  CLI11_PARSE(app, argc, argv);

  // Prepare input data
  // lambda function to compute the total size of a tensor
  auto tensor_size = [&](const std::vector<int>& size) {
      return thrust::reduce(size.begin(), size.end(), 1, thrust::multiplies<int>());
  };

  int count = tensor_size(input_dims);

  // Prepare test data.
  plas::gpu::mem_t<int> input_d =
      plas::gpu::fill_random(0, 10, count,
                             /*sorted=*/false, context);
  plas::gpu::mem_t<int> output_d = plas::gpu::fill(0, 1, context);
  std::vector<int> input_h = plas::gpu::from_mem(input_d);
  std::vector<int> ref_output_h(1, 0);
  for (int i = 0; i < count; ++i) {
    ref_output_h[0] += input_h[i];
  }

  // Launch the kernel
  int* output = output_d.data();
  int* input = input_d.data();

  context.timer_begin();
  for (int i = 0; i < 10; ++i) {
    reduce_sum_kernel(input, count, output, context, option);
  }
  double elapsed = context.timer_end();
  std::cout << "GPU time: " << elapsed / 10 << "msec." << std::endl; 

  // Verify results.
  std::vector<int> output_h = plas::gpu::from_mem(output_d);

  bool equal = (ref_output_h[0] == output_h[0]);
  std::cout << (equal ? "correct!" : "error!") << std::endl;

  // Allocate temporary storage for the reduction operation
  if (cub_test) {
    size_t temp_storage_bytes = 0;
    void* d_temp_storage = nullptr;
    cub::DeviceReduce::Sum(d_temp_storage, temp_storage_bytes,
                            input, output,
                            count);
    cudaMalloc(&d_temp_storage, temp_storage_bytes);

    context.timer_begin();
    for (int i = 0; i < 10; ++i) {
      // Perform the reduction operation and store the result in the first element of d_data
      cub::DeviceReduce::Sum(d_temp_storage, temp_storage_bytes,
                            input, output,
                            count);
    }
    elapsed = context.timer_end();
    std::cout << "CUB time: " << elapsed / 10 << "msec." << std::endl;

    // Free the temporary storage
    cudaFree(d_temp_storage);
  }

  return 0;
}
