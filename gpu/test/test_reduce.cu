#include "../reduce_kernel.h"
#include "../memory.h"
#include <iostream>
#include <vector>
#include <algorithm>
#include "../CLI11.hpp"

int main(int argc, char** argv) {
  plas::gpu::standard_context_t context;
  
  CLI::App app{"CUDA Reduce Benchmark"};
  int input_tensor_rank = 4;
  std::vector<int> input_dims = {64, 1600, 1600, 3};

  app.add_option("-d, --input_dim", input_tensor_rank, "Number of dimensions of the input tensor");
  app.add_option("-i,--input_sizes", input_dims, "Dim sizes of input tensor");

  CLI11_PARSE(app, argc, argv);

  // Prepare input data
  // lambda function to compute the total size of a tensor
  auto tensor_size = [&](const std::vector<int>& size) {
      return thrust::reduce(size.begin(), size.end(), 1, thrust::multiplies<int>());
  };

  int count = tensor_size(input_dims);

  // Prepare test data.
  plas::gpu::mem_t<int> degrees_d =
      plas::gpu::fill_random(10, 12, count,
                             /*sorted=*/false, context);
  plas::gpu::mem_t<int> output_d = plas::gpu::fill(0, 1, context);
  std::vector<int> degrees_h = plas::gpu::from_mem(degrees_d);
  std::vector<int> ref_output_h(1, 0);
  for (int i = 0; i < count; ++i) {
    ref_output_h[0] += degrees_h[i];
  }

  // Launch the kernel
  int* output = output_d.data();
  int* input = degrees_d.data();

  context.timer_begin();
  for (int i = 0; i < 10; ++i) {
    plas::gpu::reduce(input, count, output, plas::gpu::plus_t<int>(), context);
  }
  double elapsed = context.timer_end();
  std::cout << "GPU time: " << elapsed / 10 << "msec." << std::endl;

  // Verify results.
  std::vector<int> output_h = plas::gpu::from_mem(output_d);

  bool equal = (ref_output_h[0] == output_h[0]);
  std::cout << (equal ? "correct!" : "error!") << std::endl;

  return 0;
}
