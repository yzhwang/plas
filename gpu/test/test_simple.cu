#include "../kernel_launch.h"
#include "../memory.h"
#include <iostream>
#include <vector>
#include <algorithm>

int main(int argc, char** argv) {
  plas::gpu::standard_context_t context;

  // Prepare test data.
  int count = 12800;
  plas::gpu::mem_t<int> degrees_d =
      plas::gpu::fill_random(10, 12, count,
                             /*sorted=*/false, context);
  plas::gpu::mem_t<int> output_d = plas::gpu::fill(0, count, context);
  std::vector<int> degrees_h = plas::gpu::from_mem(degrees_d);
  std::vector<int> offsets_h(count + 1, 0);
  for (int i = 1; i <= count; ++i) {
    offsets_h[i] = offsets_h[i - 1] + degrees_h[i - 1];
  }
  plas::gpu::mem_t<int> offsets_d = plas::gpu::to_mem(offsets_h, context);

  // Launch the kernel
  int* offsets = offsets_d.data();
  int* output = output_d.data();
  auto get_degree = [=] PLAS_DEVICE(int index) {
    output[index] = offsets[index + 1] - offsets[index];
  };

  context.timer_begin();
  for (int i = 0; i < 10; ++i) {
    plas::gpu::transform(get_degree, count, context);
  }
  double elapsed = context.timer_end();
  std::cout << "GPU time: " << elapsed / 10 << "msec." << std::endl;

  // Verify results.
  std::vector<int> output_h = plas::gpu::from_mem(output_d);

  bool equal =
      std::equal(degrees_h.begin(), degrees_h.end(), output_h.begin(),
                 [](const int & l, const int & r)->bool { return l == r; });
  std::cout << (equal ? "correct!" : "error!") << std::endl;

  return 0;
}
