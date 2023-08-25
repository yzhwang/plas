#include "../scan_kernel.h"
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
  std::vector<int> offsets_h(count, 0);
  for (int i = 1; i < count; ++i) {
    offsets_h[i] = offsets_h[i - 1] + degrees_h[i - 1];
  }

  // Launch the kernel
  int* output = output_d.data();
  int* input = degrees_d.data();

  context.timer_begin();
  for (int i = 0; i < 10; ++i) {
    plas::gpu::scan(input, count, output, context);
  }
  double elapsed = context.timer_end();
  std::cout << "GPU time: " << elapsed / 10 << "msec." << std::endl;

  // Verify results.
  std::vector<int> output_h = plas::gpu::from_mem(output_d);

  bool equal =
      std::equal(offsets_h.begin(), offsets_h.end(), output_h.begin(),
                 [](const int & l, const int & r)->bool { return l == r; });
  std::cout << (equal ? "correct!" : "error!") << std::endl;

  return 0;
}
