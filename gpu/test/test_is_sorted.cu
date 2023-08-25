#include "../any_all_kernel.h"
#include "../memory.h"
#include <iostream>
#include <vector>
#include <algorithm>

int main(int argc, char** argv) {
  plas::gpu::standard_context_t context;

  // Prepare test data.
  int count = 12800;
  plas::gpu::mem_t<int> sorted_vals_d =
      plas::gpu::fill_random(0, 20000, count,
                             /*sorted=*/true, context);

  plas::gpu::mem_t<int> unsorted_vals_d =
      plas::gpu::fill_random(0, 20000, count,
                             /*sorted=*/false, context);

  // Test on sorted array.
  bool is_sorted = plas::gpu::is_sorted(count, sorted_vals_d.data(), context);
  bool is_unsorted = plas::gpu::is_unsorted(count, sorted_vals_d.data(), context);

  std::cout << "test is_sorted: sorted array is " << (is_sorted ? "sorted!" : "unsorted!") <<
  std::endl;
  std::cout << "test is_unsorted: sorted array is " << (is_unsorted ? "unsorted!" : "sorted!") <<
  std::endl;

  // Test on unsorted array.
  is_sorted = plas::gpu::is_sorted(count, unsorted_vals_d.data(), context);
  is_unsorted = plas::gpu::is_unsorted(count, unsorted_vals_d.data(), context);


  std::cout << "test is_sorted: unsorted array is " << (is_sorted ? "sorted!" : "unsorted!") <<
  std::endl;
  std::cout << "test is_unsorted: unsorted array is " << (is_unsorted ? "unsorted!" : "sorted!") <<
  std::endl;

  return 0;
}
