#include "../intersection_kernel.h"
#include <iostream>
#include <vector>
#include <algorithm>
#include <iterator>
#include <random>

template <typename type_t, typename indices_t>
int intersect_indices(std::vector<type_t>& a, std::vector<type_t>& b,
                  std::vector<indices_t>& c) {
  int a_idx = 0;
  int b_idx = 0;
  int c_idx = 0;
  while (a_idx < a.size() && b_idx < b.size()) {
    if (a[a_idx] < b[b_idx]) {
      ++a_idx;
    } else {
      if (b[b_idx] < a[a_idx]) {
        ++b_idx;
      } else {
      c[c_idx] = a_idx;
      ++c_idx;
      ++a_idx;
      ++b_idx;
    }
  }
}
  return c_idx;
}

int main(int argc, char** argv) {
  plas::gpu::standard_context_t context;

  // Prepare test data.
  int a_count = 128000000;
  int b_count = 128000;

  plas::gpu::mem_t<int> a_keys =
      plas::gpu::fill_random(1, 200000000, a_count, true, context);
  plas::gpu::mem_t<int> b_keys =
      plas::gpu::fill_random(1, 200000000, b_count, true, context);

  std::vector<int> a_keys_h = plas::gpu::from_mem(a_keys);
  std::vector<int> b_keys_h = plas::gpu::from_mem(b_keys);

  plas::gpu::mem_t<int> c_keys = plas::gpu::fill(0, a_count + b_count, context);
  std::vector<int> c_keys_h(a_count+b_count, 0);

  context.timer_begin();
  int cpu_total = intersect_indices(a_keys_h, b_keys_h, c_keys_h);
  double elapsed = context.timer_end();
  std::cout << "CPU time: " << elapsed * 1000 << "ms." << std::endl;

  context.timer_begin();
  int total = plas::gpu::intersect_indices<true>(a_keys.data(), a_count, b_keys.data(),
                                         b_count, c_keys.data(),
                                         plas::gpu::less_t<int>(), context);
  elapsed = context.timer_end();
  std::cout << "GPU time: " << elapsed * 1000 << "ms." << std::endl;

  std::vector<int> c_keys_verify = plas::gpu::from_mem(c_keys);

  std::cout << "cpu total: " << cpu_total << std::endl;
  std::cout << "gpu total: " << total << std::endl;

  bool equal =
      std::equal(c_keys_h.begin(), c_keys_h.begin()+cpu_total, c_keys_verify.begin(),
                 [](const int & l, const int & r)->bool { return l == r; });
  std::cout << (equal ? "correct!" : "error!") << std::endl;
}
