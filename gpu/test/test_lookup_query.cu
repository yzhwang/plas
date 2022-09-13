#include "../lookup_query_kernel.h"
#include <iostream>
#include <vector>
#include <algorithm>
#include <iterator>
#include <random>
#include <unordered_map>
#include <cstdlib>

template <typename type_t, typename indices_t>
void lookup_query(std::vector<type_t>& a, std::vector<type_t>& b,
                  std::vector<indices_t>& c) {
  std::unordered_map<type_t, indices_t> data_map;
  for (int i = 0; i < a.size(); ++i) {
    data_map.insert({a[i], i});
  }
  for (int i = 0; i < b.size(); ++i) {
    auto ind = data_map.find(b[i]);
    if (ind != data_map.end()) {
      c[i] = ind->second;
    } else {
      c[i] = -1;
    }
  }
}

int main(int argc, char** argv) {
  plas::gpu::standard_context_t context;

  // Prepare test data.
  int a_count = 1280000;
  int b_count = 12800;

  std::vector<int> a_keys_h(a_count, 0);
  std::vector<int> b_keys_h(b_count, 0);
  unsigned int seed = 42;
  for (int i = 1; i < a_count; ++i) {
    a_keys_h[i] = a_keys_h[i-1] + (rand_r(&seed)%3+1);
  }
  for (int i = 1; i < b_count; ++i) {
    b_keys_h[i] = a_keys_h[i*2];
  }

  std::random_shuffle(a_keys_h.begin(), a_keys_h.end());
  std::random_shuffle(b_keys_h.begin(), b_keys_h.end());

  plas::gpu::mem_t<int> a_keys = plas::gpu::to_mem(a_keys_h, context);
  plas::gpu::mem_t<int> b_keys = plas::gpu::to_mem(b_keys_h, context);

  plas::gpu::mem_t<int> c_keys = plas::gpu::fill(-1, b_count, context);
  std::vector<int> c_keys_h(b_count, -1);

  context.timer_begin();
  lookup_query(a_keys_h, b_keys_h, c_keys_h);
  double elapsed = context.timer_end();
  std::cout << "CPU time: " << elapsed * 1000 << "ms." << std::endl;

  context.timer_begin();
  plas::gpu::lookup_query(
    b_keys.data(), b_count, a_keys.data(), a_count, c_keys.data(), context);

  elapsed = context.timer_end();
  std::cout << "GPU time: " << elapsed * 1000 << "ms." << std::endl;

  std::vector<int> c_keys_verify = plas::gpu::from_mem(c_keys);

  bool equal =
      std::equal(c_keys_h.begin(), c_keys_h.end(), c_keys_verify.begin(),
                 [](const int & l, const int & r)->bool { return l == r; });
  std::cout << (equal ? "correct!" : "error!") << std::endl;
}
