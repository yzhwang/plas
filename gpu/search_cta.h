#ifndef PLAS_GPU_SEARCH_H_
#define PLAS_GPU_SEARCH_H_

#include "loadstore.h"
#include "merge_cta.h"
#include "memory.h"
#include "context.h"

namespace plas {
namespace gpu {

template<typename keys_it, typename int_t, typename key_t>
PLAS_HOST_DEVICE int_t binary_search(keys_it keys, int_t count, key_t key) {
  int_t begin = 0;
  int_t end = count;
  while (begin < end) {
    int_t mid = (begin + end) / 2;
    key_t key2 = keys[mid];
    if (key2 == key) return mid;
    if (key2 > key)
      end = mid;
    else
      begin = mid + 1;
  }
  return -1;
}

template <bounds_t bounds, typename a_keys_it, typename b_keys_it,
          typename comp_t>
mem_t<int> merge_path_partitions(a_keys_it a, int64_t a_count, b_keys_it b,
                                 int64_t b_count, int64_t spacing, comp_t comp,
                                 context_t& context) {
  typedef int int_t;
  int num_partitions = (int)div_up(a_count + b_count, spacing) + 1;
  mem_t<int_t> mem(num_partitions, context);
  int_t* p = mem.data();
  transform([=] PLAS_DEVICE(int index) {
              int_t diag = (int_t)min(spacing * index, a_count + b_count);
              p[index] = merge_path<bounds>(a, (int_t)a_count, b,
                                            (int_t)b_count, diag, comp);
            },
            num_partitions, context);
  return mem;
}

template <bool duplicates, typename a_keys_it, typename b_keys_it,
          typename comp_t>
mem_t<int> balanced_path_partitions(a_keys_it a, int64_t a_count, b_keys_it b,
                                    int64_t b_count, int64_t spacing,
                                    comp_t comp, context_t& context) {
  typedef int int_t;
  int num_partitions = (int)div_up(a_count + b_count, spacing);
  mem_t<int_t> mem(num_partitions + 1, context);
  int_t* mem_data = mem.data();
  transform([=] PLAS_DEVICE(int index) {
              int_t diag = (int_t)min(spacing * index, a_count + b_count);
              int2 p = balanced_path<duplicates, int64_t>(
                  a, (int_t)a_count, b, (int_t)b_count, diag, 4, comp);
              if (p.y) {
                p.x |= 0x80000000;
              }
              mem_data[index] = p.x;
            },
            num_partitions + 1, context);
  return mem;
}

}  // namespace gpu
}  // namespace plas

#endif
