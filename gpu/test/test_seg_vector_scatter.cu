#include "../scatter_kernel.h"
#include <iostream>
#include <vector>
#include <algorithm>
#include <random>

/**
 * @brief CPU reference seg_vector_reduce
 * @param udata:       feature vectors to be scattered;
 * @param seg_offsets: offset of each segment, seg_offsets[seg_count] stores
 *                     edge count;
 * @param count:       number of segments;
 * @param hidden_dim:  length of hidden feature vec;
 * @param vdata:       feature vectors to store scattered results;
 * @param hidden_dim:  hidden dim of feature vectors.
 */
template <typename type_t>
void seg_vector_scatter(std::vector<type_t>& udata_h,
                        std::vector<int>& offsets_h, int count, int hidden_dim,
                        std::vector<type_t>& vdata_h) {
  std::vector<type_t> val(hidden_dim);
  for (int i = 0; i < count; ++i) {
    int start = offsets_h[i];
    int end = offsets_h[i + 1];
    int u_idx = i * hidden_dim;
    std::copy(udata_h.begin() + u_idx, udata_h.begin() + u_idx + hidden_dim,
              val.begin());
    for (int j = start; j < end; ++j) {
      for (int idx = 0; idx < hidden_dim; ++idx) {
        vdata_h[j * hidden_dim + idx] = val[idx];
      }
    }
  }
}

int main(int argc, char** argv) {
  plas::gpu::standard_context_t context;

  // Prepare test data.
  int count = 12800;
  int hidden_dim = 128;
  plas::gpu::mem_t<int> degrees_d =
      plas::gpu::fill_random(10, 12, count,
                             /*sorted=*/false, context);
  std::vector<int> degrees_h = plas::gpu::from_mem(degrees_d);
  std::vector<int> offsets_h(count + 1, 0);
  for (int i = 1; i <= count; ++i) {
    offsets_h[i] = offsets_h[i - 1] + degrees_h[i - 1];
  }
  plas::gpu::mem_t<int> offsets_d = plas::gpu::to_mem(offsets_h, context);

  plas::gpu::mem_t<float> udata_d =
      plas::gpu::fill_random(0.0f, 10.0f, count * hidden_dim,
                             /*sorted=*/false, context);
  std::vector<float> udata_h = plas::gpu::from_mem(udata_d);

  plas::gpu::mem_t<float> vdata_d =
      plas::gpu::fill(0.0f, offsets_h[count] * hidden_dim, context);
  std::vector<float> vdata_h = plas::gpu::from_mem(vdata_d);

  // Launch the kernel
  context.timer_begin();
  const float* udata_data = udata_d.data();
  for (int i = 0; i < 10; ++i) {
    plas::gpu::seg_vector_scatter_small(udata_data, offsets_d.data(), count,
                                        hidden_dim, vdata_d.data(), context);
  }
  double elapsed = context.timer_end();
  std::vector<float> vdata_verify = plas::gpu::from_mem(vdata_d);
  std::cout << "GPU time: " << elapsed / 10 << "s." << std::endl;

  // CPU version
  for (int i = 0; i < 10; ++i) {
    seg_vector_scatter(udata_h, offsets_h, count, hidden_dim, vdata_h);
  }

  // Verify results.
  bool equal = std::equal(vdata_h.begin(), vdata_h.end(), vdata_verify.begin(),
                          [](const float & l, const float & r)
                              ->bool { return (abs(l - r) / l < 0.01); });
  std::cout << (equal ? "correct!" : "error!") << std::endl;

  return 0;
}
