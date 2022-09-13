#include "../reduce_kernel.h"
#include <iostream>
#include <vector>
#include <algorithm>
#include <random>

/**
 * @brief CPU reference seg_vector_reduce
 * @param vdata:       feature vectors to be gathered and reduced;
 * @param seg_offsets: offset of each segment, seg_offsets[seg_count] stores
 *                     edge count;
 * @param count:       number of segments;
 * @param hidden_dim:  length of hidden feature vec;
 * @param indices:     vertex indices of each neighbor node;
 * @param udata:       feature vectors to store reduction results;
 */
template <typename type_t>
void seg_vector_reduce(std::vector<type_t>& vdata_h,
                       std::vector<int>& offsets_h, int count, int hidden_dim,
                       std::vector<int>& indices_h,
                       std::vector<type_t>& udata_h) {
  for (int i = 0; i < count; ++i) {
    int start = offsets_h[i];
    int end = offsets_h[i + 1];
    int u_idx = i * hidden_dim;
    for (int j = start; j < end; ++j) {
      int v_idx = indices_h[j] * hidden_dim;
      for (int idx = 0; idx < hidden_dim; ++idx) {
        udata_h[u_idx + idx] = udata_h[u_idx + idx] + vdata_h[v_idx + idx];
      }
    }
  }
}

/**
 * @brief CPU reference seg_vector_reduce
 * @param edata:       feature vectors to be gathered and reduced;
 * @param seg_offsets: offset of each segment, seg_offsets[seg_count] stores
 *                     edge count;
 * @param count:       number of segments;
 * @param hidden_dim:  length of hidden feature vec;
 * @param udata:       feature vectors to store reduction results;
 */
template <typename type_t>
void seg_vector_reduce_edge(std::vector<type_t>& edata_h,
                            std::vector<int>& offsets_h, int count,
                            int hidden_dim, std::vector<type_t>& udata_h) {
  for (int i = 0; i < count; ++i) {
    int start = offsets_h[i];
    int end = offsets_h[i + 1];
    int u_idx = i * hidden_dim;
    for (int j = start; j < end; ++j) {
      int e_idx = j * hidden_dim;
      for (int idx = 0; idx < hidden_dim; ++idx) {
        udata_h[u_idx + idx] = udata_h[u_idx + idx] + edata_h[e_idx + idx];
      }
    }
  }
}

int main(int argc, char** argv) {
  plas::gpu::standard_context_t context;

  // Prepare test data.
  int count = 2048;
  int v_count = 30000;
  int hidden_dim = 128;
  plas::gpu::mem_t<int> degrees_d =
      plas::gpu::fill_random(20, 50, count,
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
      plas::gpu::fill_random(0.0f, 10.0f, v_count * hidden_dim,
                             /*sorted=*/false, context);
  std::vector<float> vdata_h = plas::gpu::from_mem(vdata_d);

  plas::gpu::mem_t<float> edata_d =
      plas::gpu::fill_random(0.0f, 10.0f, offsets_h[count] * hidden_dim,
                             /*sorted=*/false, context);
  std::vector<float> edata_h = plas::gpu::from_mem(edata_d);

  std::vector<int> uid_h(count, 0);
  std::vector<int> vid_h(v_count, 0);
  std::vector<int> indices_h(offsets_h[count]);
  for (int i = 0; i < v_count; ++i) {
    vid_h[i] = i;
  }

  for (int i = 0; i < count; ++i) {
    std::random_shuffle(vid_h.begin(), vid_h.end());
    int offset = offsets_h[i];
    int end = offsets_h[i + 1];
    for (int j = 0; j < degrees_h[i]; ++j) {
      indices_h[j + offset] = vid_h[j];
    }
    std::sort(indices_h.begin() + offset, indices_h.begin() + end);
  }
  plas::gpu::mem_t<int> indices_d = plas::gpu::to_mem(indices_h, context);

  // Launch the kernel
  context.timer_begin();
  const float* vdata_data = vdata_d.data();
  for (int i = 0; i < 10; ++i) {
    plas::gpu::seg_vector_reduce_small(
        vdata_data, offsets_d.data(), count, indices_d.data(), hidden_dim,
        udata_d.data(), plas::gpu::plus_t<float>(), context);
  }
  double elapsed = context.timer_end();
  std::cout << "GPU seg_vec_reduce time: " << elapsed / 10 << "s." << std::endl;

  context.timer_begin();
  const float* edata_data = edata_d.data();
  for (int i = 0; i < 10; ++i) {
    plas::gpu::seg_vector_reduce_small(
        edata_data, offsets_d.data(), count, (int*)nullptr, hidden_dim,
        udata_d.data(), plas::gpu::plus_t<float>(), context);
  }
  elapsed = context.timer_end();
  std::vector<float> udata_verify = plas::gpu::from_mem(udata_d);
  std::cout << "GPU seg_vec_reduce edge time: " << elapsed / 10 << "s."
            << std::endl;

  // CPU version
  for (int i = 0; i < 10; ++i) {
    seg_vector_reduce(vdata_h, offsets_h, count, hidden_dim, indices_h,
                      udata_h);
  }
  for (int i = 0; i < 10; ++i) {
    seg_vector_reduce_edge(edata_h, offsets_h, count, hidden_dim, udata_h);
  }

  // Verify results.
  bool equal = std::equal(udata_h.begin(), udata_h.end(), udata_verify.begin(),
                          [](const float & l, const float & r)
                              ->bool { return (abs(l - r) / l < 0.01); });
  std::cout << (equal ? "correct!" : "error!") << std::endl;

  return 0;
}
