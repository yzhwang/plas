#include "../reduce_op_kernel.h"
#include <iostream>
#include <vector>
#include <algorithm>
#include <random>

/**
 * @brief CPU reference seg_vector_reduce_op_grad
 * @param vdata:       feature vectors to be gathered and reduced;
 * @param edata:       edge feature vectors to be gathered and reduced;
 * @param seg_offsets: offset of each segment, seg_offsets[seg_count] stores
 *                     edge count;
 * @param count:       number of segments;
 * @param hidden_dim:  length of hidden feature vec;
 * @param indices:     vertex indices of each neighbor node;
 * @param udata:       feature vectors to store reduction results;
 * @param hidden_dim:  hidden dim of feature vectors.
 */
template <typename type_t, bool bcast>
void seg_vector_reduce_bcast_grad(std::vector<type_t>& vdata_h,
                          std::vector<type_t>& edata_h,
                          std::vector<int>& offsets_h, int count,
                          int hidden_dim, std::vector<int>& indices_h,
                          std::vector<type_t>& grad_h,
                          std::vector<type_t>& egrad_h,
                          std::vector<type_t>& vgrad_h) {
  for (int i = 0; i < count; ++i) {
    int start = offsets_h[i];
    int end = offsets_h[i + 1];
    int v_idx = i * hidden_dim;
    for (int idx = 0; idx < hidden_dim; ++idx) {
        vgrad_h[v_idx + idx] = 0;
    }
    for (int j = start; j < end; ++j) {
      if (bcast) egrad_h[j] = 0;
      int u_idx = indices_h[j] * hidden_dim;
      for (int idx = 0; idx < hidden_dim; ++idx) {
        if (!bcast) {
          egrad_h[j*hidden_dim + idx] = 0;
        }
        if (bcast) {
          egrad_h[j] += grad_h[u_idx + idx] * vdata_h[v_idx + idx];
        } else {
          egrad_h[j*hidden_dim + idx] = egrad_h[j*hidden_dim + idx] +
                                        grad_h[u_idx + idx] *
                                        vdata_h[v_idx + idx];
        }
        if (bcast) {
          vgrad_h[v_idx + idx] = vgrad_h[v_idx + idx] +
                                 grad_h[u_idx + idx] * edata_h[j];
        } else {
          vgrad_h[v_idx + idx] = vgrad_h[v_idx + idx] + grad_h[u_idx + idx] *
                                 edata_h[j*hidden_dim+idx];
        }
      }
    }
  }
}

int main(int argc, char** argv) {
  plas::gpu::standard_context_t context;

  // Prepare test data.
  int count = 1280;
  int v_count = 12800;
  int hidden_dim = 63;
  plas::gpu::mem_t<int> degrees_d =
      plas::gpu::fill_random(45, 50, count,
                             /*sorted=*/false, context);
  std::vector<int> degrees_h = plas::gpu::from_mem(degrees_d);
  std::vector<int> offsets_h(count + 1, 0);
  for (int i = 1; i <= count; ++i) {
    offsets_h[i] = offsets_h[i - 1] + degrees_h[i - 1];
  }
  plas::gpu::mem_t<int> offsets_d = plas::gpu::to_mem(offsets_h, context);

  plas::gpu::mem_t<float> grad_d =
      plas::gpu::fill_random(1.0f, 10.0f, v_count * hidden_dim, /*sorted=*/false, context);
  std::vector<float> grad_h = plas::gpu::from_mem(grad_d);

  plas::gpu::mem_t<float> vdata_d =
      plas::gpu::fill_random(0.0f, 10.0f, count * hidden_dim,
                             /*sorted=*/false, context);
  std::vector<float> vdata_h = plas::gpu::from_mem(vdata_d);

  plas::gpu::mem_t<float> edata_scalar_d =
      plas::gpu::fill_random(0.0f, 10.0f, offsets_h[count]*hidden_dim, false, context);
  std::vector<float> edata_scalar_h = plas::gpu::from_mem(edata_scalar_d);

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

  plas::gpu::mem_t<float> vgrad_d =
      plas::gpu::fill(0.0f, count * hidden_dim, context);
  std::vector<float> vgrad_h = plas::gpu::from_mem(vgrad_d);

  plas::gpu::mem_t<float> egrad_d =
      plas::gpu::fill(0.0f, offsets_h[count]*hidden_dim, context);
  std::vector<float> egrad_h = plas::gpu::from_mem(egrad_d);

  std::vector<int> eidx_h(offsets_h[count], 0);
  for (int i = 0; i < offsets_h[count]; ++i) {
    eidx_h[i] = i;
  }
  plas::gpu::mem_t<int> eidx_d = plas::gpu::to_mem(eidx_h, context);

  // Launch the kernel
  context.timer_begin();
  const float* grad_data = grad_d.data();
  const float* vdata_data = vdata_d.data();
  const float* edata_scalar_data = edata_scalar_d.data();
  int* offsets_data = offsets_d.data();
  int* indices_data = indices_d.data();
  const int* eidx_data = eidx_d.data();
  context.timer_begin();
    plas::gpu::seg_vector_reduce_op_grad_e_small<
        const float*, float*, const int*, int*, plas::gpu::plus_t<float>,
        plas::gpu::multiplies_t<float>, plas::gpu::nobcast>(
        grad_d.data(), eidx_data, offsets_data, count, offsets_h[count], indices_data,
        hidden_dim, 1, vdata_data, egrad_d.data(), plas::gpu::plus_t<float>(),
        plas::gpu::multiplies_t<float>(), context);
    plas::gpu::seg_vector_reduce_op_grad_v_small<
        const float*, float*, const int*, int*, plas::gpu::plus_t<float>,
        plas::gpu::multiplies_t<float>, plas::gpu::nobcast>(
        grad_data, eidx_d.data(), offsets_data, count, offsets_h[count], indices_data,
        hidden_dim, 1, edata_scalar_data, vgrad_d.data(), plas::gpu::plus_t<float>(),
        plas::gpu::multiplies_t<float>(), context);
  float elapsed = context.timer_end();
  std::vector<float> vgrad_verify = plas::gpu::from_mem(vgrad_d);
  std::vector<float> egrad_verify = plas::gpu::from_mem(egrad_d);
  std::cout << "GPU time for u gather v bcast mul e backward: " << elapsed / 10 << "s."
            << std::endl;

  // CPU version
    seg_vector_reduce_bcast_grad<float, false>(vdata_h, edata_scalar_h, offsets_h, count,
                                      hidden_dim, indices_h, grad_h, egrad_h, vgrad_h);

  // Verify results.
  bool equal = std::equal(vgrad_h.begin(), vgrad_h.end(), vgrad_verify.begin(),
                          [](const float & l, const float & r)
                              ->bool { return (abs(l - r) / l < 0.01); });
  std::cout << (equal ? "vgrad correct!" : "vgrad error!") << std::endl;

  equal = std::equal(egrad_h.begin(), egrad_h.end(), egrad_verify.begin(),
                          [](const float & l, const float & r)
                              ->bool { return (abs(l - r) / l < 0.01); });
  std::cout << (equal ? "egrad correct!" : "egrad error!") << std::endl;

  return 0;
}
