#include "../edge_softmax_kernel.h"
#include <iostream>
#include <vector>
#include <algorithm>
#include <random>

/**
 * @brief CPU reference edge_softmax
 * @param input:       feature vectors to compute edge softmax;
 * @param seg_offsets: offset of each segment, seg_offsets[seg_count] stores
 *                     edge count;
 * @param count:       number of segments;
 * @param hidden_dim:  length of hidden feature vec;
 * @param output:      feature vectors to store edge softmax results;
 */
template <typename type_t>
void edge_softmax(std::vector<type_t>& input_h, std::vector<int>& offsets_h,
                  int count, int hidden_dim, std::vector<type_t>& output_h) {
  for (int i = 0; i < count; ++i) {
    std::vector<float> hidden(hidden_dim);
    for (int k = 0; k < hidden_dim; ++k) {
      hidden[k] = std::numeric_limits<float>::lowest();
    }
    // reduce max over edges
    for (int j = offsets_h[i]; j < offsets_h[i + 1]; ++j) {
      for (size_t k = 0; k < hidden_dim; ++k) {
        hidden[k] = std::max(input_h[j * hidden_dim + k], hidden[k]);
      }
    }
    // minus max and exp over edges
    for (int j = offsets_h[i]; j < offsets_h[i + 1]; ++j) {
      for (size_t k = 0; k < hidden_dim; ++k) {
        output_h[j * hidden_dim + k] =
            std::exp(input_h[j * hidden_dim + k] - hidden[k]);
      }
    }
    // reduce sum over edges
    for (size_t k = 0; k < hidden_dim; ++k) {
      hidden[k] = 0.f;
    }
    for (int j = offsets_h[i]; j < offsets_h[i + 1]; ++j) {
      for (size_t k = 0; k < hidden_dim; ++k) {
        hidden[k] += output_h[j * hidden_dim + k];
      }
    }
    // divide sum over edges
    for (int j = offsets_h[i]; j < offsets_h[i + 1]; ++j) {
      for (size_t k = 0; k < hidden_dim; ++k) {
        output_h[j * hidden_dim + k] = output_h[j * hidden_dim + k] / hidden[k];
      }
    }
  }
}

/**
 * @brief CPU reference edge_softmax backward gradient
 * @param input_h:       computed feature vectors of edge softmax;
 * @param input_grad_h:  computed gradients of edge softmax vectors;
 * @param offsets_h:     offset of each segment, offsets_h[count] stores
 *                       edge count;
 * @param count:         number of segments;
 * @param hidden_dim:    length of hidden feature vec;
 * @param output_h:      feature vectors to store edge softmax grad results;
 */
template <typename type_t>
void edge_softmax_backward(std::vector<type_t>& input_h,
                           std::vector<type_t>& input_grad_h,
                           std::vector<int>& offsets_h, int count,
                           int hidden_dim, std::vector<type_t>& output_h) {
  for (int i = 0; i < count; ++i) {
    std::vector<type_t> accum(hidden_dim, 0.f);

    // compute accum = -(sigma(grady * datay))
    for (int j = offsets_h[i]; j < offsets_h[i + 1]; ++j) {
      for (int k = 0; k < hidden_dim; ++k) {
        accum[k] -=
            input_grad_h[j * hidden_dim + k] * input_h[j * hidden_dim + k];
      }
    }

    // compute gradx = accum * y + grady * y
    for (int j = offsets_h[i]; j < offsets_h[i + 1]; ++j) {
      for (int k = 0; k < hidden_dim; ++k) {
        output_h[j * hidden_dim + k] =
            accum[k] * input_h[j * hidden_dim + k] +
            input_grad_h[j * hidden_dim + k] * input_h[j * hidden_dim + k];
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

  plas::gpu::mem_t<float> output_d =
      plas::gpu::fill(0.0f, offsets_h[count] * hidden_dim, context);
  std::vector<float> output_h = plas::gpu::from_mem(output_d);

  plas::gpu::mem_t<float> output_grad_d = plas::gpu::fill_random(
      1.0f, 10.0f, offsets_h[count] * hidden_dim, false, context);
  std::vector<float> output_grad_h = plas::gpu::from_mem(output_grad_d);

  plas::gpu::mem_t<float> input_d = plas::gpu::fill_random(
      1.0f, 20.0f, offsets_h[count] * hidden_dim, false, context);
  std::vector<float> input_h = plas::gpu::from_mem(input_d);

  // Launch the kernel
  context.timer_begin();
  const float* input_data = input_d.data();
  for (int i = 0; i < 1; ++i) {
    plas::gpu::edge_softmax_small(input_data, offsets_d.data(), count,
                                  hidden_dim, output_d.data(), context);
  }
  double elapsed = context.timer_end();
  std::vector<float> output_verify = plas::gpu::from_mem(output_d);
  std::cout << "GPU time: " << elapsed << "s." << std::endl;

  // CPU version
  for (int i = 0; i < 1; ++i) {
    edge_softmax(input_h, offsets_h, count, hidden_dim, output_h);
  }

  // Verify results.
  bool equal =
      std::equal(output_h.begin(), output_h.end(), output_verify.begin(),
                 [](const float & l, const float & r)
                     ->bool { return (abs(l - r) / l < 0.01); });
  std::cout << (equal ? "correct!" : "error!") << std::endl;

  // Launch the bkw kernel
  context.timer_begin();
  const float* output_data = output_d.data();
  const float* output_grad_data = output_grad_d.data();
  for (int i = 0; i < 1; ++i) {
    plas::gpu::edge_softmax_bkw_small(output_data, output_grad_data,
                                      offsets_d.data(), count, hidden_dim,
                                      input_d.data(), context);
  }
  elapsed = context.timer_end();
  std::vector<float> input_verify = plas::gpu::from_mem(input_d);
  std::cout << "GPU time: " << elapsed << "s." << std::endl;

  // CPU version
  for (int i = 0; i < 1; ++i) {
    edge_softmax_backward(output_h, output_grad_h, offsets_h, count, hidden_dim,
                          input_h);
  }

  // Verify results.
  bool equal_bkw =
      std::equal(input_h.begin(), input_h.end(), input_verify.begin(),
                 [](const float & l, const float & r)->bool {
        return (abs(l - r) < 0.00001 || abs(l - r) / l < 0.01);
      });
  std::cout << (equal_bkw ? "correct!" : "error!") << std::endl;

  return 0;
}
