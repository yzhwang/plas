#include "cc/plas/cpu/api.h"

namespace plas {

namespace cpu {

void edge_softmax(CSR csr, const float* x, float* const y, const std::vector<int32>& shape) {
  // TODO(powergao) optimize
  Idx U = csr.size_u;
  Idx V = csr.size_v;
  Idx stride = 1;
  for (size_t i = 1; i < shape.size(); ++i) {
    stride *= (Idx)shape[i];
  }

#pragma omp parallel for schedule(dynamic)
  for (Idx i = 0; i < U; ++i) {
    std::vector<float> hidden(stride);
    for (Idx k = 0; k < stride; ++k) {
      hidden[k] = std::numeric_limits<float>::lowest();
    }
    // reduce max over edges
    for (Idx j = csr.indptr[i]; j < csr.indptr[i + 1]; ++j) {
      for (size_t k = 0; k < stride; ++k) {
        hidden[k] = std::max(x[j * stride + k], hidden[k]);
      }
    }
    // minus max and exp over edges
    for (Idx j = csr.indptr[i]; j < csr.indptr[i + 1]; ++j) {
      for (size_t k = 0; k < stride; ++k) {
        y[j * stride + k] = std::exp(x[j * stride + k] - hidden[k]);
      }
    }
    // reduce sum over edges
    for (size_t k = 0; k < stride; ++k) {
      hidden[k] = 0.f;
    }
    for (Idx j = csr.indptr[i]; j < csr.indptr[i + 1]; ++j) {
      for (size_t k = 0; k < stride; ++k) {
        hidden[k] += y[j * stride + k];
      }
    }
    // divide sum over edges
    for (Idx j = csr.indptr[i]; j < csr.indptr[i + 1]; ++j) {
      for (size_t k = 0; k < stride; ++k) {
        y[j * stride + k] = y[j * stride + k] / hidden[k];
      }
    }
  }
}

void edge_softmax_backward(CSR csr, const float* y, const float* grady, float* const gradx,
                           const std::vector<int32>& shape) {
  Idx stride = 1;
  for (size_t i = 1; i < shape.size(); ++i) {
    stride *= shape[i];
  }

// stride usually means number of heads, usually less than 10
// TODO(powergao) optimize
#pragma omp parallel for schedule(dynamic)
  for (int32 i = 0; i < csr.size_u; ++i) {
    std::vector<float> accum(stride, 0.f);

    // compute accum = -(sigma(grady * datay))
    for (Idx j = csr.indptr[i]; j < csr.indptr[i + 1]; ++j) {
      for (Idx k = 0; k < stride; ++k) {
        accum[k] -= grady[j * stride + k] * y[j * stride + k];
      }
    }

    // compute gradx = accum * y + grady * y
    for (Idx j = csr.indptr[i]; j < csr.indptr[i + 1]; ++j) {
      for (Idx k = 0; k < stride; ++k) {
        gradx[j * stride + k] =
            accum[k] * y[j * stride + k] + grady[j * stride + k] * y[j * stride + k];
      }
    }
  }
}

}  // namespace cpu
}  // namespace plas
