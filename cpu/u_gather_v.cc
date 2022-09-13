#include "cc/plas/cpu/api.h"

#include <mkl.h>
#include <omp.h>

namespace plas {
namespace cpu {

void u_gather_v_reduce_sum(CSR csr, float* const udata, const float* vdata,
                           const std::vector<int32>& hshape) {
  int32 U = csr.size_u;
  int32 V = csr.size_v;

  int32 stride = 1;
  for (size_t i = 0; i < hshape.size(); ++i) {
    stride *= hshape[i];
  }

#pragma omp parallel for
  for (int32 i = 0; i < U; ++i) {
    std::fill(udata + i * stride, udata + (i + 1) * stride, 0.f);
    for (Idx j = csr.indptr[i]; j < csr.indptr[i + 1]; ++j) {
      int32 u = i;
      int32 v = csr.indices[j];
      saxpy(1.f, &(vdata[v * stride]), &(udata[u * stride]), stride);
    }
  }
}

void u_gather_v_reduce_max(CSR csr, float* const udata, const float* vdata,
                           const std::vector<int32>& hshape) {
  LOG(FATAL) << "Not implemented";
}

void u_gather_v_reduce_min(CSR csr, float* const udata, const float* vdata,
                           const std::vector<int32>& hshape) {
  LOG(FATAL) << "Not implemented";
}

void u_gather_v(CSR csr, float* const udata, const float* vdata, const std::vector<int32>& hshape,
                ReduceType reduce) {
  if (reduce == kReduceSum) {
    u_gather_v_reduce_sum(csr, udata, vdata, hshape);
  } else if (reduce == kReduceMax) {
    u_gather_v_reduce_max(csr, udata, vdata, hshape);
  } else if (reduce == kReduceMin) {
    u_gather_v_reduce_min(csr, udata, vdata, hshape);
  } else {
    LOG(FATAL) << "Invalid reduce type " << reduce;
  }
}

}  // namespace cpu

}  // namespace plas
