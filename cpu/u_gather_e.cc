#include "cc/plas/cpu/api.h"

namespace plas {
namespace cpu {

void u_gather_e_reduce_sum(CSR csr, float* const udata, const float* edata,
                           const std::vector<int32>& hshape) {
  int stride = Prod<int32>(hshape);
  zero_float(udata, csr.size_u * stride);
#pragma omp parallel for
  for (int32 i = 0; i < csr.size_u; ++i) {
    for (Idx j = csr.indptr[i]; j < csr.indptr[i + 1]; ++j) {
      Idx ii_u = i;
      Idx ii_e = j;
      // udata += edata
      ewise_add(&(edata[ii_e * stride]), &(udata[ii_u * stride]), stride);
    }
  }
}

void u_gather_e(CSR csr, float* const udata, const float* edata, const std::vector<int32>& hshape,
                ReduceType reduce) {
  switch (reduce) {
    case kReduceSum: {
      u_gather_e_reduce_sum(csr, udata, edata, hshape);
      break;
    };
    default:
      LOG(FATAL) << "Not supported reduce type " << reduce;
  }
}
}  // namespace cpu
}  // namespace plas
