#include "cc/plas/cpu/api.h"

namespace plas {
namespace cpu {

void u_scatter_e(CSR csr, const float* udata, float* const edata,
                 const std::vector<int32>& hshape) {
  int32 stride = Prod<int32>(hshape);
#pragma omp parallel for
  for (int32 i = 0; i < csr.size_u; ++i) {
    for (Idx j = csr.indptr[i]; j < csr.indptr[i + 1]; ++j) {
      Idx ii_u = i;
      Idx ii_e = j;
      // edata = udata
      ewise_copy(&(edata[ii_e * stride]), &(udata[ii_u * stride]), stride);
    }
  }
}

}  // namespace cpu
}  // namespace plas
