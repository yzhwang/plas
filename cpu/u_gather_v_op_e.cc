#include "cc/plas/cpu/api.h"

namespace plas {
namespace cpu {

void u_gather_v_mul_e_reduce_sum(CSR csr, float* const udata, const float* vdata,
                                 const float* edata, const std::vector<int32>& hshape) {
  Idx hsize = (Idx)Prod<int32>(hshape);
  zero_float(udata, csr.size_u* hsize);
// TODO(powergao) optimize with rsm kernel
#pragma omp parallel for
  for (int32 i = 0; i < csr.size_u; ++i) {
    for (Idx j = csr.indptr[i]; j < csr.indptr[i + 1]; ++j) {
      // rdata += edata * vdata
      ewise_fma(&edata[j * hsize], &vdata[csr.indices[j] * hsize], &udata[i * hsize], hsize);
    }
  }
}

void u_gather_v_mul_e_reduce_sum_backward(CSR csr, const Idx* eidx, const float* vdata,
                                          const float* edata, const float* grad, float* const vgrad,
                                          float* const egrad, const std::vector<int32>& hshape) {
  // IMPORTANCE: CSR should be transposed of the forward computation csr.
  int32 feat_len = Prod<int32>(hshape);
  int32  V;  // origianl CSR U and V.
  V = csr.size_u;
  zero_float(egrad, csr.size_e * feat_len);
  zero_float(vgrad, V * feat_len);

#pragma omp parallel for
  for (int32 i = 0; i < V; ++i) {
    for (Idx j = csr.indptr[i]; j < csr.indptr[i + 1]; ++j) {
      Idx ii_e = eidx[j];
      Idx ii_u = csr.indices[j];
      Idx ii_v = i;
      ewise_fma(&grad[ii_u * feat_len], &vdata[ii_v * feat_len], &egrad[ii_e * feat_len], feat_len);
      ewise_fma(&grad[ii_u * feat_len], &edata[ii_e * feat_len], &vgrad[ii_v * feat_len], feat_len);
    }
  }
}

void u_gather_v_op_e(CSR csr, float* const udata, const float* vdata, const float* edata,
                     const std::vector<int32>& hshape, BinaryOpType op, ReduceType reduce) {
  if (op == kBinaryOpMul) {
    switch (reduce) {
      case kReduceSum: {
        u_gather_v_mul_e_reduce_sum(csr, udata, vdata, edata, hshape);
        break;
      }
      default:
        LOG(FATAL) << "Not supported reduce type " << reduce;
    }
  } else {
    LOG(FATAL) << " Not supported binary type " << op;
  }
}

void u_gather_v_op_e_backward(CSR csr, const Idx* eidx, const float* vdata, const float* edata,
                              const float* grad, float* const vgrad, float* const egrad,
                              const std::vector<int32>& hshape, BinaryOpType op,
                              ReduceType reduce) {
  if (op == kBinaryOpMul) {
    switch (reduce) {
      case kReduceSum: {
        u_gather_v_mul_e_reduce_sum_backward(csr, eidx, vdata, edata, grad, vgrad, egrad, hshape);
        break;
      }
      default:
        LOG(FATAL) << "Not supported reduce type " << reduce;
    }
  } else {
    LOG(FATAL) << " Not supported binary type " << op;
  }
}
}  // namespace cpu
}  // namespace plas
