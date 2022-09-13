#include "cc/plas/cpu/api.h"

namespace plas {
namespace cpu {

void u_gather_v_bcast_mul_e_reduce_sum(CSR csr, float* const udata, const float* vdata,
                                       const float* edata, EWiseOpBcastInfo& info) {
  // lhs - edata, rhs - vdata
  const int bcast_ndim = info.out_flat_shape.size();  // feature dim
  CHECK_LE(bcast_ndim, 2) << "bcast_ndim expected to LE 2, but got " << bcast_ndim;

  zero_float(udata, csr.size_u * info.out_flat_len);
  int32 dim0, dim1;  // dim0: broadcasted dim(saxpy), dim1: number of saxpy,

  dim0 = std::max(info.lhs_flat_shape.back(), info.rhs_flat_shape.back());
  dim1 = 1;
  if (bcast_ndim == 2) {
    dim1 = info.lhs_flat_shape[0];
  }

  // TODO(powergao) RSM kernel here.
  if (info.lhs_flat_shape.back() == 1) {
// lhs - edata, should be broadcasted
#pragma omp parallel for
    for (int32 i = 0; i < csr.size_u; ++i) {
      for (Idx j = csr.indptr[i]; j < csr.indptr[i + 1]; ++j) {
        Idx ii_u = i;
        Idx ii_v = csr.indices[j];
        Idx ii_e = j;
        for (int k = 0; k < dim1; ++k) {
          saxpy(edata[ii_e * info.lhs_flat_len + k], &vdata[ii_v * info.rhs_flat_len + k * dim0],
                &udata[ii_u * info.out_flat_len + k * dim0], dim0);
        }
      }
    }
  } else if (info.rhs_flat_shape.back() == 1) {
// rhs - vdata, should be broadcasted
#pragma omp parallel for
    for (int32 i = 0; i < csr.size_u; ++i) {
      for (Idx j = csr.indptr[i]; j < csr.indptr[i + 1]; ++j) {
        Idx ii_u = i;
        Idx ii_v = csr.indices[j];
        Idx ii_e = j;
        for (int k = 0; k < dim1; ++k) {
          saxpy(edata[ii_e * info.lhs_flat_len + k * dim0], &vdata[ii_v * info.rhs_flat_len + k],
                &udata[ii_u * info.out_flat_len + k * dim0], dim0);
        }
      }
    }
  }
}

void u_gather_v_bcast_mul_e_reduce_sum_backward(CSR csr, const Idx* eidx, const float* vdata,
                                                const float* edata, const float* grad,
                                                float* const vgrad, float* const egrad,
                                                EWiseOpBcastInfo& info) {
  // lhs and rhs are the same with original csr
  int32 V, E;
  V = csr.size_u;  // NOTICE, csr is transposed, then, shape[0] should be V in original csr
  E = csr.size_e;
  const int bcast_ndim = info.out_flat_shape.size();

  zero_float(egrad, E * Prod(info.lhs_in_shape));
  zero_float(vgrad, V * Prod(info.rhs_in_shape));

  int32 dim0, dim1;  // dim0: broadcasted dim(saxpy), dim1: number of saxpy,

  dim0 = std::max(info.lhs_flat_shape.back(), info.rhs_flat_shape.back());
  dim1 = 1;
  if (bcast_ndim == 2) {
    dim1 = info.lhs_flat_shape[0];
  }

  if (info.lhs_flat_shape.back() == 1) {
// lhs-edata are broadcasted
#pragma omp parallel for
    for (int32 i = 0; i < V; ++i) {
      for (Idx j = csr.indptr[i]; j < csr.indptr[i + 1]; ++j) {
        Idx ii_u = csr.indices[j];
        Idx ii_v = i;
        Idx ii_e = eidx[j];
        for (int k = 0; k < dim1; ++k) {
          // egrad = dot(grad, vdata)
          sdot(&grad[ii_u * info.out_flat_len + k * dim0],
               &vdata[ii_v * info.rhs_flat_len + k * dim0], dim0,
               &egrad[ii_e * info.lhs_flat_len + k]);
          // vgrad = broadcast_ewise_mul(grad, edata)
          saxpy(edata[ii_e * info.lhs_flat_len + k], &grad[ii_u * info.out_flat_len + k * dim0],
                &vgrad[ii_v * info.rhs_flat_len + k * dim0], dim0);
        }
      }
    }
  } else if (info.rhs_flat_shape.back() == 1) {
// rhs-vdata are broadcasted
#pragma omp parallel for
    for (int32 i = 0; i < V; ++i) {
      for (Idx j = csr.indptr[i]; j < csr.indptr[i + 1]; ++j) {
        Idx ii_u = csr.indices[j];
        Idx ii_v = i;
        Idx ii_e = eidx[j];
        for (int k = 0; k < dim1; ++k) {
          // egrad = broadcast_ewise_mul(vdata, grad)
          saxpy(vdata[ii_v * info.rhs_flat_len + k], &grad[ii_u * info.out_flat_len + k * dim0],
                &egrad[ii_e * info.lhs_flat_len + k * dim0], dim0);
          sdot(&grad[ii_u * info.out_flat_len + k * dim0],
               &edata[ii_v * info.lhs_flat_len + k * dim0], dim0,
               &vgrad[ii_e * info.rhs_flat_len + k]);
        }
      }
    }
  } else {
    LOG(FATAL) << "Should not happen!!!";
  }
}

void u_gather_v_bcastop_e(CSR csr, float* const udata, const float* vdata, const float* edata,
                          BinaryOpType op, EWiseOpBcastInfo& info, ReduceType reduce) {
  if (op == kBinaryOpMul) {
    switch (reduce) {
      case kReduceSum: {
        u_gather_v_bcast_mul_e_reduce_sum(csr, udata, vdata, edata, info);
        break;
      }
      default:
        LOG(FATAL) << "Not supported reduce type " << reduce;
    }
  } else {
    LOG(FATAL) << " Not supported binary type " << op;
  }
}

void u_gather_v_bcastop_e_backward(CSR csr, const Idx* eidx, const float* vdata, const float* edata,
                                  const float* grad, float* const vgrad, float* const egrad,
                                  BinaryOpType op, EWiseOpBcastInfo& info, ReduceType reduce) {
  if (op == kBinaryOpMul) {
    switch (reduce) {
      case kReduceSum: {
        u_gather_v_bcast_mul_e_reduce_sum_backward(csr, eidx, vdata, edata, grad, vgrad, egrad,
                                                   info);
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
