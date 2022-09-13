
#include "cc/plas/cpu/api.h"
#if GOOGLE_CUDA
#include "cc/plas/gpu/api.h"
#endif
#include <algorithm>
#include <cstring>
#include "cc/plas/api.h"
#include "cc/plas/types.h"

namespace plas {

/***********************************************************************
***************** Graph Neural Network Aggregation Routines ************
************************************************************************/

// CPU routines

template <>
void u_gather_v<kCPU>(CSR csr, float* const udata, const float* vdata,
                      const std::vector<int32>& hshape, ReduceType reduce, Stream* const stream) {
  plas::cpu::u_gather_v(csr, udata, vdata, hshape, reduce);
}

template <>
void u_gather_e<kCPU>(CSR csr, float* const udata, const float* edata,
                      const std::vector<int32>& hshape, ReduceType reduce, Stream* const stream) {
  plas::cpu::u_gather_e(csr, udata, edata, hshape, reduce);
}

template <>
void u_scatter_e<kCPU>(CSR csr, const float* udata, float* const edata,
                       const std::vector<int32>& hshape, Stream* const stream) {
  plas::cpu::u_scatter_e(csr, udata, edata, hshape);
}

template <>
void u_gather_v_op_e<kCPU>(CSR csr, float* const udata, const float* vdata, const float* edata,
                           const std::vector<int32>& hshape, BinaryOpType op, ReduceType reduce,
                           Stream* const stream) {
  plas::cpu::u_gather_v_op_e(csr, udata, vdata, edata, hshape, op, reduce);
}

template <>
void u_gather_v_op_e_backward<kCPU>(CSR csr, const Idx* eidx, const float* vdata,
                                    const float* edata, const float* grad, float* const vgrad,
                                    float* const egrad, const std::vector<int32>& hshape,
                                    BinaryOpType op, ReduceType reduce, Stream* const stream) {
  plas::cpu::u_gather_v_op_e_backward(csr, eidx, vdata, edata, grad, vgrad, egrad, hshape, op,
                                      reduce);
}

template <>
void u_gather_v_bcastop_e<kCPU>(CSR csr, float* const udata, const float* vdata, const float* edata,
                                BinaryOpType op, EWiseOpBcastInfo& info, ReduceType reduce,
                                Stream* const stream) {
  plas::cpu::u_gather_v_bcastop_e(csr, udata, vdata, edata, op, info, reduce);
}

template <>
void u_gather_v_bcastop_e_backward<kCPU>(CSR csr, const Idx* eidx, const float* vdata,
                                         const float* edata, const float* grad, float* const vgrad,
                                         float* const egrad, BinaryOpType op,
                                         EWiseOpBcastInfo& info, ReduceType reduce,
                                         Stream* const stream) {
  plas::cpu::u_gather_v_bcastop_e_backward(csr, eidx, vdata, edata, grad, vgrad, egrad, op, info,
                                           reduce);
}

template <>
void edge_softmax<kCPU>(CSR csr, const float* x, float* const y, const std::vector<int32>& shape,
                        Stream* const stream) {
  plas::cpu::edge_softmax(csr, x, y, shape);
}

template <>
void edge_softmax_backward<kCPU>(CSR csr, const float* y, const float* grady, float* const gradx,
                                 const std::vector<int32>& shape, Stream* const stream) {
  plas::cpu::edge_softmax_backward(csr, y, grady, gradx, shape);
}

#if GOOGLE_CUDA

// GPU routines

template <>
void u_gather_v<kGPU>(CSR csr, float* const udata, const float* vdata,
                      const std::vector<int32>& hshape, ReduceType reduce, Stream* const stream) {
  plas::gpu::u_gather_v(csr, udata, vdata, hshape, reduce, stream);
}

template <>
void u_gather_e<kGPU>(CSR csr, float* const udata, const float* edata,
                      const std::vector<int32>& hshape, ReduceType reduce, Stream* const stream) {
  plas::gpu::u_gather_e(csr, udata, edata, hshape, reduce, stream);
}

template <>
void u_scatter_e<kGPU>(CSR csr, const float* udata, float* const edata,
                       const std::vector<int32>& hshape, Stream* const stream) {
  plas::gpu::u_scatter_e(csr, udata, edata, hshape, stream);
}

template <>
void u_gather_v_op_e<kGPU>(CSR csr, float* const udata, const float* vdata, const float* edata,
                           const std::vector<int32>& hshape, BinaryOpType op, ReduceType reduce,
                           Stream* const stream) {
  plas::gpu::u_gather_v_op_e(csr, udata, vdata, edata, hshape, op, reduce, stream);
}

template <>
void u_gather_v_op_e_backward<kGPU>(CSR csr, const Idx* eidx, const float* vdata,
                                    const float* edata, const float* grad, float* const vgrad,
                                    float* const egrad, const std::vector<int32>& hshape,
                                    BinaryOpType op, ReduceType reduce, Stream* const stream) {
  plas::gpu::u_gather_v_op_e_backward(csr, eidx, vdata, edata, grad, vgrad, egrad, hshape, op,
                                      reduce, stream);
}

template <>
void u_gather_v_bcastop_e<kGPU>(CSR csr, float* const udata, const float* vdata, const float* edata,
                                BinaryOpType op, EWiseOpBcastInfo& info, ReduceType reduce,
                                Stream* const stream) {
  plas::gpu::u_gather_v_bcastop_e(csr, udata, vdata, edata, op, info, reduce, stream);
}

template <>
void u_gather_v_bcastop_e_backward<kGPU>(CSR csr, const Idx* eidx, const float* vdata,
                                         const float* edata, const float* grad, float* const vgrad,
                                         float* const egrad, BinaryOpType op,
                                         EWiseOpBcastInfo& info, ReduceType reduce,
                                         Stream* const stream) {
  plas::gpu::u_gather_v_bcastop_e_backward(csr, eidx, vdata, edata, grad, vgrad, egrad, op, info,
                                           reduce, stream);
}

template <>
void edge_softmax<kGPU>(CSR csr, const float* x, float* const y, const std::vector<int32>& shape,
                        Stream* const stream) {
  plas::gpu::edge_softmax(csr, x, y, shape, stream);
}

template <>
void edge_softmax_backward<kGPU>(CSR csr, const float* y, const float* grady, float* const gradx,
                                 const std::vector<int32>& shape, Stream* const stream) {
  plas::gpu::edge_softmax_backward(csr, y, grady, gradx, shape, stream);
}

#endif  // GOOGLE_CUDA

/***********************************************************************
***************************** Basic Graph Routines *********************
************************************************************************/

// CPU routines

template <>
void csr_to_coo<kCPU>(CSR csr, const Idx* eidx, COO& coo, Idx* coo_eidx, Stream* const stream) {
  plas::cpu::csr_to_coo(csr, eidx, coo, coo_eidx);
}

template <>
void csr_transpose<kCPU>(CSR csr, const Idx* eidx, CSR& t_csr, Idx* const t_eidx,
                         Stream* const stream) {
  plas::cpu::csr_transpose(csr, eidx, t_csr, t_eidx);
}

template <>
void node_degree<kCPU>(CSR csr, float* const deg, const bool is_u, Stream* const stream) {
  plas::cpu::node_degree(csr, deg, is_u);
}

#if GOOGLE_CUDA
// GPU routines

template <>
void csr_to_coo<kGPU>(CSR csr, const Idx* eidx, COO& coo, Idx* coo_eidx, Stream* const stream) {
  plas::gpu::csr_to_coo(csr, eidx, coo, coo_eidx, stream);
}

template <>
void csr_transpose<kGPU>(CSR csr, const Idx* eidx, CSR& t_csr, Idx* const t_eidx,
                         Stream* const stream) {
  plas::gpu::csr_transpose(csr, eidx, t_csr, t_eidx, stream);
}

template <>
void node_degree<kGPU>(CSR csr, float* const deg, const bool is_u, Stream* const stream) {
  plas::gpu::node_degree(csr, deg, is_u, stream);
}

#endif  // GOOGLE_CUDA

/***********************************************************************
***************************** Index Routine ****************************
************************************************************************/
template <>
void lookup_value_indices<kCPU>(const ID* data, const int32 data_len, const ID* query,
                                const int32 query_len, int32* const indices, Stream* const stream) {
  plas::cpu::lookup_value_indices(data, data_len, query, query_len, indices);
}

#if GOOGLE_CUDA
template <>
void lookup_value_indices<kGPU>(const ID* data, const int32 data_len, const ID* query,
                                const int32 query_len, int32* const indices, Stream* const stream) {
  plas::gpu::lookup_value_indices(data, data_len, query, query_len, indices, stream);
}

#endif  // GOOGLE_CUDA

}  // namespace plas
