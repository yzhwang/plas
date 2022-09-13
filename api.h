#ifndef CC_PLAS_API_H
#define CC_PLAS_API_H

#include <algorithm>
#include <cstring>
#include "cc/plas/cpu/api.h"
#if GOOGLE_CUDA
#include "cc/plas/gpu/api.h"
#endif
#include "cc/plas/types.h"

namespace plas {

/***********************************************************************
***************** Graph Neural Network Aggregation Routines ************
************************************************************************/

/**
 * @brief Gather V-side node data to U-side node. IMPORTANCE: udata and vdata should have the same
 * hidden shape. i.e. broadcast is not supported.
 *
 * @tparam XPU device type, CPU or GPU
 * @param csr Input, given graph in csr format
 * @param udata Output, u-side node data
 * @param vdata Input, v-side node data
 * @param hshape Input, hidden shape, which is data shape stripped Graph dimension(node dimension or
 * edge dimension)
 * @param reduce Input, reduce type
 * @param stream Input, for GPU device.
 */
template <DeviceType XPU>
void u_gather_v(CSR csr, float* const udata, const float* vdata, const std::vector<int32>& hshape,
                ReduceType reduce, Stream* const stream) {
  LOG(FATAL) << "Not implemented for device " << XPU;
}

template <>
void u_gather_v<kCPU>(CSR csr, float* const udata, const float* vdata,
                      const std::vector<int32>& hshape, ReduceType reduce, Stream* const stream);

#if GOOGLE_CUDA
template <>
void u_gather_v<kGPU>(CSR csr, float* const udata, const float* vdata,
                      const std::vector<int32>& hshape, ReduceType reduce, Stream* const stream);
#endif

/**
 * @brief Gather Edge node data to U-side node data. IMPORTANCE: udata and edata should have the
 * same
 * feature(hidden) shape, i.e. broadcast sementic is not supported.
 *
 * @tparam XPU device type, CPU or GPU
 * @param csr Input, given graph in csr format
 * @param udata Output, u-side node data
 * @param edata Input, data associated with edges
 * @param hshape Input, feature(hidden) shape, i.e. data shape stripped off graph dimensions
 * @param reduce Input, reduce type
 * @param stream Input, for GPU device
 */
template <DeviceType XPU>
void u_gather_e(CSR csr, float* const udata, const float* edata, const std::vector<int32>& hshape,
                ReduceType reduce, Stream* const stream) {
  LOG(FATAL) << "Not implemented for device " << XPU;
}

template <>
void u_gather_e<kCPU>(CSR csr, float* const udata, const float* edata,
                      const std::vector<int32>& hshape, ReduceType reduce, Stream* const stream);

#if GOOGLE_CUDA
template <>
void u_gather_e<kGPU>(CSR csr, float* const udata, const float* edata,
                      const std::vector<int32>& hshape, ReduceType reduce, Stream* const stream);
#endif

/**
 * @brief Scatter U-side node data to edges.
 *
 * @tparam XPU device type, CPU or GPU
 * @param csr Input, given graph in csr format
 * @param udata Input, u-side node data
 * @param edata Output, edge data
 * @param hshape Input, feature shape
 * @param stream Input, for GPU device
 */
template <DeviceType XPU>
void u_scatter_e(CSR csr, const float* udata, float* const edata, const std::vector<int32>& hshape,
                 Stream* const stream) {
  LOG(FATAL) << "Not implemented for device " << XPU;
}

template <>
void u_scatter_e<kCPU>(CSR csr, const float* udata, float* const edata,
                       const std::vector<int32>& hshape, Stream* const stream);

#if GOOGLE_CUDA
template <>
void u_scatter_e<kGPU>(CSR csr, const float* udata, float* const edata,
                       const std::vector<int32>& hshape, Stream* const stream);
#endif


/**
 * @brief Gather V-side node data and edge data by
 *        1. tmp = binop(vdata, edata)
 *        2. udata = reduce(tmp) for all edges
 *        IMPORTANCE: broadcast semantic is not supported for binop.
 * @tparam XPU device type. Cpu or GPU
 * @param csr Input, given graph in csr format
 * @param udata Output, u-side node data
 * @param vdata Input, v-side node data
 * @param edata Input, edge data
 * @param hshape Input, feature shape, i.e. shape stripped graph dimension(node dim or edge dim)
 * @param op Input, BinaryOpType
 * @param reduce Input, ReduceType
 * @param stream Input, for GPU device.
 */
template <DeviceType XPU>
void u_gather_v_op_e(CSR csr, float* const udata, const float* vdata, const float* edata,
                     const std::vector<int32>& hshape, BinaryOpType op, ReduceType reduce,
                     Stream* const stream) {
  LOG(FATAL) << "Not implemented for device " << XPU;
}
template <>
void u_gather_v_op_e<kCPU>(CSR csr, float* const udata, const float* vdata, const float* edata,
                           const std::vector<int32>& hshape, BinaryOpType op, ReduceType reduce,
                           Stream* const stream);

#if GOOGLE_CUDA
template <>
void u_gather_v_op_e<kGPU>(CSR csr, float* const udata, const float* vdata, const float* edata,
                           const std::vector<int32>& hshape, BinaryOpType op, ReduceType reduce,
                           Stream* const stream);
#endif


/**
 * @brief Backward computation of `u_gather_v_op_e`
 *        IMPORTANCE: broadcast is not supported
 * @tparam XPU device type, CPU or GPU
 * @param csr Input, given graph in csr format, notice that for backward, we assume csr is the
 * transposed of the csr in forward computation
 * @param eidx Input, edge data index, consider the csr is transposed.
 * @param vdata Input, v-side node data
 * @param edata Input, edge data
 * @param grad Input, gradient
 * @param vgrad Output, gradient of v-side node data
 * @param egrad Output, gradient of edge data
 * @param hshape Input, feature shape
 * @param op Input, BinaryOpType
 * @param reduce Input, ReduceType
 * @param stream Input, for GPU device.
 */
template <DeviceType XPU>
void u_gather_v_op_e_backward(CSR csr, const Idx* eidx, const float* vdata, const float* edata,
                              const float* grad, float* const vgrad, float* const egrad,
                              const std::vector<int32>& hshape, BinaryOpType op, ReduceType reduce,
                              Stream* const stream) {
  LOG(FATAL) << "Not implemented for device " << XPU;
}

template <>
void u_gather_v_op_e_backward<kCPU>(CSR csr, const Idx* eidx, const float* vdata,
                                    const float* edata, const float* grad, float* const vgrad,
                                    float* const egrad, const std::vector<int32>& hshape,
                                    BinaryOpType op, ReduceType reduce, Stream* const stream);

#if GOOGLE_CUDA
template <>
void u_gather_v_op_e_backward<kGPU>(CSR csr, const Idx* eidx, const float* vdata,
                                    const float* edata, const float* grad, float* const vgrad,
                                    float* const egrad, const std::vector<int32>& hshape,
                                    BinaryOpType op, ReduceType reduce, Stream* const stream);
#endif


/**
 * @brief Gather neighbors via:
 *        step 1. tmp_j = bcast_op(v_j, e_j)
 *        step 2. u_i = reduce(tmp_j0, tmp_j1)
 * IMPORTANCE: bcast_op must have broadcast semantic.
 * @tparam XPU device type, CPU or GPU
 * @param csr Input, graph in CSR
 * @param udata Output, gathered data
 * @param vdata Input, V-side node data
 * @param edata Input, edge data
 * @param op Input, binary op that will applied on edge and vdata
 * @param info Input, broadcast info
 * @param reduce Input, reduction type
 * @param stream Input, for GPU
 */
template <DeviceType XPU>
void u_gather_v_bcastop_e(CSR csr, float* const udata, const float* vdata, const float* edata,
                          BinaryOpType op, EWiseOpBcastInfo& info, ReduceType reduce,
                          Stream* const stream) {
  LOG(FATAL) << "Not implemented for device " << XPU;
}

template <>
void u_gather_v_bcastop_e<kCPU>(CSR csr, float* const udata, const float* vdata, const float* edata,
                                BinaryOpType op, EWiseOpBcastInfo& info, ReduceType reduce,
                                Stream* const stream);

#if GOOGLE_CUDA
template <>
void u_gather_v_bcastop_e<kGPU>(CSR csr, float* const udata, const float* vdata, const float* edata,
                                BinaryOpType op, EWiseOpBcastInfo& info, ReduceType reduce,
                                Stream* const stream);
#endif

/**
 * @brief Backward computation of `u_gather_v_op_e`
 * @tparam XPU device type, CPU or GPU
 * @param csr Input, given graph in csr format, notice that for backward, we assume csr is the
 * transposed of the csr in forward computation
 * @param eidx Input, edge data index, consider the csr is transposed.
 * @param vdata Input, v-side node data
 * @param edata Input, edge data
 * @param grad Input, gradient
 * @param vgrad Output, gradient of v-side node data
 * @param egrad Output, gradient of edge data
 * @param op Input, BinaryOpType
 * @param info Input, broadcast info
 * @param reduce Input, ReduceType
 * @param stream Input, for GPU device.
 */
template <DeviceType XPU>
void u_gather_v_bcastop_e_backward(CSR csr, const Idx* eidx, const float* vdata, const float* edata,
                                   const float* grad, float* const vgrad, float* const egrad,
                                   BinaryOpType op, EWiseOpBcastInfo& info, ReduceType reduce,
                                   Stream* const stream) {
  LOG(FATAL) << "Not implemented for device " << XPU;
}

template <>
void u_gather_v_bcastop_e_backward<kCPU>(CSR csr, const Idx* eidx, const float* vdata,
                                         const float* edata, const float* grad, float* const vgrad,
                                         float* const egrad, BinaryOpType op,
                                         EWiseOpBcastInfo& info, ReduceType reduce,
                                         Stream* const stream);

#if GOOGLE_CUDA
template <>
void u_gather_v_bcastop_e_backward<kGPU>(CSR csr, const Idx* eidx, const float* vdata,
                                         const float* edata, const float* grad, float* const vgrad,
                                         float* const egrad, BinaryOpType op,
                                         EWiseOpBcastInfo& info, ReduceType reduce,
                                         Stream* const stream);
#endif


/**
 * @brief Perform softmax over edge values
 *
 * @tparam XPU device type, CPU or GPU
 * @param csr Input, given graph in csr format
 * @param x Input, data associated with edge
 * @param y Output, softmaxed value
 * @param shape Input, shape of edge data
 * @param stream Input, for GPU
 */
template <DeviceType XPU>
void edge_softmax(CSR csr, const float* x, float* const y, const std::vector<int32>& shape,
                  Stream* const stream) {
  LOG(FATAL) << "Not implemented for device " << XPU;
}

template <>
void edge_softmax<kCPU>(CSR csr, const float* x, float* const y, const std::vector<int32>& shape,
                        Stream* const stream);

#if GOOGLE_CUDA
template <>
void edge_softmax<kGPU>(CSR csr, const float* x, float* const y, const std::vector<int32>& shape,
                        Stream* const stream);
#endif

/**
 * @brief Backward computation of edge-softmax. IMPORTANCE: broadcast is not supported.
 *
 * @tparam XPU device type, CPU or GPU
 * @param csr Input, given graph in csr format
 * @param y Input, softmaxed edge data
 * @param grady Input, gradient of y
 * @param gradx Output, gradient of x(original value of edge data)
 * @param shape Input, shape of edge data
 * @param stream Input, for GPU
 */
template <DeviceType XPU>
void edge_softmax_backward(CSR csr, const float* y, const float* grady, float* const gradx,
                           const std::vector<int32>& shape, Stream* const stream) {
  LOG(FATAL) << "Not implemented for device " << XPU;
}

template <>
void edge_softmax_backward<kCPU>(CSR csr, const float* y, const float* grady, float* const gradx,
                                 const std::vector<int32>& shape, Stream* const stream);

#if GOOGLE_CUDA
template <>
void edge_softmax_backward<kGPU>(CSR csr, const float* y, const float* grady, float* const gradx,
                                 const std::vector<int32>& shape, Stream* const stream);
#endif

/***********************************************************************
***************************** Basic Graph Routines *********************
************************************************************************/

/**
 * @brief Convert CSR to COO, if eidx is not nullptr, it will also permutate edge values when
 * converting.
 *
 * @tparam XPU
 * @param csr Input, graph in CSR
 * @param eidx Input, edge index, if not nullptr, permutated edge indices are set in coo_eidx
 * @param coo Output, graph in COO format
 * @param coo_eidx Output, edge index in COO format
 */
template <DeviceType XPU>
void csr_to_coo(CSR csr, const Idx* eidx, COO& coo, Idx* coo_eidx, Stream* const stream) {
  LOG(FATAL) << "Not implemented for device " << XPU;
}

template <>
void csr_to_coo<kCPU>(CSR csr, const Idx* eidx, COO& coo, Idx* coo_eidx, Stream* const stream);

#if GOOGLE_CUDA
template <>
void csr_to_coo<kGPU>(CSR csr, const Idx* eidx, COO& coo, Idx* coo_eidx, Stream* const stream);
#endif

/**
 * @brief Transpose given CSR.
 *
 * @tparam XPU device type, CPU or GPU
 * @param csr Input, CSR
 * @param eidx Input, edge index in CSR representation, if nullptr, edge index won't be transposed
 * @param t_csr Output, CSC
 * @param t_eidx Output, edge index in CSC representation
 */
template <DeviceType XPU>
void csr_transpose(CSR csr, const Idx* eidx, CSR& t_csr, Idx* const t_eidx, Stream* const stream) {
  LOG(FATAL) << "Not implemented for device " << XPU;
}

template <>
void csr_transpose<kCPU>(CSR csr, const Idx* eidx, CSR& t_csr, Idx* const t_eidx,
                         Stream* const stream);

#if GOOGLE_CUDA
template <>
void csr_transpose<kGPU>(CSR csr, const Idx* eidx, CSR& t_csr, Idx* const t_eidx,
                         Stream* const stream);
#endif


/**
 * @brief Get the degree of nodes in CSR
 *
 * @tparam XPU device type, CPU or GPU
 * @param csr Input, CSR
 * @param deg Output, degree
 * @param is_u Input, true on get u-side node degree
 */
template <DeviceType XPU>
void node_degree(CSR csr, float* const deg, const bool is_u, Stream* const stream) {
  LOG(FATAL) << "Not implemented for device " << XPU;
}

template <>
void node_degree<kCPU>(CSR csr, float* const deg, const bool is_u, Stream* const stream);

#if GOOGLE_CUDA
template <>
void node_degree<kGPU>(CSR csr, float* const deg, const bool is_u, Stream* const stream);
#endif

/***********************************************************************
***************************** Index Routine ****************************
************************************************************************/

/**
 * @brief Lookup the indices of `query` in `data`. IMPORTANCE: query must appear once in data.
 *
 * @tparam XPU device type, CPU or GPU
 * @param data Input, the data to lookup from
 * @param data_len Input, data length
 * @param query Input, the query to lookup
 * @param query_len Input, query length
 * @param indices Output, the answer
 */
template <DeviceType XPU>
void lookup_value_indices(const ID* data, const int32 data_len, const ID* query,
                          const int32 query_len, int32* const indices, Stream* const stream) {
  LOG(FATAL) << "Not Implemented for device " << XPU;
}

template <>
void lookup_value_indices<kCPU>(const ID* data, const int32 data_len, const ID* query,
                                const int32 query_len, int32* const indices, Stream* const stream);

#if GOOGLE_CUDA
template <>
void lookup_value_indices<kGPU>(const ID* data, const int32 data_len, const ID* query,
                                const int32 query_len, int32* const indices, Stream* const stream);
#endif


}  // namespace plas

#endif
