#ifndef CC_PLAS_CPU_API_H
#define CC_PLAS_CPU_API_H

#include <vector>
#include "cc/plas/types.h"

namespace plas {
namespace cpu {
/***************** Basic Linear Algrebra Kernels **********/

/**
 * @brief broadcast element-wise multiply alpha and x, and add results to y.
 *        Assume, alpha is the one to be broadcasted, x and y are vector of the same shape.
 *        y += broadcast(alpha) * x
 * @param alpha input, data to be broadcasted
 * @param x input, vector
 * @param y input, vector
 * @param dim input, dimension of vector x and y
 */
void saxpy(const float alpha, const float* x, float* const y, int32 dim);

/**
 * @brief perform dot product of vector x and y, add results in out
 *
 * @param x Input, vector x
 * @param y Input, vector y
 * @param dim Input, vector length
 * @param out Output, results
 */
void sdot(const float* x, const float* y, int32 dim, float* const out);

/**
 * @brief Element-wise fused multiply-add op. out += x * y.
 *
 * @param x input, vector
 * @param y input, vector
 * @param out output, vector
 * @param dim input, vector dimension
 */
void ewise_fma(const float* x, const float* y, float* const out, int32 dim);

/**
 * @brief Element-wise add op. out [i] += x[i]
 *
 * @param x Input, vector
 * @param out Output, vector
 * @param dim Input, vector length
 */
void ewise_add(const float* x, float* const out, int32 dim);

/**
 * @brief Element-wise copy op. dst[i] = src[i]
 *
 * @param dst Output, vector
 * @param src Input, vector
 * @param dim Input, vector length
 */
void ewise_copy(float* const dst, const float* src, int32 dim);

inline void zero_float(float* const data, int32 len) {
#pragma omp parallel
  {
    int num_thr = omp_get_num_threads();
    int chunk = len / num_thr;
    int tid = omp_get_thread_num();
    float* const st = data + chunk * tid;
    float* const en = (tid == num_thr - 1) ? data + len : st + chunk;
    std::memset(st, 0, (en - st) * sizeof(float));
  }
}

/***************** Graph Aggregation Kernels *****************/
void u_gather_v(CSR csr, float* const udata, const float* vdata, const std::vector<int32>& hshape,
                ReduceType reduce);

void u_gather_e(CSR csr, float* const udata, const float* edata, const std::vector<int32>& hshape,
                ReduceType reduce);

void u_scatter_e(CSR csr, const float* udata, float* const edata, const std::vector<int32>& hshape);

void u_gather_v_op_e(CSR csr, float* const udata, const float* vdata, const float* edata,
                     const std::vector<int32>& hshape, BinaryOpType op, ReduceType reduce);

void u_gather_v_op_e_backward(CSR csr, const Idx* eidx, const float* vdata, const float* edata,
                              const float* grad, float* const vgrad, float* const egrad,
                              const std::vector<int32>& hshape, BinaryOpType op, ReduceType reduce);

void u_gather_v_bcastop_e(CSR csr, float* const udata, const float* vdata, const float* edata,
                          BinaryOpType op, EWiseOpBcastInfo& info, ReduceType reduce);

void u_gather_v_bcastop_e_backward(CSR csr, const Idx* eidx, const float* vdata, const float* edata,
                                   const float* grad, float* const vgrad, float* const egrad,
                                   BinaryOpType op, EWiseOpBcastInfo& info, ReduceType reduce);

void edge_softmax(CSR csr, const float* x, float* const y, const std::vector<int32>& shape);

void edge_softmax_backward(CSR csr, const float* y, const float* grady, float* const gradx,
                           const std::vector<int32>& shape);

/**************** Basic Graph Operators *******************/
void csr_to_coo(CSR csr, const Idx* eidx, COO& coo, Idx* coo_eidx);

void csr_transpose(CSR csr, const Idx* eidx, CSR& t_csr, Idx* const t_eidx);

void node_degree(CSR csr, float* const deg, const bool is_u);

/************************** indices kernels **************/
void lookup_value_indices(const ID* data, const int32 data_len, const ID* query,
                          const int32 query_len, int32* const indices);

template <typename T>
bool is_sorted_ascending(T const* data, int32 len) {
  CHECK_GT(len, 0) << "vector length should be GT 0";
  int tid_size = -1;
#pragma omp parallel
  { tid_size = omp_get_num_threads(); }
  std::vector<bool> tid_sorted(tid_size, true);
  int chunk = (len + tid_size - 1) / tid_size;

  bool sorted = true;

#pragma omp parallel shared(tid_sorted)
  {
    int tid = omp_get_thread_num();
    int st = std::min(len, chunk * tid);
    int en = std::min(len, chunk * (tid + 1));

    bool am_sorted = true;
    for (int i = st; i < en - 1; ++i) {
      if (data[i] > data[i + 1]) {
        am_sorted = false;
        // LOG(INFO) << "tid " << tid << " / " << tid_size << " break";
        break;
      }
    }

#pragma omp critical
    tid_sorted[tid] = am_sorted;
  }

  __sync_synchronize();
  asm volatile("" : : : "memory");

  // check boundary
  for (int tid = 0; tid < tid_size; ++tid) {
    int idx = std::min(len, chunk * tid);
    if (idx == 0) continue;
    if (data[idx - 1] > data[idx]) {
      sorted = false;
      break;
    }
  }

  for (size_t i = 0; i < tid_sorted.size(); ++i) {
    sorted = sorted && tid_sorted[i];
  }
  return sorted;
}

template <typename T>
inline std::string vec2str(const T* const data, int len) {
  std::string info;
  for (int i = 0; i < len; ++i) {
    info += std::to_string(data[i]) + ", ";
  }
  return info;
}

}  // namespace cpu
}  // namespace plas

#endif
