#ifndef PLAS_GPU_GRAPH_OP_KERNEL_H_
#define PLAS_GPU_GRAPH_OP_KERNEL_H_

#include "scan_kernel.h"
#include "memory.h"
#include "kernel_launch.h"

namespace plas {
namespace gpu {

/**
 * @brief Generate COO from CSR.
 *
 * @tparam index_it:     type of index;
 *
 * @param csr_indptr:  CSR's index pointer array;
 * @param csr_indices: CSR's indices array;
 * @param csr_idx_e:   CSR's edge indices array (could be nullptr);
 * @param coo_idx_u:   COO's source indices array;
 * @param coo_idx_v:   COO's destination indices array;
 * @param coo_idx_e:   COO's edge indices array (could be nullptr);
 * @param count:       CSR's node count;
 * @param context:     reference to CUDA context singleton instance.
 */
template <typename index_it, typename const_index_it, bool has_edge_id = false>
void csr2coo(index_it csr_indptr, index_it csr_indices, const_index_it csr_idx_e,
             index_it coo_idx_u, index_it coo_idx_v, index_it coo_idx_e,
             int count, context_t& context) {
  auto flat_edges_no_edge_id = [=] PLAS_DEVICE(int index) {
    int idx_start = ldg(csr_indptr + index);
    int neighbor_length = ldg(csr_indptr + index + 1) - idx_start;
    for (int i = idx_start; i < idx_start + neighbor_length; ++i) {
      coo_idx_u[i] = index;
      coo_idx_v[i] = ldg(csr_indices + i);
    }
  };
  auto flat_edges_edge_id = [=] PLAS_DEVICE(int index) {
    int idx_start = ldg(csr_indptr + index);
    int neighbor_length = ldg(csr_indptr + index + 1) - idx_start;
    for (int i = idx_start; i < idx_start + neighbor_length; ++i) {
      coo_idx_u[i] = index;
      coo_idx_v[i] = ldg(csr_indices + i);
      coo_idx_e[i] = csr_idx_e[i];
    }
  };
  if (has_edge_id) {
    assert(nullptr != coo_idx_e && nullptr != csr_idx_e);
    transform(flat_edges_edge_id, count, context);
  } else {
    transform(flat_edges_no_edge_id, count, context);
  }
}

/**
 * @brief Transpose the CSR graph into TCSR (CSC).
 *
 * @tparam index_it:     type of index;
 *
 * @param csr_indptr:    CSR's index pointer array;
 * @param csr_indices:   CSR's indices array;
 * @param csr_idx_e:     CSR's edge indices array (could be nullptr);
 * @param t_csr_indptr:  Transposed CSR's index pointer array;
 * @param t_csr_indices: Transposed CSR's indices array;
 * @param t_csr_idx_e:   Transposed CSR's edge indices array (could be nullptr);
 * @param count_u:       CSR's node count;
 * @param count_v:       Transposed CSR's node count;
 * @param count_e:       Edge count of the graph;
 * @param context:       reference to CUDA context singleton instance.
 */
template <typename index_it, typename const_index_it, typename index_it_const,
          bool has_edge_id = false>
void csr_transpose(index_it csr_indptr, index_it csr_indices,
                   const_index_it csr_idx_e, index_it t_csr_indptr,
                   index_it t_csr_indices, index_it_const t_csr_idx_e, int count_u,
                   int count_v, int count_e, context_t& context) {
  typedef typename std::iterator_traits<index_it>::value_type type_t;
  auto indptr_init = [=] PLAS_DEVICE(int index) { t_csr_indptr[index] = 0; };
  transform(indptr_init, count_v, context);

  auto get_degree = [=] PLAS_DEVICE(int index) {
    int idx_start = ldg(csr_indptr + index);
    int neighbor_length = ldg(csr_indptr + index + 1) - idx_start;
    for (int i = idx_start; i < idx_start + neighbor_length; ++i) {
      type_t dst_idx = ldg(csr_indices + i);
      atomicInc(t_csr_indptr + dst_idx, count_e);
    }
  };
  transform(get_degree, count_u, context);

  scan(t_csr_indptr, count_v + 1, t_csr_indptr, context);

  // TODO(slashwang): atomic's performance is poor. Plus the neighbor IDs
  // are unsorted in this way.
  auto set_indices_no_edge_id = [=] PLAS_DEVICE(int index) {
    int idx_start = ldg(csr_indptr + index);
    int neighbor_length = ldg(csr_indptr + index + 1) - idx_start;
    for (int i = idx_start; i < idx_start + neighbor_length; ++i) {
      type_t dst_idx = ldg(csr_indices + i);
      type_t tcsr_indptr = atomicAdd(t_csr_indptr + dst_idx, 1);
      t_csr_indices[tcsr_indptr] = index;
    }
  };

  auto set_indices_edge_id = [=] PLAS_DEVICE(int index) {
    int idx_start = ldg(csr_indptr + index);
    int neighbor_length = ldg(csr_indptr + index + 1) - idx_start;
    for (int i = idx_start; i < idx_start + neighbor_length; ++i) {
      type_t dst_idx = ldg(csr_indices + i);
      type_t tcsr_indptr = atomicAdd(t_csr_indptr + dst_idx, 1);
      t_csr_indices[tcsr_indptr] = index;
      t_csr_idx_e[tcsr_indptr] = ldg(csr_idx_e + i);
    }
  };
  if (has_edge_id) {
    assert(nullptr != t_csr_idx_e && nullptr != csr_idx_e);
    transform(set_indices_edge_id, count_u, context);
  } else {
    transform(set_indices_no_edge_id, count_u, context);
  }

  // TODO(slashwang): Fix this stupid repetative work.
  // easier to do it this way, than any other way I could think of.
  transform(indptr_init, count_v, context);
  transform(get_degree, count_u, context);
  scan(t_csr_indptr, count_v + 1, t_csr_indptr, context);
}

/**
 * @brief Get degrees of a graph.
 *
 * @tparam index_it:     type of index;
 *
 * @param csr_indptr:    CSR's index pointer array;
 * @param csr_indices:   CSR's indices array;
 * @param count_u:       CSR's node count;
 * @param count_v:       Transposed CSR's node count;
 * @param degrees:       Computed degrees;
 * @param context:       reference to CUDA context singleton instance.
 */
template <typename index_it, bool is_u = true>
void get_degrees(index_it csr_indptr, index_it csr_indices, int count_u,
                 int count_v, float* const degrees, context_t& context) {
  typedef typename std::iterator_traits<index_it>::value_type type_t;

  auto get_degree_u = [=] PLAS_DEVICE(int index) {
    degrees[index] = csr_indptr[index + 1]*1.0f - csr_indptr[index];
  };

  auto degree_init = [=] PLAS_DEVICE(int index) { degrees[index] = 0.0f; };

  auto get_degree_v = [=] PLAS_DEVICE(int index) {
    int idx_start = ldg(csr_indptr + index);
    int neighbor_length = ldg(csr_indptr + index + 1) - idx_start;
    for (int i = idx_start; i < idx_start + neighbor_length; ++i) {
      type_t dst_idx = ldg(csr_indices + i);
      atomicAdd(degrees + dst_idx, 1.0f);
    }
  };

  if (is_u) {
    transform(get_degree_u, count_u, context);
  } else {
    transform(degree_init, count_v, context);
    transform(get_degree_v, count_u, context);
  }
}

}  // namespace gpu
}  // namespace plas

#endif
