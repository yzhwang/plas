#include "cc/plas/cpu/api.h"

#include <immintrin.h>
#include <omp.h>

namespace plas {
namespace cpu {

void csr_to_coo(CSR csr, const Idx* eidx, COO& coo, Idx* coo_eidx) {
#pragma omp parallel for
  for (int32 i = 0; i < csr.size_u; ++i) {
    for (Idx j = csr.indptr[i]; j < csr.indptr[i + 1]; ++j) {
      coo.idx_u[j] = i;
      coo.idx_v[j] = csr.indices[j];
    }
  }

  if (eidx != nullptr) {
// just copy eidx to coo_eidx
#pragma omp parallel for
    for (int32 i = 0; i < csr.size_e; ++i) {
      coo_eidx[i] = eidx[i];
    }
  }
}

void csr_transpose(CSR csr, const Idx* eidx, CSR& t_csr, Idx* const t_eidx) {
  const Idx* Ap = csr.indptr;
  const Idx* Aj = csr.indices;
  const Idx* Ax = eidx;

  Idx* const Bp = t_csr.indptr;
  Idx* const Bj = t_csr.indices;
  Idx* const Bx = t_eidx;

  t_csr.size_u = csr.size_v;
  t_csr.size_v = csr.size_u;
  t_csr.size_e = csr.size_e;

  // TODO(powergao) parallel this routine
  std::fill(Bp, Bp + csr.size_v, 0);
  for (int32 j = 0; j < csr.size_e; ++j) {
    Bp[Aj[j]]++;
  }

  for (int32 i = 0, cumsum = 0; i < csr.size_v; ++i) {
    const Idx tmp = Bp[i];
    Bp[i] = cumsum;
    cumsum += tmp;
  }
  Bp[csr.size_v] = csr.size_e;

  if (eidx == nullptr) {
    for (int32 i = 0; i < csr.size_u; ++i) {
      for (Idx j = Ap[i]; j < Ap[i + 1]; ++j) {
        const Idx dst = Aj[j];
        Bj[Bp[dst]] = i;
        Bp[dst]++;
      }
    }
  } else {
    for (int32 i = 0; i < csr.size_u; ++i) {
      for (Idx j = Ap[i]; j < Ap[i + 1]; ++j) {
        const Idx dst = Aj[j];
        Bj[Bp[dst]] = i;
        Bx[Bp[dst]] = Ax[j];
        Bp[dst]++;
      }
    }
  }

  for (int32 i = 0, last = 0; i <= csr.size_v; ++i) {
    Idx temp = Bp[i];
    Bp[i] = last;
    last = temp;
  }
}

void node_degree(CSR csr, float* const deg, const bool is_u) {
  if (is_u) {
    zero_float(deg, csr.size_u);
#pragma omp parallel for
    for (int32 i = 0; i < csr.size_u; ++i) {
      for (Idx j = csr.indptr[i]; j < csr.indptr[i + 1]; ++j) {
        deg[i] += 1.f;
      }
    }

  } else {
    zero_float(deg, csr.size_v);
#pragma omp parallel for
    for (int32 i = 0; i < csr.size_u; ++i) {
      for (Idx j = csr.indptr[i]; j < csr.indptr[i + 1]; ++j) {
#pragma omp atomic
        deg[csr.indices[j]] += 1.f;
      }
    }
  }
}

}  // namespace cpu
}  // namespace plas
