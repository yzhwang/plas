#ifndef PLAS_GPU_TYPES_H_
#define PLAS_GPU_TYPES_H_

#include <cstdint>

namespace plas {
using uint64 = std::uint64_t;
using int64 = std::int64_t;
using int32 = int;
using uint32 = unsigned int;

using Idx = uint32;
using ID = uint64;

struct CSR {
  Idx* indptr;
  Idx* indices;
  int32 size_u;  // number of u-side node
  int32 size_v;  // number of v-side node
  int32 size_e;  // number of edges(nnz)
};

struct COO {
  Idx* idx_u;  // with length of size_e, first element of coo
  Idx* idx_v;  // with length of size_e, second element of coo
  int32 size_u;
  int32 size_v;
  int32 size_e;
};

}  // namespace plas

#endif
