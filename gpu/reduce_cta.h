#ifndef PLAS_GPU_REDUCE_CTA_H_
#define PLAS_GPU_REDUCE_CTA_H_

#include "intrinsics.h"
#include "udt.h"

namespace plas {
namespace gpu {

// warp_reduce_t reduces group_size (group_size <= warp_size) of elements and
// returns the result in lane 0
template <typename type_t, int group_size>
struct warp_reduce_t {
  static_assert(
      group_size <= warp_size && is_pow2(group_size),
      "warp_reduce_t only operates on a pow2 of threads <= warp_size (32)");
  enum {
    num_passes = s_log2(group_size)
  };

  template <typename op_t = plus_t<type_t> >
  PLAS_DEVICE type_t reduce(int lane, type_t x, int count, op_t op = op_t()) {
    if (count == group_size) {
      // can fuse op with shfl
      iterate<num_passes>([&](int pass) {
        int offset = 1 << pass;
        x = shfl_down_op(x, offset, op, group_size);
      });
    } else {
      // shfl then op
      iterate<num_passes>([&](int pass) {
        int offset = 1 << pass;
        type_t y = shfl_down(x, offset, group_size);
        if (lane + offset < count) x = op(x, y);
      });
    }
    return x;
  }
};

// cta_reduce_t returns the reduction of all inputs for thread 0, and returns
// type_t() for all other threads. This behavior saves a broadcast.
template <int nt, typename type_t>
struct cta_reduce_t {
  enum {
    group_size = min(nt, (int)warp_size),
    num_passes = s_log2(group_size),
    num_items = nt / group_size
  };

  static_assert(
      0 == nt % warp_size,
      "cta_reduce_t requires num threads to be a multiple of warp_size (32)");

  struct storage_t {
    struct {
      type_t data[max(nt, 2 * group_size)];  // NOLINT
    };
  };

  typedef warp_reduce_t<type_t, group_size> group_reduce_t;

  template <typename op_t = plus_t<type_t> >
  PLAS_DEVICE type_t
  reduce(int tid, type_t x, storage_t& storage, int count = nt,
         op_t op = op_t(), bool all_return = true) const {
    // Store your data into shmem.
    storage.data[tid] = x;
    __syncthreads();

    if (tid < group_size) {
      // Each thread scans within its lane.
      strided_iterate<group_size, num_items>([&](int i, int j) {
                                               if (i > 0)
                                                 x = op(x, storage.data[j]);
                                             },
                                             tid, count);

      // cooperative reduction.
      x = group_reduce_t().reduce(tid, x, min(count, (int)group_size), op);

      if (all_return) storage.data[tid] = x;
    }
    __syncthreads();

    if (all_return) {
      x = storage.data[0];
      __syncthreads();
    }
    return x;
  }
};

template <int nt, int vt, int shmem_dim, typename type_t>
struct cta_vector_reduce_t {
  // reduce at most 64 shmem_dim(64)-length vectors.
  // one warp reduce (with possible one local reduce
  // to reduce at most 64 elements into 32 elements)
  // one col. Each thread operates on 4 col. With a
  // 512 thread number in a block, 4 * (512/32=16)=64
  // So everything should be finish within one round.

  struct storage_t {
    struct {
      type_t data[shmem_dim * 64];  // NOLINT
    };
  };

  typedef warp_reduce_t<type_t, warp_size> row_reduce_t;

  template <typename op_t = plus_t<type_t> >
  PLAS_DEVICE void reduce(int tid, type_t* result, storage_t& storage,
                          int count, op_t op = op_t()) const {
    int warp_id = tid / warp_size;
    int lane_id = tid & (warp_size - 1);

    // load from shmem to reg
    array_t<type_t, vt> x;
    int shmem_idx = warp_id * vt + lane_id * shmem_dim;
    int shmem_stride = warp_size * shmem_dim;
    iterate<vt>([&](int i) {
      type_t first = (lane_id < count) ? storage.data[shmem_idx + i] : 0;
      type_t second = (lane_id + warp_size < count)
                          ? storage.data[shmem_idx + i + shmem_stride]
                          : 0;
      x[i] = op(first, second);
    });
    __syncthreads();

    // Cooperative reduction.
    iterate<vt>([&](int i) {
      x[i] =
          row_reduce_t().reduce(lane_id, x[i], min(count, (int)warp_size), op);
    });

    // Write back to result for each warp's first lane.
    if (lane_id == 0) {
      int result_idx = warp_id * vt;
      iterate<vt>([&](int i) {
        type_t old_result = result[result_idx + i];
        result[result_idx + i] = op(old_result, x[i]);
      });
    }
  }
};

}  // namespace gpu
}  // namespace plas

#endif
