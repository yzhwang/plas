#ifndef PLAS_GPU_REDUCE_BASELINE_H_
#define PLAS_GPU_REDUCE_BASELINE_H_

#include "../memory.h"
#include "../loadstore.h"
#include "../kernel_launch.h"

#include "../intrinsics.h"
#include "../udt.h"

namespace plas {
namespace gpu {

template <typename T>
__device__ void warp_reduce_naive(volatile T* shmem_data, T tid) {
  shmem_data[tid] += shmem_data[tid + 32];
  shmem_data[tid] += shmem_data[tid + 16];
  shmem_data[tid] += shmem_data[tid + 8];
  shmem_data[tid] += shmem_data[tid + 4];
  shmem_data[tid] += shmem_data[tid + 2];
  shmem_data[tid] += shmem_data[tid + 1];
}

// warp_reduce_shfl reduces group_size (group_size <= warp_size) of elements and
// returns the result in lane 0
template <typename type_t, int group_size>
struct warp_reduce_shfl {
  static_assert(
      group_size <= warp_size && is_pow2(group_size),
      "warp_reduce_t only operates on a pow2 of threads <= warp_size (32)");
  enum {
    num_passes = s_log2(group_size)
  };

  PLAS_DEVICE type_t reduce(int lane, type_t x, int count) {
    if (count == group_size) {
      // can fuse op with shfl
      iterate<num_passes>([&](int pass) {
        int offset = 1 << pass;
        x = shfl_down_op(x, offset, plus_t<type_t>(), group_size);
      });
    } else {
      // shfl then op
      iterate<num_passes>([&](int pass) {
        int offset = 1 << pass;
        type_t y = shfl_down(x, offset, group_size);
        if (lane + offset < count) x += y;
      });
    }
    return x;
  }
};

// cta_reduce returns the reduction of all inputs for thread 0, and returns
// type_t() for all other threads. This behavior saves a broadcast.
template <int nt, typename type_t>
struct cta_reduce {
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

  typedef warp_reduce_shfl<type_t, group_size> group_reduce_t;

  PLAS_DEVICE type_t
  reduce(int tid, type_t x, storage_t& storage, int count = nt) const {
    // Store your data into shmem.
    storage.data[tid] = x;
    __syncthreads();

    if (tid < group_size) {
      // Each thread scans within its lane.
      strided_iterate<group_size, num_items>([&](int i, int j) {
                                               if (i > 0)
                                                 x += storage.data[j];
                                             },
                                             tid, count);

      // cooperative reduction.
      x = group_reduce_t().reduce(tid, x, min(count, (int)group_size));

    }
    __syncthreads();
    return x;
  }
};

template <typename input_it, typename output_it>
void reduce_baseline(input_it input, int count, output_it reduction, context_t& context) {
  typedef launch_params_t<256, 1> launch_t;

  typedef typename std::iterator_traits<input_it>::value_type type_t;

  int num_ctas = launch_t::cta_dim(context).num_ctas(count);
  mem_t<type_t> partials(num_ctas, context);
  type_t* partials_data = partials.data();

  // Reduce each tile to a scalar and store to partials.
  auto k = [=] PLAS_DEVICE(int tid, int cta) {
    typedef typename launch_t::sm_ptx params_t;
    enum {
      nt = params_t::nt,
      vt = params_t::vt,
      nv = nt * vt
    };

    __shared__ type_t shared_vals[nt];

    // Load global data into shmem
    // Note we ignore boundary check and assume that the total data size is
    // always divisible by the number of threads in a block.
    shared_vals[tid] = ldg(input + cta * nt + tid);
    __syncthreads();

    // Reduce across all threads.
    for (unsigned int s = 1; s < nt; s *= 2) {
      if (tid % (2*s) == 0) {
        shared_vals[tid] += shared_vals[tid + s];
      }
      __syncthreads();
    }

    // Store the final reduction to the partials.
    if (!tid) {
      if (1 == num_ctas) *reduction = shared_vals[0];
      else partials_data[cta] = shared_vals[0];
    }
  };
  cta_launch<launch_t>(k, num_ctas, context);

  // Recursively call reduce until there's just one scalar.
  if (num_ctas > 1) {
    reduce_baseline(partials_data, num_ctas, reduction,
		    context);
  }
}

template <typename input_it, typename output_it>
void reduce_interleave(input_it input, int count, output_it reduction,
                context_t& context) {
  typedef launch_params_t<256, 1> launch_t;

  typedef typename std::iterator_traits<input_it>::value_type type_t;

  int num_ctas = launch_t::cta_dim(context).num_ctas(count);
  mem_t<type_t> partials(num_ctas, context);
  type_t* partials_data = partials.data();

  // Reduce each tile to a scalar and store to partials.
  auto k = [=] PLAS_DEVICE(int tid, int cta) {
    typedef typename launch_t::sm_ptx params_t;
    enum {
      nt = params_t::nt,
      vt = params_t::vt,
      nv = nt * vt
    };

    __shared__ type_t shared_vals[nt];

    // Load global data into shmem
    // Note we ignore boundary check and assume that the total data size is
    // always divisible by the number of threads in a block.
    shared_vals[tid] = ldg(input + cta * nt + tid);
    __syncthreads();

    // Reduce across all threads.
    for (unsigned int s = 1; s < nt; s *= 2) {
      int idx = 2 * s * tid;
      if (idx < nt) {
        shared_vals[idx] += shared_vals[idx + s];
      }
      __syncthreads();
    }

    // Store the final reduction to the partials.
    if (!tid) {
      if (1 == num_ctas) *reduction = shared_vals[0];
      else partials_data[cta] = shared_vals[0];
    }
  };
  cta_launch<launch_t>(k, num_ctas, context);

  // Recursively call reduce until there's just one scalar.
  if (num_ctas > 1) {
    reduce_interleave(partials_data, num_ctas, reduction,
		    context);
  }
}

template <typename input_it, typename output_it>
void reduce_bank_conflict_free(input_it input, int count, output_it reduction,
                context_t& context) {
  typedef launch_params_t<256, 1> launch_t;

  typedef typename std::iterator_traits<input_it>::value_type type_t;

  int num_ctas = launch_t::cta_dim(context).num_ctas(count);
  mem_t<type_t> partials(num_ctas, context);
  type_t* partials_data = partials.data();

  // Reduce each tile to a scalar and store to partials.
  auto k = [=] PLAS_DEVICE(int tid, int cta) {
    typedef typename launch_t::sm_ptx params_t;
    enum {
      nt = params_t::nt,
      vt = params_t::vt,
      nv = nt * vt
    };

    __shared__ type_t shared_vals[nt];

    // Load global data into shmem
    // Note we ignore boundary check and assume that the total data size is
    // always divisible by the number of threads in a block.
    shared_vals[tid] = ldg(input + cta * nt + tid);
    __syncthreads();

    // Reduce across all threads.
    for (unsigned int s = nt/2; s > 0; s >>= 1) {
      if (tid < s) {
        shared_vals[tid] += shared_vals[tid + s];
      }
      __syncthreads();
    }

    // Store the final reduction to the partials.
    if (!tid) {
      if (1 == num_ctas) *reduction = shared_vals[0];
      else partials_data[cta] = shared_vals[0];
    }
  };
  cta_launch<launch_t>(k, num_ctas, context);

  // Recursively call reduce until there's just one scalar.
  if (num_ctas > 1) {
    reduce_bank_conflict_free(partials_data, num_ctas, reduction,
		    context);
  }
}

template <typename input_it, typename output_it>
void reduce_idle_thread(input_it input, int count, output_it reduction,
                context_t& context) {
  typedef launch_params_t<256, 1> launch_t;

  typedef typename std::iterator_traits<input_it>::value_type type_t;

  int num_ctas = launch_t::cta_dim(context).num_ctas(count/2);
  mem_t<type_t> partials(num_ctas, context);
  type_t* partials_data = partials.data();

  // Reduce each tile to a scalar and store to partials.
  auto k = [=] PLAS_DEVICE(int tid, int cta) {
    typedef typename launch_t::sm_ptx params_t;
    enum {
      nt = params_t::nt,
      vt = params_t::vt,
      nv = nt * vt
    };

    __shared__ type_t shared_vals[nt];

    // Load global data into shmem
    // Note we ignore boundary check and assume that the total data size is
    // always divisible by the number of threads in a block.
    shared_vals[tid] = ldg(input + nt * 2 * cta + tid) + ldg(input + nt * (2 * cta + 1) + tid);
    __syncthreads();

    // Reduce across all threads.
    for (unsigned int s = nt/2; s > 0; s >>= 1) {
      if (tid < s) {
        shared_vals[tid] += shared_vals[tid + s];
      }
      __syncthreads();
    }

    // Store the final reduction to the partials.
    if (!tid) {
      if (1 == num_ctas) *reduction = shared_vals[0];
      else partials_data[cta] = shared_vals[0];
    }
  };
  cta_launch<launch_t>(k, num_ctas, context);

  // Recursively call reduce until there's just one scalar.
  if (num_ctas > 1) {
    reduce_idle_thread(partials_data, num_ctas, reduction,
		    context);
  }
}

template <typename input_it, typename output_it>
void reduce_warp_1(input_it input, int count, output_it reduction,
                context_t& context) {
  typedef launch_params_t<256, 1> launch_t;

  typedef typename std::iterator_traits<input_it>::value_type type_t;

  int num_ctas = launch_t::cta_dim(context).num_ctas(count/2);
  mem_t<type_t> partials(num_ctas, context);
  type_t* partials_data = partials.data();

  // Reduce each tile to a scalar and store to partials.
  auto k = [=] PLAS_DEVICE(int tid, int cta) {
    typedef typename launch_t::sm_ptx params_t;
    enum {
      nt = params_t::nt,
      vt = params_t::vt,
      nv = nt * vt
    };

    __shared__ type_t shared_vals[nt];

    // Load global data into shmem
    // Note we ignore boundary check and assume that the total data size is
    // always divisible by the number of threads in a block.
    shared_vals[tid] = ldg(input + nt * 2 * cta + tid) + ldg(input + nt * (2 * cta + 1) + tid);
    __syncthreads();

    // Reduce across all threads.
    for (unsigned int s = nt/2; s > 32; s >>= 1) {
      if (tid < s) {
        shared_vals[tid] += shared_vals[tid + s];
      }
      __syncthreads();
    }
    if (tid < 32) {
      warp_reduce_naive(shared_vals, tid);
    }

    // Store the final reduction to the partials.
    if (!tid) {
      if (1 == num_ctas) *reduction = shared_vals[0];
      else partials_data[cta] = shared_vals[0];
    }
  };
  cta_launch<launch_t>(k, num_ctas, context);

  // Recursively call reduce until there's just one scalar.
  if (num_ctas > 1) {
    reduce_warp_1(partials_data, num_ctas, reduction,
		    context);
  }
}

template <typename input_it, typename output_it>
void reduce_warp_multi_add(input_it input, int count, output_it reduction,
                context_t& context) {
  typedef launch_params_t<256, 8> launch_t;

  typedef typename std::iterator_traits<input_it>::value_type type_t;

  int num_ctas = launch_t::cta_dim(context).num_ctas(count);
  mem_t<type_t> partials(num_ctas, context);
  type_t* partials_data = partials.data();

  // Reduce each tile to a scalar and store to partials.
  auto k = [=] PLAS_DEVICE(int tid, int cta) {
    typedef typename launch_t::sm_ptx params_t;
    enum {
      nt = params_t::nt,
      vt = params_t::vt,
      nv = nt * vt
    };

    __shared__ type_t shared_vals[nt];

    // Load global data into shmem
    // Note we ignore boundary check and assume that the total data size is
    // always divisible by the number of threads in a block.
    shared_vals[tid] = 0;
    #pragma unroll
    for(int iter=0; iter<vt; iter++){
        shared_vals[tid] += ldg(input + cta * nv + tid + iter * nt);
    }
    __syncthreads();

    // Reduce across all threads.
    for (unsigned int s = nt/2; s > 32; s >>= 1) {
      if (tid < s) {
        shared_vals[tid] += shared_vals[tid + s];
      }
      __syncthreads();
    }
    if (tid < 32) {
      warp_reduce_naive(shared_vals, tid);
    }

    // Store the final reduction to the partials.
    if (!tid) {
      if (1 == num_ctas) *reduction = shared_vals[0];
      else partials_data[cta] = shared_vals[0];
    }
  };
  cta_launch<launch_t>(k, num_ctas, context);

  // Recursively call reduce until there's just one scalar.
  if (num_ctas > 1) {
    reduce_warp_multi_add(partials_data, num_ctas, reduction,
		    context);
  }
}

template <typename input_it, typename output_it>
void reduce_final(input_it input, int count, output_it reduction,
		context_t& context) {

  typedef launch_params_t<512, 4> launch_t;

  typedef typename std::iterator_traits<input_it>::value_type type_t;

  int num_ctas = launch_t::cta_dim(context).num_ctas(count);
  mem_t<type_t> partials(num_ctas, context);
  type_t* partials_data = partials.data();

  // Reduce each tile to a scalar and store to partials.
  auto k = [=] PLAS_DEVICE(int tid, int cta) {
    typedef typename launch_t::sm_ptx params_t;
    enum {
      nt = params_t::nt,
      vt = params_t::vt,
      nv = nt * vt
    };
    typedef cta_reduce<nt, type_t> reduce_t;

    __shared__ typename reduce_t::storage_t shared_reduce;

    // Load the tile's data into register.
    range_t tile = get_tile(cta, nv, count);
    array_t<type_t, vt> x =
      mem_to_reg_strided<nt, vt>(input + tile.begin, tid, tile.count());

    // Reduce the thread's vt values into a scalar.
    type_t scalar;
    strided_iterate<nt, vt>([&](int i, int j) {
			scalar = i ? (scalar + x[i]) : x[0];
		      },
		      tid, tile.count());

    // Reduce across all threads.
    scalar = reduce_t().reduce(tid, scalar, shared_reduce,
		min(tile.count(), (int)nt));

    // Store the final reduction to the partials.
    if (!tid) {
      if (1 == num_ctas) *reduction = scalar;
      else partials_data[cta] = scalar;
    }
  };
  cta_launch<launch_t>(k, num_ctas, context);

  // Recursively call reduce until there's just one scalar.
  if (num_ctas > 1) {
    reduce_final(partials_data, num_ctas, reduction, context);
  }
}

}  // namespace gpu
}  // namespace plas

#endif
