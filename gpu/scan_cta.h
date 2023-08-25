#ifndef PLAS_GPU_SCAN_CTA_H_
#define PLAS_GPU_SCAN_CTA_H_

#include "intrinsics.h"
#include "udt.h"

namespace plas {
namespace gpu {

enum scan_type_t {
  scan_type_exc,
  scan_type_inc
};

template <typename type_t, int vt = 0, bool is_array = (vt > 0)>
struct scan_result_t {
  type_t scan;
  type_t reduction;
};

template <typename type_t, int vt>
struct scan_result_t<type_t, vt, true> {
  array_t<type_t, vt> scan;
  type_t reduction;
};

template <int nt, typename type_t>
struct cta_scan_t {
  enum {
    num_warps = nt / warp_size,
    capacity = nt + num_warps
  };
  union storage_t {
    type_t data[2 * nt];  // NOLINT
    struct {
      type_t threads[nt], warps[num_warps];
    };
  };

  template <typename op_t = plus_t<type_t> >
  PLAS_DEVICE scan_result_t<type_t> scan(int tid, type_t x, storage_t& storage,
                                         int count = nt, op_t op = op_t(),
                                         type_t init = type_t(),
                                         scan_type_t type =
                                             scan_type_exc) const {
    int warp = tid / warp_size;

    // warp san using shfl_add.
    type_t warp_scan = x;
    iterate<s_log2(warp_size)>([&](int pass) {
      warp_scan = shfl_up_op(warp_scan, 1 << pass, op, (int)warp_size);
    });

    // store the intra-warp scans.
    storage.threads[tid] = warp_scan;

    // store the reduction (last element) of each warp into storage.
    if (min(warp_size * (warp + 1), count) - 1 == tid)
      storage.warps[warp] = warp_scan;
    __syncthreads();

    // Scan the warp reductions.
    if (tid < num_warps) {
      type_t cta_scan = storage.warps[tid];
      iterate<s_log2(num_warps)>([&](int pass) {
        cta_scan = shfl_up_op(cta_scan, 1 << pass, op, (int)num_warps);
      });
      storage.warps[tid] = cta_scan;
    }
    __syncthreads();

    type_t scan = warp_scan;
    if (scan_type_exc == type) {
      scan = tid ? storage.threads[tid - 1] : init;
      warp = (tid - 1) / warp_size;
    }
    if (warp > 0) scan = op(scan, storage.warps[warp - 1]);

    type_t reduction = storage.warps[div_up(count, warp_size) - 1];

    scan_result_t<type_t> result{tid < count ? scan : reduction, reduction};
    __syncthreads();

    return result;
  }

  // CTA vectorized scan. Accepts multiple values per thread and adds in
  // optional global carry-in.

  template <int vt, typename op_t = plus_t<type_t> >
  PLAS_DEVICE scan_result_t<type_t, vt> scan(
      int tid, array_t<type_t, vt> x, storage_t& storage,
      type_t carry_in = type_t(), bool use_carry_in = false, int count = nt,
      op_t op = op_t(), type_t init = type_t(),
      scan_type_t type = scan_type_exc) const {
    // Start with an inclusive scan of the in-range elements.
    if (count >= nt * vt) {
      iterate<vt>([&](int i) { x[i] = i ? op(x[i], x[i - 1]) : x[i]; });
    } else {
      iterate<vt>([&](int i) {
        int index = vt * tid + i;
        x[i] = i ? ((index < count) ? op(x[i], x[i - 1]) : x[i - 1])
                 : (x[i] = (index < count) ? x[i] : init);
      });
    }

    // Scan the thread-local reductions for a carry-in for each thread.
    scan_result_t<type_t> result = scan(
        tid, x[vt - 1], storage, div_up(count, vt), op, init, scan_type_exc);

    // Perform the scan downsweep and add both the global carry-in and the
    // thread carry-in to the values.
    if (use_carry_in) {
      result.reduction = op(carry_in, result.reduction);
      result.scan = tid ? op(carry_in, result.scan) : carry_in;
    } else {
      use_carry_in = tid > 0;
    }

    array_t<type_t, vt> y;
    iterate<vt>([&](int i) {
      if (scan_type_exc == type) {
        y[i] = i ? x[i - 1] : result.scan;
        if (use_carry_in && i > 0) y[i] = op(result.scan, y[i]);
      } else {
        y[i] = use_carry_in ? op(x[i], result.scan) : x[i];
      }
    });

    return scan_result_t<type_t, vt>{y, result.reduction};
  }
};

// Overload for scan of bools.
template <int nt>
struct cta_scan_t<nt, bool> {
  enum {
    num_warps = nt / warp_size
  };
  struct storage_t {
    int warps[num_warps];  // NOLINT
  };

  PLAS_DEVICE scan_result_t<int> scan(int tid, bool x,
                                      storage_t& storage) const {
    // Store the bit totals for each warp.
    int lane = (warp_size - 1) & tid;
    int warp = tid / warp_size;

    int bits = ballot(x);
    storage.warps[warp] = popc(bits);
    __syncthreads();

    if (tid < num_warps) {
      // Cooperative warp scan of partial reductions.
      int scan = storage.warps[tid];
      iterate<s_log2(num_warps)>([&](int i) {
        scan = shfl_up_op(scan, 1 << i, plus_t<int>(), num_warps);
      });
      storage.warps[tid] = scan;
    }
    __syncthreads();

    int scan =
        ((warp > 0) ? storage.warps[warp - 1] : 0) + popc(bfe(bits, 0, lane));
    int reduction = storage.warps[num_warps - 1];
    __syncthreads();

    return scan_result_t<int>{scan, reduction};
  }
};

}  // namespace gpu
}  // namespace plas

#endif
