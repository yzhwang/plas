#ifndef PLAS_GPU_MERGE_CTA_H_
#define PLAS_GPU_MERGE_CTA_H_

#include "intrinsics.h"
#include "udt.h"

namespace plas {
namespace gpu {

template <bounds_t bounds, typename int_t, typename it_t, typename T,
          typename comp_t>
PLAS_HOST_DEVICE void binary_search_it(it_t data, int& begin, int& end, T key,
                                       int shift, comp_t comp) {
  int_t scale = (1 << shift) - 1;
  int mid = (int)((begin + scale * end) >> shift);
  T key2 = data[mid];
  bool pred = (bounds_upper == bounds) ? !comp(key, key2) : comp(key2, key);
  if (pred)
    begin = mid + 1;
  else
    end = mid;
}

template <bounds_t bounds, typename int_t, typename T, typename it_t,
          typename comp_t>
PLAS_HOST_DEVICE int biased_binary_search(it_t data, int count, T key,
                                          int levels, comp_t comp) {
  int begin = 0;
  int end = count;

  if (levels >= 4 && begin < end)
    binary_search_it<bounds, int_t>(data, begin, end, key, 9, comp);
  if (levels >= 3 && begin < end)
    binary_search_it<bounds, int_t>(data, begin, end, key, 7, comp);
  if (levels >= 2 && begin < end)
    binary_search_it<bounds, int_t>(data, begin, end, key, 5, comp);
  if (levels >= 1 && begin < end)
    binary_search_it<bounds, int_t>(data, begin, end, key, 4, comp);

  while (begin < end)
    binary_search_it<bounds, int>(data, begin, end, key, 1, comp);
  return begin;
}

template <bounds_t bounds, typename T, typename it_t, typename comp_t>
PLAS_HOST_DEVICE int binary_search(it_t data, int count, T key, comp_t comp) {
  int begin = 0;
  int end = count;
  while (begin < end)
    binary_search_it<bounds, int>(data, begin, end, key, 1, comp);
  return begin;
}

template <bounds_t bounds = bounds_lower, typename a_keys_it,
          typename b_keys_it, typename int_t, typename comp_t>
PLAS_HOST_DEVICE int_t
merge_path(a_keys_it a_keys, int_t a_count, b_keys_it b_keys, int_t b_count,
           int_t diag, comp_t comp) {
  typedef typename std::iterator_traits<a_keys_it>::value_type type_t;
  int_t begin = max((int_t)0, diag - b_count);
  int_t end = min(diag, a_count);

  while (begin < end) {
    int_t mid = (begin + end) / 2;
    type_t a_key = a_keys[mid];
    type_t b_key = b_keys[diag - 1 - mid];
    bool pred =
        (bounds_upper == bounds) ? comp(a_key, b_key) : !comp(b_key, a_key);

    if (pred)
      begin = mid + 1;
    else
      end = mid;
  }
  return begin;
}

template <bounds_t bounds, typename keys_it, typename comp_t>
PLAS_HOST_DEVICE int merge_path(keys_it keys, merge_range_t range, int diag,
                                comp_t comp) {
  return merge_path<bounds>(keys + range.a_begin, range.a_count(),
                            keys + range.b_begin, range.b_count(), diag, comp);
}

template <bounds_t bounds, bool range_check, typename type_t, typename comp_t>
PLAS_HOST_DEVICE bool merge_predicate(type_t a_key, type_t b_key,
                                      merge_range_t range, comp_t comp) {
  bool p;
  if (range_check && !range.a_valid())
    p = false;
  else if (range_check && !range.b_valid())
    p = true;
  else
    p = (bounds_upper == bounds) ? comp(a_key, b_key) : !comp(b_key, a_key);
  return p;
}

PLAS_HOST_DEVICE merge_range_t compute_merge_range(int a_count, int b_count,
                                                   int partition, int spacing,
                                                   int mp0, int mp1) {
  int diag0 = spacing * partition;
  int diag1 = min(a_count + b_count, diag0 + spacing);

  return merge_range_t{mp0, mp1, diag0 - mp0, diag1 - mp1};
}

PLAS_HOST_DEVICE void compute_intersect_ranges(int a_count, int b_count,
                                               int partition, int spacing,
                                               int mp0, int mp1, bool& extended,
                                               merge_range_t& range1,
                                               merge_range_t& range2) {
  int diag0 = spacing * partition;

  // Compute the intervals into the two source arrays.
  int a0 = 0x7fffffff & mp0;
  int a1 = 0x7fffffff & mp1;
  int b0 = diag0 - a0;
  int b1 = min(a_count + b_count, diag0 + spacing) - a1;

  // If  the most sig bit flag is set, we're dealing with a starred diagonal.
  int bit0 = (0x80000000 & mp0) ? 1 : 0;
  int bit1 = (0x80000000 & mp1) ? 1 : 0;
  b0 += bit0;
  b1 += bit1;

  int a_count2 = a1 - a0;
  int b_count2 = b1 - b0;
  extended = (a1 - a_count) && (b1 < b_count);
  int b_start = a_count2 + (int)extended;

  range1.a_begin = a0;
  range1.a_end = a0 + a_count2 + (int)extended;
  range1.b_begin = b0;
  range1.b_end = b0 + b_count2 + (int)extended;

  range2.a_begin = 0;
  range2.a_end = a_count2;
  range2.b_begin = b_start;
  range2.b_end = b_start + b_count2;

  return;
}

template <bool duplicates, typename int_t, typename a_keys_it,
          typename b_keys_it, typename comp_t>
PLAS_HOST_DEVICE int2
balanced_path(a_keys_it a_keys, int a_count, b_keys_it b_keys, int b_count,
              int diag, int levels, comp_t comp) {
  typedef typename std::iterator_traits<a_keys_it>::value_type T;
  int p =
      merge_path<bounds_lower>(a_keys, a_count, b_keys, b_count, diag, comp);
  int a_index = p;
  int b_index = diag - p;

  bool star = false;
  if (b_index < b_count) {
    if (duplicates) {
      T x = b_keys[b_index];

      // Search for the beginning of the duplicate run in both A and B.
      int a_start = biased_binary_search<bounds_lower, int_t>(a_keys, a_index,
                                                              x, levels, comp);
      int b_start = biased_binary_search<bounds_lower, int_t>(b_keys, b_index,
                                                              x, levels, comp);

      int a_run = a_index - a_start;
      int b_run = b_index - b_start;
      int x_count = a_run + b_run;

      // Attempt to advance b and regress a.
      int b_adv = max(x_count >> 1, b_run);
      int b_end = min(b_count, b_start + b_adv + 1);
      int b_run_end = binary_search<bounds_upper>(b_keys + b_index,
                                                  b_end - b_index, x, comp) +
                      b_index;
      b_run = b_run_end - b_start;

      b_adv = min(b_adv, b_run);
      int a_adv = x_count - b_adv;

      bool roundup = (a_adv == b_adv + 1) && (b_adv < b_run);
      a_index = a_start + a_adv;

      if (roundup) star = true;
    } else {
      if (a_index && a_count) {
        T a_key = a_keys[a_index - 1];
        T b_key = b_keys[b_index];

        // last consumed element in A is the same as the next element in B.
        // we have a starred partition.
        if (!comp(a_key, b_key)) star = true;
      }
    }
  }
  return make_int2(a_index, star);
}

// Specialization that emits just one LD instruction. Can only reliably used
// with raw pointer types. Fixed not to use pointer arithmetic so that
// we don't get undefined behaviors with unaligned types.
template <int nt, int vt, typename type_t>
PLAS_DEVICE array_t<type_t, vt> load_two_streams_reg(const type_t* a,
                                                     int a_count,
                                                     const type_t* b,
                                                     int b_count, int tid) {
  b -= a_count;
  array_t<type_t, vt> x;
  strided_iterate<nt, vt>([&](int i, int index) {
                            const type_t* p = (index >= a_count) ? b : a;
                            x[i] = p[index];
                          },
                          tid, a_count + b_count);

  return x;
}

template <int nt, int vt, typename type_t, typename a_it, typename b_it>
PLAS_DEVICE
enable_if_t<!(std::is_pointer<a_it>::value && std::is_pointer<b_it>::value),
            array_t<type_t, vt> >
load_two_streams_reg(a_it a, int a_count, b_it b, int b_count, int tid) {
  b -= a_count;
  array_t<type_t, vt> x;
  strided_iterate<nt, vt>([&](int i, int index) {
                            x[i] = (index < a_count) ? a[index] : b[index];
                          },
                          tid, a_count + b_count);
  return x;
}

template <int nt, int vt, typename a_it, typename b_it, typename type_t,
          int shared_size>
PLAS_DEVICE void load_two_streams_shared(a_it a, int a_count, b_it b,
                                         int b_count, int tid,
                                         type_t (&shared)[shared_size],
                                         bool sync = true) {
  // Load into register then make an unconditional strided store into memory.
  array_t<type_t, vt> x =
      load_two_streams_reg<nt, vt, type_t>(a, a_count, b, b_count, tid);
  reg_to_shared_strided<nt>(x, tid, shared, sync);
}

// This function must be able to dereference keys[a_begin] and keys[b_begin],
// no matter the indices for each. The caller should allocate at least
// nt * vt + 1 elements for
template <bounds_t bounds, int vt, typename type_t, typename comp_t>
PLAS_DEVICE merge_pair_t<type_t, vt> serial_merge(const type_t* keys_shared,
                                                  merge_range_t range,
                                                  comp_t comp,
                                                  bool sync = true) {
  type_t a_key = keys_shared[range.a_begin];
  type_t b_key = keys_shared[range.b_begin];

  merge_pair_t<type_t, vt> merge_pair;
  iterate<vt>([&](int i) {
    bool p = merge_predicate<bounds, true>(a_key, b_key, range, comp);
    int index = p ? range.a_begin : range.b_begin;

    merge_pair.keys[i] = p ? a_key : b_key;
    merge_pair.indices[i] = index;

    type_t c_key = keys_shared[++index];
    if (p)
      a_key = c_key, range.a_begin = index;
    else
      b_key = c_key, range.b_begin = index;
  });

  if (sync) __syncthreads();
  return merge_pair;
}

template <int vt, bool range_check, typename type_t, typename comp_t>
PLAS_DEVICE int serial_intersect(const type_t* keys_shared, merge_range_t range,
                                 int end, merge_pair_t<type_t, vt>& merge_pair,
                                 int block_start_idx, comp_t comp) {
  const int min_iterations = vt / 2;
  int commit = 0;

  iterate<vt>([&](int i) {
    bool test =
        range_check
            ? ((range.a_begin + range.b_begin < end) &&
               (range.a_begin < range.a_end) && (range.b_begin < range.b_end))
            : (i < min_iterations || (range.a_begin + range.b_begin < end));
    if (test) {
      type_t a_key = keys_shared[range.a_begin];
      type_t b_key = keys_shared[range.b_begin];

      bool p_a = comp(a_key, b_key);
      bool p_b = comp(b_key, a_key);

      // The output must come from A by definition of set intersection.
      merge_pair.keys[i] = a_key;
      merge_pair.indices[i] = range.a_begin + block_start_idx;
      if (!p_b) ++range.a_begin;
      if (!p_a) ++range.b_begin;
      if (p_a == p_b) {
        commit |= 1 << i;
      }
    }
  });

  return commit;
}

// Load arrays a and b from global memory and merge into register.
template <bounds_t bounds, int nt, int vt, typename a_it, typename b_it,
          typename type_t, typename comp_t, int shared_size>
PLAS_DEVICE merge_pair_t<type_t, vt> cta_merge_from_mem(
    a_it a, b_it b, merge_range_t range_mem, int tid, comp_t comp,
    type_t (&keys_shared)[shared_size]) {
  static_assert(shared_size >= nt * vt + 1,
                "cta_merge_from_mem requires temporary storage of at "
                "least nt * vt + 1 items");

  // Load the data into shared memory.
  load_two_streams_shared<nt, vt>(a + range_mem.a_begin, range_mem.a_count(),
                                  b + range_mem.b_begin, range_mem.b_count(),
                                  tid, keys_shared, true);

  // Run a merge path to find the start of the serial merge for each thread.
  merge_range_t range_local = range_mem.to_local();
  int diag = vt * tid;
  int mp = merge_path<bounds>(keys_shared, range_local, diag, comp);

  // Compute the ranges of the sources in shared memory. The end iterators
  // of the range are inaccurate, but still facilitate exact merging, because
  // only vt elements will be merged.
  merge_pair_t<type_t, vt> merged = serial_merge<bounds, vt>(
      keys_shared, range_local.partition(mp, diag), comp);

  return merged;
}

// Load arrays a and b from global memory and compute set availability.
// TODO(slashwang): could expand to all set operations.
template <int nt, int vt, bool duplicates, typename a_it, typename b_it,
          typename type_t, typename comp_t, int shared_size>
PLAS_DEVICE int cta_intersect_from_mem(a_it a, b_it b, merge_range_t range1,
                                       merge_range_t range2, bool& extended,
                                       merge_pair_t<type_t, vt>& intersected,
                                       int tid, int mp0, comp_t comp,
                                       type_t (&keys_shared)[shared_size]) {
  static_assert(shared_size >= nt * vt + vt,
                "cta_intersect_from_mem requires temporary storage of at "
                "least nt * vt + vt items");

  // Load the data into shared memory.
  load_two_streams_shared<nt, vt + 1>(a + range1.a_begin, range1.a_count(),
                                      b + range1.b_begin, range1.b_count(), tid,
                                      keys_shared, true);
  int count = extended ? range1.a_count() + range1.b_count() - 2
                       : range1.a_count() + range1.b_count();
  int bit0 = (0x80000000 & mp0) ? 1 : 0;
  int diag = min(vt * tid - bit0, count);

  int2 bp = balanced_path<duplicates, int>(
      keys_shared + range2.a_begin, range2.a_count(),
      keys_shared + range2.b_begin, range2.b_count(), diag, 2, comp);

  int a0tid = bp.x;
  int b0tid = vt * tid + bp.y - bp.x - bit0;
  merge_range_t intersect_range = {a0tid,                  range2.a_end,
                                   range2.b_begin + b0tid, range2.b_end};

  int commit;
  int end = intersect_range.a_begin + intersect_range.b_begin + vt - bp.y;
  if (extended) {
    commit = serial_intersect<vt, false>(keys_shared, intersect_range, end,
                                         intersected, range1.a_begin, comp);
  } else {
    end = min(end, intersect_range.a_end + intersect_range.b_end);
    commit = serial_intersect<vt, true>(keys_shared, intersect_range, end,
                                        intersected, range1.a_begin, comp);
  }
  return commit;
}

}  // namespace gpu
}  // namespace plas

#endif
