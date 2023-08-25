#ifndef PLAS_GPU_INTRINSICS_H_
#define PLAS_GPU_INTRINSICS_H_

#include "meta.h"

namespace plas {
namespace gpu {

#ifndef MEMBERMASK
#define MEMBERMASK 0xffffffff
#endif

#if __CUDACC_VER_MAJOR__ >= 9 && defined(__CUDA_ARCH__) && \
    !defined(USE_SHFL_SYNC)
#define USE_SHFL_SYNC
#endif

// ballot
PLAS_DEVICE unsigned ballot(int predicate, unsigned mask = MEMBERMASK) {
  unsigned y = 0;
#ifdef USE_SHFL_SYNC
  y = __ballot_sync(mask, predicate);
#else
  y = __ballot(predicate);
#endif
  return y;
}

// count number of bits in a register.
PLAS_DEVICE int popc(unsigned x) { return __popc(x); }

PLAS_HOST_DEVICE unsigned bfe(unsigned x, unsigned bit, unsigned num_bits) {
  unsigned result;
  asm("bfe.u32 %0, %1, %2, %3;"
      : "=r"(result)
      : "r"(x), "r"(bit), "r"(num_bits));
  return result;
}

////////////////////////////////////////////////////////////////////////////////
// Wrappers around PTX shfl_up and shfl_down.

template <typename type_t>
PLAS_DEVICE type_t shfl_up(type_t x, int offset, int width = warp_size) {
  enum {
    num_words = div_up(sizeof(type_t), sizeof(int))
  };
  union {
    int x[num_words];  // NOLINT
    type_t t;
  } u;
  u.t = x;

  iterate<num_words>([&](int i) {
#ifdef USE_SHFL_SYNC
    if (i < width) {
      unsigned mask = __activemask();
      u.x[i] = __shfl_up_sync(mask, u.x[i], offset);
    }
#else
    u.x[i] = __shfl_up(u.x[i], offset, width);
    #endif
  });
  return u.t;
}

template <typename type_t>
PLAS_DEVICE type_t shfl_down(type_t x, int offset, int width = warp_size) {
  enum {
    num_words = div_up(sizeof(type_t), sizeof(int))
  };
  union {
    int x[num_words];  // NOLINT
    type_t t;
  } u;
  u.t = x;

  iterate<num_words>([&](int i) {
#ifdef USE_SHFL_SYNC
    if (i < width) {
      unsigned mask = __activemask();
      u.x[i] = __shfl_down_sync(mask, u.x[i], offset);
    }
#else
    u.x[i] = __shfl_down(u.x[i], offset, width);
    #endif
  });
  return u.t;
}

#ifdef USE_SHFL_SYNC
#define SHFL_OP_MACRO(dir, is_up, ptx_type, r, c_type, ptx_op, c_op)  \
  PLAS_DEVICE inline c_type shfl_##dir##_op(                          \
      c_type x, int offset, c_op<c_type> op, int width = warp_size) { \
    c_type result = x;                                                \
    int mask = (warp_size - width) << 8 | (is_up ? 0 : (width - 1));  \
    int lane = threadIdx.x & (warp_size - 1);                         \
    if (lane < width) {                                               \
      unsigned threadmask = __activemask();                           \
      asm("{.reg ." #ptx_type                                         \
          " r0;"                                                      \
          ".reg .pred p;"                                             \
          "shfl.sync." #dir                                           \
          ".b32 r0|p, %1, %2, %3, %4;"                                \
          "@p " #ptx_op "." #ptx_type                                 \
          " r0, r0, %5;"                                              \
          "mov." #ptx_type " %0, r0; }"                               \
          : "=" #r(result)                                            \
          : #r(x), "r"(offset), "r"(mask), "r"(threadmask), #r(x));   \
    }                                                                 \
    return result;                                                    \
  }
#else
#define SHFL_OP_MACRO(dir, is_up, ptx_type, r, c_type, ptx_op, c_op)  \
  PLAS_DEVICE inline c_type shfl_##dir##_op(                          \
      c_type x, int offset, c_op<c_type> op, int width = warp_size) { \
    c_type result = c_type();                                         \
    int mask = (warp_size - width) << 8 | (is_up ? 0 : (width - 1));  \
    asm("{.reg ." #ptx_type                                           \
        " r0;"                                                        \
        ".reg .pred p;"                                               \
        "shfl." #dir                                                  \
        ".b32 r0|p, %1, %2, %3;"                                      \
        "@p " #ptx_op "." #ptx_type                                   \
        " r0, r0, %4;"                                                \
        "mov." #ptx_type " %0, r0; }"                                 \
        : "=" #r(result)                                              \
        : #r(x), "r"(offset), "r"(mask), #r(x));                      \
    return result;                                                    \
  }
#endif

SHFL_OP_MACRO(up, true, s32, r, int, add, plus_t)
SHFL_OP_MACRO(up, true, s32, r, int, max, maximum_t)
SHFL_OP_MACRO(up, true, s32, r, int, min, minimum_t)
SHFL_OP_MACRO(down, false, s32, r, int, add, plus_t)
SHFL_OP_MACRO(down, false, s32, r, int, max, maximum_t)
SHFL_OP_MACRO(down, false, s32, r, int, min, minimum_t)

SHFL_OP_MACRO(up, true, u32, r, unsigned, add, plus_t)
SHFL_OP_MACRO(up, true, u32, r, unsigned, max, maximum_t)
SHFL_OP_MACRO(up, true, u32, r, unsigned, min, minimum_t)
SHFL_OP_MACRO(down, false, u32, r, unsigned, add, plus_t)
SHFL_OP_MACRO(down, false, u32, r, unsigned, max, maximum_t)
SHFL_OP_MACRO(down, false, u32, r, unsigned, min, minimum_t)

SHFL_OP_MACRO(up, true, f32, f, float, add, plus_t)
SHFL_OP_MACRO(up, true, f32, f, float, max, maximum_t)
SHFL_OP_MACRO(up, true, f32, f, float, max, minimum_t)
SHFL_OP_MACRO(down, false, f32, f, float, add, plus_t)
SHFL_OP_MACRO(down, false, f32, f, float, max, maximum_t)
SHFL_OP_MACRO(down, false, f32, f, float, max, minimum_t)

#undef SHFL_OP_MACRO

#ifdef USE_SHFL_SYNC
#define SHFL_OP_64b_MACRO(dir, is_up, ptx_type, r, c_type, ptx_op, c_op) \
  PLAS_DEVICE inline c_type shfl_##dir##_op(                             \
      c_type x, int offset, c_op<c_type> op, int width = warp_size) {    \
    c_type result = x;                                                   \
    int mask = (warp_size - width) << 8 | (is_up ? 0 : (width - 1));     \
    int lane = threadIdx.x & (warp_size - 1);                            \
    if (lane < width) {                                                  \
      unsigned threadmask = __activemask();                              \
      asm("{.reg ." #ptx_type                                            \
          " r0;"                                                         \
          ".reg .u32 lo;"                                                \
          ".reg .u32 hi;"                                                \
          ".reg .pred p;"                                                \
          "mov.b64 {lo, hi}, %1;"                                        \
          "shfl.sync." #dir                                              \
          ".b32 lo|p, lo, %2, %3, %4;"                                   \
          "shfl.sync." #dir                                              \
          ".b32 hi  , hi, %2, %3, %4;"                                   \
          "mov.b64 r0, {lo, hi};"                                        \
          "@p " #ptx_op "." #ptx_type                                    \
          " r0, r0, %5;"                                                 \
          "mov." #ptx_type " %0, r0; }"                                  \
          : "=" #r(result)                                               \
          : #r(x), "r"(offset), "r"(mask), "r"(threadmask), #r(x));      \
    }                                                                    \
    return result;                                                       \
  }
#else
#define SHFL_OP_64b_MACRO(dir, is_up, ptx_type, r, c_type, ptx_op, c_op) \
  PLAS_DEVICE inline c_type shfl_##dir##_op(                             \
      c_type x, int offset, c_op<c_type> op, int width = warp_size) {    \
    c_type result = c_type();                                            \
    int mask = (warp_size - width) << 8 | (is_up ? 0 : (width - 1));     \
    asm("{.reg ." #ptx_type                                              \
        " r0;"                                                           \
        ".reg .u32 lo;"                                                  \
        ".reg .u32 hi;"                                                  \
        ".reg .pred p;"                                                  \
        "mov.b64 {lo, hi}, %1;"                                          \
        "shfl." #dir                                                     \
        ".b32 lo|p, lo, %2, %3;"                                         \
        "shfl." #dir                                                     \
        ".b32 hi  , hi, %2, %3;"                                         \
        "mov.b64 r0, {lo, hi};"                                          \
        "@p " #ptx_op "." #ptx_type                                      \
        " r0, r0, %4;"                                                   \
        "mov." #ptx_type " %0, r0; }"                                    \
        : "=" #r(result)                                                 \
        : #r(x), "r"(offset), "r"(mask), #r(x));                         \
    return result;                                                       \
  }
#endif

SHFL_OP_64b_MACRO(up, true, s64, l, int64_t, add, plus_t)
SHFL_OP_64b_MACRO(up, true, s64, l, int64_t, max, maximum_t)
SHFL_OP_64b_MACRO(up, true, s64, l, int64_t, min, minimum_t)
SHFL_OP_64b_MACRO(down, false, s64, l, int64_t, add, plus_t)
SHFL_OP_64b_MACRO(down, false, s64, l, int64_t, max, maximum_t)
SHFL_OP_64b_MACRO(down, false, s64, l, int64_t, min, minimum_t)

SHFL_OP_64b_MACRO(up, true, u64, l, uint64_t, add, plus_t)
SHFL_OP_64b_MACRO(up, true, u64, l, uint64_t, max, maximum_t)
SHFL_OP_64b_MACRO(up, true, u64, l, uint64_t, min, minimum_t)
SHFL_OP_64b_MACRO(down, false, u64, l, uint64_t, add, plus_t)
SHFL_OP_64b_MACRO(down, false, u64, l, uint64_t, max, maximum_t)
SHFL_OP_64b_MACRO(down, false, u64, l, uint64_t, min, minimum_t)

SHFL_OP_64b_MACRO(up, true, f64, d, double, add, plus_t)
SHFL_OP_64b_MACRO(up, true, f64, d, double, max, maximum_t)
SHFL_OP_64b_MACRO(up, true, f64, d, double, min, minimum_t)
SHFL_OP_64b_MACRO(down, false, f64, d, double, add, plus_t)
SHFL_OP_64b_MACRO(down, false, f64, d, double, max, maximum_t)
SHFL_OP_64b_MACRO(down, false, f64, d, double, min, minimum_t)

#undef SHFL_OP_64b_MACRO

}  // namespace gpu
}  // namespace plas

#endif
