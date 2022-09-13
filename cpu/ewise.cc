#include <immintrin.h>
#include "cc/plas/cpu/api.h"

namespace plas {
namespace cpu {

#if defined (__AVX__) && defined (__FMA__)
void ewise_fma_kernel_32_avx(const float* x, const float* y, float* const out, int32 dim) {
  int32 register i = 0;

  while (i < dim) {
    __m256 xx0 = _mm256_loadu_ps(x + i);
    __m256 xx1 = _mm256_loadu_ps(x + i + 8);
    __m256 xx2 = _mm256_loadu_ps(x + i + 16);
    __m256 xx3 = _mm256_loadu_ps(x + i + 24);

    __m256 yy0 = _mm256_loadu_ps(y + i);
    __m256 yy1 = _mm256_loadu_ps(y + i + 8);
    __m256 yy2 = _mm256_loadu_ps(y + i + 16);
    __m256 yy3 = _mm256_loadu_ps(y + i + 24);

    __m256 zz0 = _mm256_loadu_ps(out + i);
    __m256 zz1 = _mm256_loadu_ps(out + i + 8);
    __m256 zz2 = _mm256_loadu_ps(out + i + 16);
    __m256 zz3 = _mm256_loadu_ps(out + i + 24);

    zz0 = _mm256_fmadd_ps(xx0, yy0, zz0);
    zz1 = _mm256_fmadd_ps(xx1, yy1, zz1);
    zz2 = _mm256_fmadd_ps(xx2, yy2, zz2);
    zz3 = _mm256_fmadd_ps(xx3, yy3, zz3);

    _mm256_storeu_ps(out + i, zz0);
    _mm256_storeu_ps(out + i + 8, zz1);
    _mm256_storeu_ps(out + i + 16, zz2);
    _mm256_storeu_ps(out + i + 24, zz3);

    i += 32;
  }
}
#endif

void ewise_fma_kernel_32_naive(const float* x, const float* y, float* const out, int32 dim) {
  for (int32 i = 0; i < dim; ++i) {
    out[i] += x[i] * y[i];
  }
}

void ewise_fma(const float* x, const float* y, float* const out, int32 dim) {
  int32 n = dim & (-32);
  if (n) {
#if defined (__AVX__) && defined (__FMA__)
    ewise_fma_kernel_32_avx(x, y, out, n);
#else
    ewise_fma_kernel_32_naive(x, y, out, n);
#endif
  }
  for (int32 i = n; i < dim; ++i) {
    out[i] += x[i] * y[i];
  }
}

#ifdef __AVX__
/**
 * @brief y += x
 *
 * @param x input, vector
 * @param y output, vector
 * @param dim input, vector length
 */
void ewise_add_kernel_32_avx(const float* x, float* const y, int32 dim) {
  int32 register i = 0;
  while (i < dim) {
    __m256 xx0 = _mm256_loadu_ps(x + i);
    __m256 xx1 = _mm256_loadu_ps(x + i + 8);
    __m256 xx2 = _mm256_loadu_ps(x + i + 16);
    __m256 xx3 = _mm256_loadu_ps(x + i + 24);

    __m256 yy0 = _mm256_loadu_ps(y + i);
    __m256 yy1 = _mm256_loadu_ps(y + i + 8);
    __m256 yy2 = _mm256_loadu_ps(y + i + 16);
    __m256 yy3 = _mm256_loadu_ps(y + i + 24);

    yy0 = _mm256_add_ps(xx0, yy0);
    yy1 = _mm256_add_ps(xx1, yy1);
    yy2 = _mm256_add_ps(xx2, yy2);
    yy3 = _mm256_add_ps(xx3, yy3);

    _mm256_storeu_ps(y + i, yy0);
    _mm256_storeu_ps(y + i + 8, yy1);
    _mm256_storeu_ps(y + i + 16, yy2);
    _mm256_storeu_ps(y + i + 24, yy3);

    i += 32;
  }
}
#endif

void ewise_add_kernel_32_naive(const float* x, float* const y, int32 dim) {
  for (int i = 0; i < dim; ++i) {
    y[i] += x[i];
  }
}

void ewise_add(const float* x, float* const out, int32 dim) {
  int32 n = dim & (-32);
  if (n) {
#ifdef __AVX__
    ewise_add_kernel_32_avx(x, out, n);
#else
    ewise_add_kernel_32_naive(x, out, n);
#endif
  }
  for (int32 i = n; i < dim; ++i) {
    out[i] += x[i];
  }
}

#ifdef __AVX__
void ewise_copy_kernel_32_avx(float* const dst, const float* src, int32 dim) {
  int32 register i = 0;
  while (i < dim) {
    __m256 xx0 = _mm256_loadu_ps(src + i);
    __m256 xx1 = _mm256_loadu_ps(src + i + 8);
    __m256 xx2 = _mm256_loadu_ps(src + i + 16);
    __m256 xx3 = _mm256_loadu_ps(src + i + 24);

    _mm256_storeu_ps(dst + i, xx0);
    _mm256_storeu_ps(dst + i + 8, xx1);
    _mm256_storeu_ps(dst + i + 16, xx2);
    _mm256_storeu_ps(dst + i + 24, xx3);

    i += 32;
  }
}
#endif

void ewise_copy_kernel_32_naive(float* const dst, const float* src, int32 dim) {
  for (int32 i = 0; i < dim; ++i) {
    dst[i] = src[i];
  }
}

void ewise_copy(float* const dst, const float* src, int32 dim) {
  int32 n = dim & (-32);
  if (n) {
#ifdef __AVX__
    ewise_copy_kernel_32_avx(dst, src, n);
#else
    ewise_copy_kernel_32_naive(dst, src, n);
#endif
  }
  for (int32 i = n; i < dim; ++i) {
    dst[i] = src[i];
  }
}

}  // namespace cpu
}  // namespace plas
