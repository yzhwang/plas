#include <immintrin.h>
#include "cc/plas/cpu/api.h"
namespace plas {
namespace cpu {

#if defined (__AVX__) && defined (__FMA__)
void saxpy_kernel_32_avx(const float alpha, float const* x, float* const y, int32 dim) {
  int32 register i = 0;

  __m256 aa = _mm256_broadcast_ss(&alpha);

  // TODO(powergao) prefetch
  // TODO(powergao) compare with saxpy with different dim
  // loop over dim
  while (i < dim) {
    __m256 xx0 = _mm256_loadu_ps(x + i);
    __m256 xx1 = _mm256_loadu_ps(x + i + 8);
    __m256 xx2 = _mm256_loadu_ps(x + i + 16);
    __m256 xx3 = _mm256_loadu_ps(x + i + 24);

    __m256 yy0 = _mm256_loadu_ps(y + i);
    __m256 yy1 = _mm256_loadu_ps(y + i + 8);
    __m256 yy2 = _mm256_loadu_ps(y + i + 16);
    __m256 yy3 = _mm256_loadu_ps(y + i + 24);

    yy0 = _mm256_fmadd_ps(aa, xx0, yy0);
    yy1 = _mm256_fmadd_ps(aa, xx1, yy1);
    yy2 = _mm256_fmadd_ps(aa, xx2, yy2);
    yy3 = _mm256_fmadd_ps(aa, xx3, yy3);

    _mm256_storeu_ps(y + i, yy0);
    _mm256_storeu_ps(y + i + 8, yy1);
    _mm256_storeu_ps(y + i + 16, yy2);
    _mm256_storeu_ps(y + i + 24, yy3);
    i += 32;
  }
}
#endif

void saxpy_kernel_32_naive(const float alpha, float const* x, float* const y, int32 dim) {
  for (int32 i = 0; i < dim; ++i) {
    y[i] += alpha * x[i];
  }
}

void saxpy(const float alpha, const float* x, float* const y, int32 dim) {
  int32 n = dim & (-32);
  if (n) {
#if defined (__AVX__) && defined (__FMA__)
    saxpy_kernel_32_avx(alpha, x, y, n);
#else
    saxpy_kernel_32_naive(alpha, x, y, n);
#endif
  }
  for (int32 i = n; i < dim; ++i) {
    y[i] += alpha * x[i];
  }
}

}  // namespace cpu
}  // namespace plas
