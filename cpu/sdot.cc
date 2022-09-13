#include "cc/plas/cpu/api.h"

namespace plas {
namespace cpu {

#if defined(__AVX__) && defined(__SSE__)
void sdot_kernel_32_avx(const float* x, const float* y, int32 dim, float* const res) {
  int register i = 0;
  while (i < dim) {
    __m256 xx0 = _mm256_loadu_ps(x + i);
    __m256 xx1 = _mm256_loadu_ps(x + i + 8);
    __m256 xx2 = _mm256_loadu_ps(x + i + 16);
    __m256 xx3 = _mm256_loadu_ps(x + i + 24);

    __m256 yy0 = _mm256_loadu_ps(y + i);
    __m256 yy1 = _mm256_loadu_ps(y + i + 8);
    __m256 yy2 = _mm256_loadu_ps(y + i + 16);
    __m256 yy3 = _mm256_loadu_ps(y + i + 24);

    __m256 dst0 = _mm256_mul_ps(xx0, yy0);
    __m256 dst1 = _mm256_mul_ps(xx1, yy1);
    __m256 dst2 = _mm256_mul_ps(xx2, yy2);
    __m256 dst3 = _mm256_mul_ps(xx3, yy3);

    dst0 = _mm256_add_ps(dst0, dst1);
    dst1 = _mm256_add_ps(dst2, dst3);

    dst0 = _mm256_add_ps(dst0, dst1);

    __m128 dst0_hi = _mm256_extractf128_ps(dst0, 1);
    __m128 dst0_lo = _mm256_castps256_ps128(dst0);

    __m128 sum = _mm_add_ps(dst0_hi, dst0_lo);

    dst0_lo = sum;
    dst0_hi = _mm_movehl_ps(sum, sum);

    sum = _mm_add_ps(dst0_hi, dst0_lo);

    dst0_lo = sum;
    dst0_hi = _mm_shuffle_ps(sum, sum, 0x1);

    sum = _mm_add_ss(dst0_hi, dst0_lo);
    *res += _mm_cvtss_f32(sum);
    i += 32;
  }
}
#endif

void sdot_kernel_32_naive(const float* x, const float* y, int32 dim, float* const res) {
  for (int32 i = 0; i < dim; ++i) {
    *res += x[i] * y[i];
  }
}

void sdot(const float* x, const float* y, int32 dim, float* const res) {
  int32 n = dim & (-32);
  if (n) {
#if defined(__AVX__) && defined(__SSE__)
    sdot_kernel_32_avx(x, y, n, res);
#else
    sdot_kernel_32_naive(x, y, n, res);
#endif
  }
  for (int32 i = n; i < dim; ++i) {
    *res += x[i] * y[i];
  }
}

}  // namespace cpu
}  // namespace plas
