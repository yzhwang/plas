#include <immintrin.h>
#include <omp.h>
#include <tsl/sparse_map.h>
#include <unordered_map>
#include "cc/plas/cpu/api.h"

namespace plas {
namespace cpu {

void index_lookup_kernel_naive(const ID* dict, const int32 dict_st, int32 dict_en, const ID* query,
                               const int32 query_st, int32 query_en, int32* const indices,
                               int32* const indices_len) {
  const ID* a = dict;
  const ID* b = query;
  int32* const out = indices;
  int32* const out_len = indices_len;

  int32 i = dict_st;
  int32 j = query_st;
  while (i < dict_en && j < query_en) {
    if (a[i] == b[j]) {
      out[*out_len] = i;
      (*out_len)++;
      i++;
      j++;
    } else if (a[i] < b[j]) {
      i++;
    } else {
      j++;
    }
  }
}

#if defined(__AVX__) && defined(__AVX2__)
void index_lookup_kernel_4x64bit_avx2(const ID* dict, const int32 dict_len, const ID* query,
                                      const int32 query_len, int32* const indices,
                                      int32* const indices_len) {
  int veclen = sizeof(__m256i) / sizeof(uint64);
  CHECK_EQ(veclen, 4);
  int register i = 0;
  int register j = 0;
  int32* const out_len = indices_len;
  const uint64* a = dict;
  const uint64* b = query;
  int32* const out = indices;
  int a_size = dict_len - (dict_len & (veclen - 1));
  int b_size = query_len - (query_len & (veclen - 1));

  while (i < a_size && j < b_size) {
    __m256i aa = _mm256_lddqu_si256(reinterpret_cast<const __m256i*>(a + i));
    __m256i bb = _mm256_lddqu_si256(reinterpret_cast<const __m256i*>(b + j));

    const uint64 a_max = a[i + veclen - 1];
    const uint64 b_max = b[j + veclen - 1];

    int32 cyclic_shift = _MM_SHUFFLE(0, 3, 2, 1);
    __m256i cmp_mask0 = _mm256_cmpeq_epi64(aa, bb);
    bb = _mm256_permute4x64_epi64(bb, cyclic_shift);
    __m256i cmp_mask1 = _mm256_cmpeq_epi64(aa, bb);
    bb = _mm256_permute4x64_epi64(bb, cyclic_shift);
    __m256i cmp_mask2 = _mm256_cmpeq_epi64(aa, bb);
    bb = _mm256_permute4x64_epi64(bb, cyclic_shift);
    __m256i cmp_mask3 = _mm256_cmpeq_epi64(aa, bb);

    __m256d cmp_mask = (__m256d)_mm256_or_si256(_mm256_or_si256(cmp_mask0, cmp_mask1),
                                                _mm256_or_si256(cmp_mask2, cmp_mask3));

    int32 mask = _mm256_movemask_pd(cmp_mask);

    // mask to index, index_mask_4x64
    if (mask & 0x1) {
      out[*out_len] = i;
      (*out_len)++;
    }

    if ((mask >> 1) & 0x1) {
      out[*out_len] = i + 1;
      (*out_len)++;
    }

    if ((mask >> 2) & 0x1) {
      out[*out_len] = i + 2;
      (*out_len)++;
    }

    if ((mask >> 3) & 0x1) {
      out[*out_len] = i + 3;
      (*out_len)++;
    }

    if (a_max == b_max) {
      i += veclen;
      j += veclen;
      _mm_prefetch(a + i, _MM_HINT_NTA);
      _mm_prefetch(b + j, _MM_HINT_NTA);
    } else if (a_max < b_max) {
      i += veclen;
      _mm_prefetch(a + i, _MM_HINT_NTA);
    } else {
      j += veclen;
      _mm_prefetch(b + j, _MM_HINT_NTA);
    }
  }

  index_lookup_kernel_naive(dict, i, dict_len, query, j, query_len, out, out_len);
  // index_lookup_kernel_naive(dict, 0, dict_len, query, j, query_len, out, out_len);
}

#endif

void lookup_value_indices_ordered(const ID* data, const int32 data_len, const ID* query,
                                  const int32 query_len, int32* const indices) {
#pragma omp parallel
  {
    int tid_size = omp_get_num_threads();
    int tid = omp_get_thread_num();
    int chunk = (query_len + tid_size - 1) / tid_size;
    int st = std::min(query_len, chunk * tid);
    int en = std::min(query_len, chunk * (tid + 1));
    int32 i_len = 0;
#if defined(__AVX__) && defined(__AVX2__)
    index_lookup_kernel_4x64bit_avx2(data, data_len, query + st, en - st, indices + st, &i_len);
#else
    index_lookup_kernel_naive(data, 0, data_len, query, st, en, indices + st, &i_len);
#endif
    CHECK_EQ(i_len, en - st) << " tid " << tid << " tid_size " << tid_size << " query_len "
                             << query_len << " data_len " << data_len << " st " << st << " en "
                             << en << " data " << vec2str<ID>(data, data_len) << " query "
                             << vec2str<ID>(query, query_len) << " res "
                             << vec2str<int32>(indices + st, i_len);
  }
}

void lookup_value_indices_unordered(const ID* data, const int32 data_len, const ID* query,
                                    const int32 query_len, int32* const indices) {
  tsl::sparse_map<ID, Idx> value_map(data_len + 10);
  for (int32 i = 0; i < data_len; ++i) {
    if (value_map.find(data[i]) != value_map.end()) {
      LOG(FATAL) << "values have duplicated elements " << data[i];
    }
    value_map.insert(std::make_pair(data[i], i));
  }

#pragma omp parallel for
  for (int32 i = 0; i < query_len; ++i) {
    auto search = value_map.find(query[i]);
    Idx ind = -1;
    if (search != value_map.end()) {
      ind = search->second;
    } else {
      LOG(FATAL) << "Cannot find key " << query[i] << " in data";
    }
    indices[i] = ind;
  }
}

void lookup_value_indices(const ID* data, const int32 data_len, const ID* query,
                          const int32 query_len, int32* const indices) {
  // Step 1. Check if both sets are sorted
  // Step 2. if sorted, use simd set-intersection
  // if not, use hash map version.
  // NOTICE, platodeep will sort node_maps by default, except the nodes in seed layer.
  bool sorted;
  sorted = is_sorted_ascending<ID>(query, query_len);
  if (sorted) {
    sorted = sorted && is_sorted_ascending<ID>(data, data_len);
  }

  if (sorted) {
    lookup_value_indices_ordered(data, data_len, query, query_len, indices);
  } else {
    lookup_value_indices_unordered(data, data_len, query, query_len, indices);
  }
}

}  // namespace cpu
}  // namespace plas
