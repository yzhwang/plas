#ifndef PLAS_GPU_LAUNCH_PARAMS_H_
#define PLAS_GPU_LAUNCH_PARAMS_H_

#include "meta.h"

#ifdef __CUDA_ARCH__
#if __CUDA_ARCH__ == 860
#define PLAS_SM_TAG sm_86
#elif __CUDA_ARCH__ >= 800
#define PLAS_SM_TAG sm_80
#elif __CUDA_ARCH__ >= 750
#define PLAS_SM_TAG sm_75
#elif __CUDA_ARCH__ >= 700
#define PLAS_SM_TAG sm_70
#elif __CUDA_ARCH__ == 620
#define PLAS_SM_TAG sm_62
#elif __CUDA_ARCH__ >= 610
#define PLAS_SM_TAG sm_61
#elif __CUDA_ARCH__ >= 600
#define PLAS_SM_TAG sm_60
#elif __CUDA_ARCH__ == 530
#define PLAS_SM_TAG sm_53
#elif __CUDA_ARCH__ >= 520
#define PLAS_SM_TAG sm_52
#elif __CUDA_ARCH__ >= 500
#define PLAS_SM_TAG sm_50
#elif __CUDA_ARCH__ == 370
#define PLAS_SM_TAG sm_37
#elif __CUDA_ARCH__ >= 350
#define PLAS_SM_TAG sm_35
#elif __CUDA_ARCH__ == 320
#define PLAS_SM_TAG sm_32
#elif __CUDA_ARCH__ >= 300
#define PLAS_SM_TAG sm_30
#else
#error "Plas GPU does not support builds for sm_2.x and below"
#endif
#else  // __CUDA_ARCH__
#define PLAS_SM_TAG sm_00
#endif

#define PLAS_LAUNCH_PARAMS(launch_box) typename launch_box::PLAS_SM_TAG
#define PLAS_LAUNCH_BOUNDS(launch_box) \
  __launch_bounds__(launch_box::sm_ptx::nt, launch_box::sm_ptx::occ)

namespace plas {
namespace gpu {

struct PLAS_ALIGN(8) cta_dim_t {
  int nt, vt;
  int nv() const { return nt * vt; }
  int num_ctas(int count) const { return div_up(count, nv()); }
};

// Generic thread cta kernel.
template <typename launch_box, typename func_t, typename... args_t>
__global__ PLAS_LAUNCH_BOUNDS(launch_box) void launch_box_cta_k(
    func_t f, args_t... args) {
  // Masking threadIdx.x by (nt - 1) may help strength reduction because the
  // compiler now knows the range of tid: (0, nt).
  typedef typename launch_box::sm_ptx params_t;
  int tid = (int)(threadIdx.x % (unsigned)params_t::nt);
  int cta = blockIdx.x;

  f(tid, cta, make_restrict(args)...);
}

// Dummy kernel for retrieving PTX version.
template <int dummy_arg>
__global__ void dummy_k() {}

template <int nt_, int vt_ = 1, int vt0_ = vt_, int occ_ = 0>
struct launch_cta_t {
  enum {
    nt = nt_,
    vt = vt_,
    vt0 = vt0_,
    occ = occ_
  };
};

#define DEF_ARCH_STRUCT(ver)                               \
  template <typename params_t, typename base_t = empty_t>  \
  struct arch_##ver : base_t {                             \
    typedef params_t sm_##ver;                             \
                                                           \
    template <typename new_base_t>                         \
    using rebind = arch_##ver<params_t, new_base_t>;       \
  };                                                       \
                                                           \
  template <int nt, int vt = 1, int vt0 = vt, int occ = 0> \
  using arch_##ver##_cta = arch_##ver<launch_cta_t<nt, vt, vt0, occ> >;

DEF_ARCH_STRUCT(30)
DEF_ARCH_STRUCT(32)
DEF_ARCH_STRUCT(35)
DEF_ARCH_STRUCT(37)
DEF_ARCH_STRUCT(50)
DEF_ARCH_STRUCT(52)
DEF_ARCH_STRUCT(53)
DEF_ARCH_STRUCT(60)
DEF_ARCH_STRUCT(61)
DEF_ARCH_STRUCT(62)
DEF_ARCH_STRUCT(70)
DEF_ARCH_STRUCT(75)
DEF_ARCH_STRUCT(80)
DEF_ARCH_STRUCT(86)

#undef DEF_ARCH_STRUCT

struct context_t;

// Non-specializable launch parameters.
template <int nt, int vt, int vt0 = vt, int occ = 0>
struct launch_params_t : launch_cta_t<nt, vt, vt0, occ> {
  typedef launch_params_t sm_ptx;

  static cta_dim_t cta_dim() {
    return cta_dim_t{nt, vt};
  }

  static cta_dim_t cta_dim(int) { return cta_dim(); }

  static cta_dim_t cta_dim(const context_t& context) { return cta_dim(); }

  static int nv(const context_t& context) { return cta_dim().nv(); }
};

}  // namespace gpu
}  // namespace plas

#endif
