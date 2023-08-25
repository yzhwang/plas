#include "../contrib/eri_kernel.h"
#include "../kernel_launch.h"
#include "../memory.h"
#include "../loadstore.h"
#include <iostream>
#include <limits>
#include <vector>
#include <algorithm>
#include "../CLI11.hpp"

#include <thrust/copy.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/generate.h>
#include <thrust/random/linear_congruential_engine.h>
#include <thrust/random/uniform_int_distribution.h>
#include <functional>

bool VerifyEri(thrust::host_vector<int>& N, thrust::host_vector<float>& R, thrust::host_vector<float>& Z,
  int batch, int* min_a, int* min_c, int* max_ab, int* max_cd, int* Ms, thrust::device_vector<float>& output) {

    thrust::host_vector<float> output_host(batch);
    thrust::copy(output.begin(), output.end(), output_host.begin());
    // Verify
    for (int j = 0; j < batch; ++j) {
    FLOAT out = plas::gpu::eri(N[j], N[batch+j], N[batch*2+j],
        N[batch*3+j], N[batch*4+j], N[batch*5+j],
        N[batch*6+j], N[batch*7+j], N[batch*8+j],
        N[batch*9+j], N[batch*10+j], N[batch*11+j],
        R[j], R[batch+j], R[batch*2+j],
        R[batch*3+j], R[batch*4+j], R[batch*5+j],
        R[batch*6+j], R[batch*7+j], R[batch*8+j],
        R[batch*9+j], R[batch*10+j], R[batch*11+j],
        Z[j], Z[batch+j], Z[batch*2+j], Z[batch*3+j],
        min_a, min_c, max_ab, max_cd, Ms);
    if (std::abs(output_host[j] - out) > 1e-2) {
            return false;
    }
  }
    return true;
}

int main(int argc, char** argv) {
  plas::gpu::standard_context_t context;

  int batch = 4096;
  CLI::App app{"CUDA ERI Benchmark"};
  app.add_option("-b, --batch", batch, "Batch size");

  CLI11_PARSE(app, argc, argv);

  // Initialize N, R and Z
  thrust::host_vector<int> N(12*batch);
  thrust::fill(thrust::host, N.begin(), N.end(), 1);
  thrust::host_vector<float> R(12*batch);
  thrust::fill(thrust::host, R.begin(), R.end(), 0.001);
  thrust::inclusive_scan(R.begin(), R.end(), R.begin());
  thrust::host_vector<float> Z(4*batch);
  thrust::fill(thrust::host, Z.begin(), Z.end(), 1.1);


  thrust::device_vector<float> output_d(batch);

  // Prepare data
  std::vector<int> min_a(3, std::numeric_limits<int>::max());
  std::vector<int> min_c(3, std::numeric_limits<int>::max());
  std::vector<int> max_ab(3, std::numeric_limits<int>::min());
  std::vector<int> max_cd(3, std::numeric_limits<int>::min());
  for (int i = 0; i < 3; ++i) {
    for (int j = 0; j < batch; ++j) {
      min_a[i] = min(min_a[i], N[i*batch + j]);
      min_c[i] = min(min_c[i], N[(i + 6)*batch + j]);
      max_ab[i] = max(max_ab[i], N[i*batch + j] + N[(i + 3)*batch + j]);
      max_cd[i] = max(max_cd[i], N[(i + 6)*batch + j] + N[(i + 9)*batch + j]);
    }
  }

  int max_xyz, max_yz, max_z;
  max_xyz = std::numeric_limits<int>::min();
  max_yz = std::numeric_limits<int>::min();
  max_z = std::numeric_limits<int>::min();
  for (int j = 0; j < batch; ++j) {
    max_xyz = max(max_xyz, N[j] + N[batch+j] + N[batch*2+j] +
                  N[batch*3+j] + N[batch*4+j] + N[batch*5+j] +
                  N[batch*6+j] + N[batch*7+j] + N[batch*8+j] +
                  N[batch*9+j] + N[batch*10+j] + N[batch*11+j]);
    max_yz = max(max_yz, N[batch+j] + N[batch*2+j] +
                 N[batch*4+j] + N[batch*5+j] +
                 N[batch*7+j] + N[batch*8+j] +
                 N[batch*10+j] + N[batch*11+j]);
    max_z = max(max_z, N[batch*2+j] + N[batch*5+j] +
                N[batch*8+j] + N[batch*11+j]);
  }

  std::vector<int> Ms = {max_xyz, max_yz, max_z, 1};

  // prepare device array GPU needs
  plas::gpu::mem_t<int> min_a_d = plas::gpu::to_mem(min_a, context);
  plas::gpu::mem_t<int> min_c_d = plas::gpu::to_mem(min_c, context);
  plas::gpu::mem_t<int> max_ab_d = plas::gpu::to_mem(max_ab, context);
  plas::gpu::mem_t<int> max_cd_d = plas::gpu::to_mem(max_cd, context);
  plas::gpu::mem_t<int> Ms_d = plas::gpu::to_mem(Ms, context);

  
  thrust::device_vector<int> N_d(N.begin(), N.end());
  thrust::device_vector<float> R_d(R.begin(), R.end());
  thrust::device_vector<float> Z_d(Z.begin(), Z.end());

  int* n_d_ptr = thrust::raw_pointer_cast(N_d.data());
  float* r_d_ptr = thrust::raw_pointer_cast(R_d.data());
  float* z_d_ptr = thrust::raw_pointer_cast(Z_d.data());
  float* output_d_ptr = thrust::raw_pointer_cast(output_d.data());

  context.timer_begin();
  for (int i = 0; i < 10; ++i) {
    // GPU kernel
    plas::gpu::eri_batch(n_d_ptr, r_d_ptr, z_d_ptr, batch, min_a_d.data(), min_c_d.data(), max_ab_d.data(), max_cd_d.data(), Ms_d.data(), output_d_ptr, context);
  }
  double elapsed = context.timer_end();
  std::cout << "GPU time: " << elapsed / 10 << "msec." << std::endl;

  bool result = VerifyEri(N, R, Z, batch, min_a.data(), min_c.data(), max_ab.data(), max_cd.data(), Ms.data(), output_d);
  if (result)
      std::cout << "Verification succeeded" << std::endl;
  else
      std::cout << "Verification failed" << std::endl;

  return 0;
}
