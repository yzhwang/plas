#include "../contrib/matmul_kernel.h"
#include "../kernel_launch.h"
#include "../memory.h"
#include "../loadstore.h"
#include <iostream>
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

template <typename T>
bool VerifyMatMul(thrust::device_vector<T>& A, thrust::device_vector<T>& B,
thrust::device_vector<T>& C, int m, int n, int k)
{
    thrust::host_vector<T> A_host(m*k);
    thrust::host_vector<T> B_host(k*n);
    thrust::host_vector<T> C_host(m*n);
    thrust::copy(A.begin(), A.end(), A_host.begin());
    thrust::copy(B.begin(), B.end(), B_host.begin());
    thrust::copy(C.begin(), C.end(), C_host.begin());

    for (int i = 0; i < m; ++i) {
      for (int j = 0; j < n; ++j) {
        T tmp = 0;
        for (int kk = 0; kk < k; ++kk) {
          tmp += A_host[i*k + kk] * B_host[kk*n + j];
        }
        if (std::abs(C_host[i * n + j] - tmp) > 1e-2) {
            return false;
        }
      }
    }
    return true;
}

template <typename input_it, typename output_it>
void matmul_kernel(input_it A, input_it B, int m, int n, int k, output_it C, plas::gpu::standard_context_t& context, int op) {
  // Define the function object for the bound kernel function
    std::function<void(input_it, input_it, int, int, int, output_it, plas::gpu::standard_context_t&)> bound_kernel;
    // Check the value of command line option "p"
    switch (op) {
        default:
        case 0:
            bound_kernel = std::bind(&plas::gpu::matmul_baseline<input_it, output_it>, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3,
                                std::placeholders::_4, std::placeholders::_5, std::placeholders::_6, std::placeholders::_7);
            break;
        case 1:
            bound_kernel = std::bind(&plas::gpu::matmul_coalesce<input_it, output_it>, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3,
                                std::placeholders::_4, std::placeholders::_5, std::placeholders::_6, std::placeholders::_7);
            break;
        case 2:
            bound_kernel = std::bind(&plas::gpu::matmul_shmem<input_it, output_it>, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3,
                                std::placeholders::_4, std::placeholders::_5, std::placeholders::_6, std::placeholders::_7);
            break;
        case 3:
            bound_kernel = std::bind(&plas::gpu::matmul_1dblocktiling<input_it, output_it>, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3,
                                std::placeholders::_4, std::placeholders::_5, std::placeholders::_6, std::placeholders::_7);
            break;
        case 4:
            bound_kernel = std::bind(&plas::gpu::matmul_2dblocktiling<input_it, output_it>, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3,
                                std::placeholders::_4, std::placeholders::_5, std::placeholders::_6, std::placeholders::_7);
            break;
        case 5:
            bound_kernel = std::bind(&plas::gpu::matmul_2dblocktiling_vec<input_it, output_it>, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3,
                                std::placeholders::_4, std::placeholders::_5, std::placeholders::_6, std::placeholders::_7);
            break;
    }
    bound_kernel(A, B, m, n, k, C, context);
}



int main(int argc, char** argv) {
  plas::gpu::standard_context_t context;

  int m = 4096;
  int n = 4096;
  int k = 4096;
  int option = 0;
  bool test = false;
  CLI::App app{"CUDA Matmul Benchmark"};
  app.add_option("-m, --num_row", m, "Number of rows of matrix A");
  app.add_option("-n, --num_col", n, "Number of columns of matrix B");
  app.add_option("-k, --num_row_col", k, "Number of {columns,rows} of {matrix A, matrix B}");
  app.add_option("-t, --test", test, "whether to verify results with CPU code");
  app.add_option("-p, --option", option, "type of kernel used for test");

  CLI11_PARSE(app, argc, argv);

  // Initialize A B and C
  thrust::device_vector<float> A(m*k);
  thrust::device_vector<float> B(k*n);
  thrust::device_vector<float> C(m*n);
  plas::gpu::GetRandomRealDeviceArray(A, 1, 10, m*k);
  plas::gpu::GetRandomRealDeviceArray(B, 1, 10, k*n);
  plas::gpu::GetRandomRealDeviceArray(C, 1, 10, m*n);

  float* A_device = thrust::raw_pointer_cast(A.data());
  float* B_device = thrust::raw_pointer_cast(B.data());
  float* C_device = thrust::raw_pointer_cast(C.data());

  // bind kernel according to option

  context.timer_begin();
  for (int i = 0; i < 10; ++i) {
    matmul_kernel(A_device, B_device, m, n, k, C_device, context, option);
  }
  double elapsed = context.timer_end();
  std::cout << "GPU time: " << elapsed / 10 << "msec." << std::endl;

  std::cout << "GFLOPs/sec: " << double(m) * double(n) * double(k) * 2.0 / (1000 * 1000 * elapsed / 10) << std::endl;

  if (test) {
    bool result = VerifyMatMul(A, B, C, m, n, k);
    if (result)
        std::cout << "Verification succeeded" << std::endl;
    else
        std::cout << "Verification failed" << std::endl;
  }

  return 0;
}