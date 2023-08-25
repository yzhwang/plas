#ifndef PLAS_GPU_UTIL_H_
#define PLAS_GPU_UTIL_H_

#include <cstdarg>
#include <string>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <cuda_runtime.h>

#include <random>
#include <iostream>

namespace plas {
namespace gpu {

namespace detail {

inline std::string stringprintf(const char* format, ...) {
  va_list args;
  va_start(args, format);
  int len = vsnprintf(0, 0, format, args);
  va_end(args);

  // allocate space.
  std::string text;
  text.resize(len);

  va_start(args, format);
  vsnprintf(&text[0], len + 1, format, args);
  va_end(args);

  return text;
}

}  // namespace detail

struct RandomRealGenerator {
    std::mt19937_64 engine;
    std::uniform_real_distribution<float> dist;

    RandomRealGenerator(float lower, float upper) : dist(lower, upper) {
    std::random_device rd;
    engine = std::mt19937_64(rd());
    }

    float operator()() {
        return dist(engine);
    }
};

struct RandomIntGenerator {
    std::mt19937_64 engine;
    std::uniform_int_distribution<int> dist;

    RandomIntGenerator(int lower, int upper) : dist(lower, upper) {
    std::random_device rd;
    engine = std::mt19937_64(rd());
    }

    float operator()() {
        return dist(engine);
    }
};

void GetRandomRealDeviceArray(thrust::device_vector<float> &device_vec, float lower, float upper, size_t num) {
    thrust::host_vector<float> host_vec(num);
    RandomRealGenerator generator(lower, upper);
    for (size_t i = 0; i < num; ++i) {
        host_vec[i] = generator();
    }
    thrust::copy(host_vec.begin(), host_vec.end(), device_vec.begin());
}

void GetRandomIntDeviceArray(thrust::device_vector<int> &device_vec, int lower, int upper, size_t num) {
    thrust::host_vector<int> host_vec(num);
    RandomIntGenerator generator(lower, upper);
    for (size_t i = 0; i < num; ++i) {
        host_vec[i] = generator();
    }
    thrust::copy(host_vec.begin(), host_vec.end(), device_vec.begin());
}

}  // namespace gpu
}  // namespace plas

#endif
