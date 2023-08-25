#include "../contrib/boysf_kernel.h"
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

double AccurateBoysF(double x, int n)
{
    // NH trapezoidal calculations.
    const int NH = 4;
    double tdts[NH];
    double tresults[NH];
    int ngrids = 25;    
    for(int i = 0; i < NH; ++i)
    {
        ngrids *= 4;
        int n2 = 2*n;    
        double tresult = (n2 == 0) ? 0.5 : 0;
        double dt = 1.0/ngrids;
        double t = dt;
        for(int j = 1; j < ngrids; ++j)
        {
            tresult += exp(-x*t*t)*pow(t, n2);
            t += dt;
        }
        tresult += exp(-x*t*t)*pow(t, n2)/2;
        tresult *= dt;
        tdts[i] = dt;
        tresults[i] = tresult;
    }
    
    // Neville polynomial extrapolate them to zero step.    
    double tresult01 = (-tdts[1])*(tresults[0]-tresults[1])/(tdts[0]-tdts[1])+tresults[1];
    double tresult12 = (-tdts[2])*(tresults[1]-tresults[2])/(tdts[1]-tdts[2])+tresults[2];
    double tresult23 = (-tdts[3])*(tresults[2]-tresults[3])/(tdts[2]-tdts[3])+tresults[3];
    double tresult012 = (-tdts[2])*(tresult01-tresult12)/(tdts[0]-tdts[2])+tresult12;
    double tresult123 = (-tdts[3])*(tresult12-tresult23)/(tdts[1]-tdts[3])+tresult23;
    double result = (-tdts[3])*(tresult012-tresult123)/(tdts[0]-tdts[3])+tresult123;

    return result;
}

int main(int argc, char** argv) {
  plas::gpu::standard_context_t context;

  CLI::App app{"CUDA BoysF Benchmark"};

  // by default set 64 values to compute
  int count = 64;
  int n = 10; // algorithm related param, set to 10 by default
  bool test = false;

  app.add_option("-c, --count", count, "Number of values to compute");
  app.add_option("-n, --nparam", n, "n is a parameter of the algorithm");
  app.add_option("-t, --test", test, "whether to verify results with CPU code");

  CLI11_PARSE(app, argc, argv);
  
  plas::gpu::mem_t<double> input_d =
      plas::gpu::fill_random(10.0, 100.0, count,
                             /*sorted=*/false, context);
  plas::gpu::mem_t<double> output_d = plas::gpu::fill(0.0, count, context);

  // Launch the kernel
  double* output = output_d.data();
  double* input = input_d.data();

  context.timer_begin();
  for (int i = 0; i < 10; ++i) {
    plas::gpu::boysf(input, n, count, output, context);
  }
  double elapsed = context.timer_end();
  std::cout << "GPU time: " << elapsed / 10 << "msec." << std::endl;

  std::vector<double> input_h = plas::gpu::from_mem(input_d);
  std::vector<double> output_h = plas::gpu::from_mem(output_d);

  if (test) {
    bool correct = true;
    context.cpu_timer_begin();
    for (int i = 0; i < output_h.size(); ++i) {
      double accurate = AccurateBoysF(input_h[i], n);
      if (std::abs(output_h[i] - accurate) > 1e-3) {
        correct = false;
        break;
      }
    }

    double elapsed_cpu = context.cpu_timer_end();
    std::cout << "CPU time: " << elapsed_cpu << "msec." << std::endl;
    std::cout << (correct ? "correct!" : "error!") << std::endl;
  }

  return 0;
}
