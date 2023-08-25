import torch
def simple_sum(a):
    return torch.sum(a)

t1 = torch.randn(64, 1600, 1600, 3, device='cuda')

import torch.utils.benchmark as benchmark

t0 = benchmark.Timer(
    stmt='simple_sum(t1)',
    setup='from __main__ import simple_sum',
    globals={'t1': t1})

print(t0.timeit(100))
