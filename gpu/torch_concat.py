import torch
def simple_concat(a, b):
    return torch.cat((a, b), dim=1)

t1 = torch.randn(160, 1000000, device='cuda')
t2 = torch.randn(160, 1000000, device='cuda')

import torch.utils.benchmark as benchmark

t0 = benchmark.Timer(
    stmt='simple_concat(t1, t2)',
    setup='from __main__ import simple_concat',
    globals={'t1': t1, 't2': t2})

print(t0.timeit(100))
