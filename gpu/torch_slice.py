import torch
def slice(a):
    return torch.clone(torch.narrow(a, 0, 0, 31000000))

t = torch.randn(32000000, 2, device='cuda')

import torch.utils.benchmark as benchmark

t0 = benchmark.Timer(
    stmt='slice(t)',
    setup='from __main__ import slice',
    globals={'t': t})

print(t0.timeit(100))
