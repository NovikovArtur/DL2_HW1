import json
import time
from pathlib import Path

import numba.cuda
import numpy as np
import minitorch as mt


def measure(a, b, backend, repeats=3):
    def operation():
        if backend is None:
            n = a.shape[0]
            return (a.view(n, n, 1) * b.view(1, n, n)).sum(1).view(n, n)
        return a @ b
    result = operation()
    if backend is not None and backend.cuda:
        numba.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(repeats):
        result = operation()
    if backend is not None and backend.cuda:
        numba.cuda.synchronize()
    return (time.perf_counter() - start) / repeats, result.to_numpy()


def main():
    backends = {'simple': None, 'fast': mt.TensorBackend(mt.FastOps)}
    if numba.cuda.is_available():
        backends['cuda'] = mt.TensorBackend(mt.CudaOps)
    rng = np.random.default_rng(0)
    rows = []
    for size in (16, 32, 64):
        a_np, b_np = rng.random((2, size, size))
        for name, backend in backends.items():
            a = mt.tensor(a_np.tolist(), backend=backend or mt.SimpleBackend)
            b = mt.tensor(b_np.tolist(), backend=backend or mt.SimpleBackend)
            seconds, result = measure(a, b, backend)
            np.testing.assert_allclose(result, a_np @ b_np)
            row = dict(size=size, backend=name, seconds=seconds)
            rows.append(row)
            print(json.dumps(row), flush=True)
    Path('results').mkdir(exist_ok=True)
    Path('results/matmul.json').write_text(json.dumps(rows, indent=2))


if __name__ == '__main__':
    main()
