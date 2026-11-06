"""Cost estimates for ideas that were not implemented (EXPERIMENTAL).

Kernel-only timings with synthetic but realistic data:

idea 7  fewer buckets: monthly for the first year, annual after that
        (12 + 5 = 17 buckets instead of 72), estimated by the cost of the
        kernel with 17 instead of 72 buckets.
idea 8  parameter-set dimension for calibration: N parameter sets of an
        n-point glacier = one kernel call with N * n rows, compared with N
        separate calls.
idea 5  ring buffer vs shift for the aging in the numpy kernel (also
        available as window_aging in FastSfcTypeTIModel).

Writes results/bench_ideas.json.
"""
import os
import sys
import json

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
import common  # noqa: E402
from oggm.sandbox import sfc_type_fast as sf  # noqa: E402

OUT = os.path.join(os.path.dirname(__file__), 'results')


def data(n, nb, nt, seed=0):
    rng = np.random.default_rng(seed)
    mf = np.concatenate([6.5 + (3.25 - 6.5) * np.exp(-np.linspace(0, 6, nb)),
                         [6.5]])
    rho = np.linspace(300, 900, nb)
    mb0 = np.abs(rng.normal(100, 50, (n, nb + 1)))
    mb0[:, -1] = 0
    ps = np.abs(rng.normal(30, 20, (nt, n)))
    tm = np.abs(rng.normal(30, 30, (nt, n)))
    age = np.ones(nt, dtype=bool)
    outs = [np.empty((nt, n)) for _ in range(5)]
    rec = (np.full(nt, -1, dtype=np.int64), np.empty((0, 0)),
           np.empty((0, 0)), np.empty((0, 0)))
    return mb0, ps, tm, mf, age, outs, rho, rec


def time_kernel(kernel, n, nb, nt=12, **kw):
    mb0, ps, tm, mf, age, outs, rho, rec = data(n, nb, nt)
    f = sf._bucket_kernel_numba if kernel == 'numba' else \
        sf._bucket_kernel_numpy
    run = lambda: f(mb0.copy(), ps, tm, mf, min(12, nb), age, True,  # noqa
                    *outs, False, rho, *rec, **kw)
    run()
    t, _ = common.timeit(run, repeat=5, number=5)
    return 1e6 * t / nt  # us per step


def main():
    os.makedirs(OUT, exist_ok=True)
    kernels = ['numpy'] + (['numba'] if sf.HAS_NUMBA else [])
    res = {'idea7': {}, 'idea8': {}, 'idea5': {}}
    for k in kernels:
        for n in [10, 50, 100, 300, 1000]:
            a = time_kernel(k, n, 72)
            b = time_kernel(k, n, 17)
            res['idea7'][f'{k} n={n}'] = {'72 buckets': a, '17 buckets': b,
                                          'speedup': a / b}
            print(f'idea7 {k:6s} n={n:5d}: 72 buckets {a:8.1f} us/step, '
                  f'17 buckets {b:8.1f} us/step ({a / b:.2f}x)')
    for k in kernels:
        for n, N in [(69, 10), (69, 50), (300, 10)]:
            single = time_kernel(k, n, 72)
            batched = time_kernel(k, n * N, 72)
            res['idea8'][f'{k} n={n} N={N}'] = {
                'N separate': N * single, 'one batched': batched,
                'speedup': N * single / batched}
            print(f'idea8 {k:6s} n={n} N={N}: {N} separate calls '
                  f'{N * single:8.1f} us/step, batched {batched:8.1f} '
                  f'({N * single / batched:.2f}x, kernel only)')
    for n in [10, 100, 1000]:
        a = time_kernel('numpy', n, 72, window_aging=False)
        b = time_kernel('numpy', n, 72, window_aging=True)
        res['idea5'][f'n={n}'] = {'shift': a, 'window': b, 'speedup': a / b}
        print(f'idea5 numpy n={n:5d}: shift {a:8.1f} window {b:8.1f} '
              f'({a / b:.2f}x)')
    with open(os.path.join(OUT, 'bench_ideas.json'), 'w') as f:
        json.dump(res, f, indent=1)


if __name__ == '__main__':
    main()
