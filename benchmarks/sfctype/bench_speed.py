"""Speed benchmarks of the SfcTypeTIModel variants (EXPERIMENTAL).

Sections (all results go to results/bench_speed.json):

scaling  mass balance only, HEF main flowline resampled to n grid points.
         Two access patterns:
         - 'annual':  spinup + get_annual_mb for 1970-2001 (calibration-like,
                      12 steps per call with the same heights)
         - 'monthly': spinup + get_monthly_mb for each month 1970-2001 with
                      slightly changing heights (dynamics-like, 1 step/call)
kernel   bucket update only (climate precomputed), block of 12 or 365 steps
         (365 = daily resolution, extrapolation as no daily data offline)
hef      real workflows on HEF: calibration and dynamic runs
dyn_scaling  dynamic run of one resampled HEF flowline with several dx

Usage: python benchmarks/sfctype/bench_speed.py [--quick] [--sections ...]
"""
import os
import sys
import json
import time
import argparse

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
import common  # noqa: E402

from oggm.utils import date_to_floatyear  # noqa: E402
from oggm.core.massbalance import (SfcTypeTIModel, MonthlyTIModel,  # noqa
                                   MultipleFlowlineMassBalance)
from oggm.sandbox import sfc_type_fast as sf  # noqa: E402
from oggm.sandbox.sfc_type_fast import (FastSfcTypeTIModel,  # noqa: E402
                                        FixedHeightSfcTypeTIModel, HAS_NUMBA)

OUT = os.path.join(os.path.dirname(__file__), 'results')

# variants: name -> (class, kwargs). The exact ones are cumulative.
VARIANTS = {
    'reference': (SfcTypeTIModel, {}),
    'calendar+lean numpy (4+6)': (FastSfcTypeTIModel, dict(
        kernel='numpy', batch_climate=False, window_aging=False)),
    '+batch climate (3)': (FastSfcTypeTIModel, dict(
        kernel='numpy', batch_climate=True, window_aging=False)),
    '+window aging (5)': (FastSfcTypeTIModel, dict(
        kernel='numpy', batch_climate=True, window_aging=True)),
}
if HAS_NUMBA:
    VARIANTS['+numba kernel (2)'] = (FastSfcTypeTIModel, dict(kernel='numba'))
for _dz in [10, 25, 50, 100]:
    VARIANTS[f'fixed heights dz={_dz} (1)'] = (FixedHeightSfcTypeTIModel,
                                               dict(dz=_dz))
# marginal cost once the table exists (e.g. other flowlines of the same
# glacier, or repeated runs with the same parameters)
VARIANTS['fixed heights dz=25, cached table'] = (FixedHeightSfcTypeTIModel,
                                                 dict(dz=25, _keep_cache=True))
VARIANTS['MonthlyTIModel (no buckets)'] = (MonthlyTIModel, {})

MONTHS = [float(date_to_floatyear(y, m)) for y in range(1970, 2002)
          for m in range(1, 13)]


def make_model(cls, kw, gdir, fl):
    kw = dict(kw)
    if not kw.pop('_keep_cache', False):
        sf.clear_caches()  # no table sharing between repetitions
    if cls is MonthlyTIModel:
        return MonthlyTIModel(gdir)
    return cls(gdir, fl=fl, mb_model_class=MonthlyTIModel, ys=1970, **kw)


def workload_annual(cls, kw, gdir, fl):
    mb = make_model(cls, kw, gdir, fl)
    h = fl.surface_h
    for yr in range(1970, 2002):
        mb.get_annual_mb(h, yr)
    return mb


def workload_monthly(cls, kw, gdir, fl):
    mb = make_model(cls, kw, gdir, fl)
    h = fl.surface_h.copy()
    for m in MONTHS:
        h -= 0.01
        mb.get_monthly_mb(h, m)
    return mb


def bench_scaling(gdir, sizes, repeat, out):
    res = {}
    for pattern, func in [('annual', workload_annual),
                          ('monthly', workload_monthly)]:
        res[pattern] = {}
        for nx in sizes:
            fl = common.resampled_flowline(gdir, nx=nx)
            res[pattern][nx] = {}
            for name, (cls, kw) in VARIANTS.items():
                func(cls, kw, gdir, fl)  # warm up
                t, _ = common.timeit(lambda: func(cls, kw, gdir, fl),
                                     repeat=repeat)
                nsteps = (32 + 6) * 12
                res[pattern][nx][name] = {'s': t, 'us_per_step':
                                          1e6 * t / nsteps}
                print(f'{pattern:8s} nx={nx:5d} {name:32s} {t * 1e3:9.2f} ms '
                      f'{1e6 * t / nsteps:8.1f} us/step', flush=True)
    out['scaling'] = res


def bench_kernel(sizes, repeat, out):
    """Kernel only (random but realistic data), us per step."""
    rng = np.random.default_rng(0)
    res = {}
    nb = 72
    mf = np.concatenate([6.5 + (3.25 - 6.5) * np.exp(-np.linspace(0, 6, nb)),
                         [6.5]])
    rho = np.linspace(300, 900, nb)
    for nt in [12, 365]:
        res[nt] = {}
        for n in sizes:
            mb0 = np.abs(rng.normal(100, 50, (n, nb + 1)))
            mb0[:, -1] = 0
            ps = np.abs(rng.normal(30, 20, (nt, n))) * 12 / nt
            tm = np.abs(rng.normal(30, 30, (nt, n))) * 12 / nt
            age = np.zeros(nt, dtype=bool)
            age[::max(nt // 12, 1)] = True
            outs = [np.empty((nt, n)) for _ in range(5)]
            rec = (np.full(nt, -1, dtype=np.int64), np.empty((0, 0)),
                   np.empty((0, 0)), np.empty((0, 0)))
            res[nt][n] = {}
            kernels = {'numpy': lambda mb: sf._bucket_kernel_numpy(
                mb, ps, tm, mf, 12, age, True, *outs, False, rho, *rec)}
            if HAS_NUMBA:
                kernels['numba'] = lambda mb: sf._bucket_kernel_numba(
                    mb, ps, tm, mf, 12, age, True, *outs, False, rho, *rec)
            for kname, k in kernels.items():
                k(mb0.copy())
                t, _ = common.timeit(lambda: k(mb0.copy()), repeat=repeat,
                                     number=5)
                res[nt][n][kname] = 1e6 * t / nt
            print(f'kernel nt={nt:4d} n={n:5d} ' + ' '.join(
                f'{k}={v:8.2f} us/step' for k, v in res[nt][n].items()),
                flush=True)
    out['kernel'] = res


def bench_hef(gdir, repeat, out):
    res = {}
    for name, (cls, kw) in VARIANTS.items():
        if '_keep_cache' in kw:
            continue
        if cls is MonthlyTIModel:
            mcls = MonthlyTIModel
        else:
            mcls = common.sfc_class(cls, **kw)

        def calib():
            sf.clear_caches()
            return common.calibrate(gdir, mcls)

        def dyn(feedback):
            sf.clear_caches()
            if cls is MonthlyTIModel:
                return common.dynamic_run(gdir, mcls,
                                          mb_elev_feedback=feedback)
            return common.dynamic_run(gdir, common.sfc_class(cls, **kw),
                                      mb_elev_feedback=feedback)
        r = {}
        r['calibration'], _ = common.timeit(calib, repeat=repeat)
        r['dynamic monthly'], _ = common.timeit(lambda: dyn('monthly'),
                                                repeat=repeat)
        r['dynamic annual'], _ = common.timeit(lambda: dyn('annual'),
                                               repeat=repeat)
        res[name] = r
        print(f'HEF {name:32s} ' + ' '.join(f'{k}={v:6.3f}s'
                                            for k, v in r.items()), flush=True)
    out['hef'] = res


def bench_dyn_scaling(gdir, nxs, repeat, out):
    """One resampled flowline, mb time vs dynamics time."""
    res = {}
    names = ['reference', '+window aging (5)'] + \
        (['+numba kernel (2)'] if HAS_NUMBA else []) + \
        ['fixed heights dz=25 (1)', 'MonthlyTIModel (no buckets)']
    for nx in nxs:
        fl = common.synthetic_glacier(nx)
        res[nx] = {}
        for name in names:
            cls, kw = VARIANTS[name]

            def run():
                sf.clear_caches()
                if cls is MonthlyTIModel:
                    mb = MonthlyTIModel(gdir)
                else:
                    mb = cls(gdir, fl=fl, mb_model_class=MonthlyTIModel,
                             ys=1980, **kw)
                return common.single_fl_dynamic_run(gdir, fl, mb, ys=1980,
                                                    ye=2003)
            t, model = common.timeit(run, repeat=repeat)
            res[nx][name] = t
            print(f'dyn single fl nx={nx:5d} dx={fl.dx_meter:6.1f} m '
                  f'{name:30s} {t:7.3f} s', flush=True)
    out['dyn_scaling'] = res


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--quick', action='store_true')
    p.add_argument('--sections', nargs='*',
                   default=['kernel', 'scaling', 'hef', 'dyn_scaling'])
    args = p.parse_args()
    os.makedirs(OUT, exist_ok=True)
    repeat = 1 if args.quick else 3
    sizes = [10, 30, 100, 300, 1000] if args.quick else \
        [10, 20, 50, 100, 200, 500, 1000]
    fn = os.path.join(OUT, 'bench_speed.json')
    out = {}
    if os.path.exists(fn):
        with open(fn) as f:
            out = json.load(f)
    out['meta'] = {'numba': HAS_NUMBA, 'cpu_count': os.cpu_count(),
                   'date': time.strftime('%Y-%m-%d')}
    gdir = common.get_hef_gdir()
    # warm up numba (compile time is measured separately)
    if HAS_NUMBA:
        t0 = time.perf_counter()
        workload_annual(FastSfcTypeTIModel, dict(kernel='numba'), gdir,
                        common.main_model_flowline(gdir))
        out['meta']['numba_first_call_s'] = time.perf_counter() - t0
    if 'kernel' in args.sections:
        bench_kernel(sizes, repeat, out)
    if 'scaling' in args.sections:
        bench_scaling(gdir, sizes, repeat, out)
    if 'hef' in args.sections:
        bench_hef(gdir, repeat, out)
    if 'dyn_scaling' in args.sections:
        bench_dyn_scaling(gdir, [70, 140, 280, 560], repeat, out)
    with open(fn, 'w') as f:
        json.dump(out, f, indent=1)


if __name__ == '__main__':
    main()
