"""Profile the reference SfcTypeTIModel in a real calibration and dynamic run.

EXPERIMENTAL benchmark script. Two complementary measurements:

1. cProfile of the full workflow: how much of the total time is spent in the
   SfcTypeTIModel at all (vs. calibration overhead, dynamics, I/O).
2. An instrumented run (wrappers with ``time.perf_counter`` around the
   sub-steps of ``_apply_climate_step_and_aging_to_buckets``): how a climate
   step splits into climate lookup, bucket update, date bookkeeping, aging,
   storing and the rest. cProfile is not used for this because its per-call
   overhead distorts many tiny calls.

Usage: python benchmarks/sfctype/profile_reference.py [--out results/]
"""
import os
import sys
import time
import json
import pstats
import cProfile
import argparse
from collections import defaultdict


sys.path.insert(0, os.path.dirname(__file__))
import common  # noqa: E402

from oggm.core import massbalance  # noqa: E402
from oggm.core.massbalance import SfcTypeTIModel, MonthlyTIModel  # noqa: E402

TIMERS = defaultdict(float)
COUNTS = defaultdict(int)


def _wrap(name, func):
    def wrapped(*args, **kwargs):
        t0 = time.perf_counter()
        try:
            return func(*args, **kwargs)
        finally:
            TIMERS[name] += time.perf_counter() - t0
            COUNTS[name] += 1
    wrapped._orig = func
    return wrapped


class InstrumentedSfc(SfcTypeTIModel):
    """Reference model with timers around the step sub-parts."""

    def _apply_climate_step_and_aging_to_buckets(self, heights, year=None,
                                                 save_mbs=True):
        t0 = time.perf_counter()
        out = super()._apply_climate_step_and_aging_to_buckets(
            heights, year=year, save_mbs=save_mbs)
        TIMERS['step_total'] += time.perf_counter() - t0
        COUNTS['step_total'] += 1
        return out


def instrument():
    """Patch the module-level helpers used by the step."""
    massbalance.floatyear_to_date = _wrap('dates', massbalance.floatyear_to_date)
    massbalance.date_to_floatyear = _wrap('dates', massbalance.date_to_floatyear)
    MonthlyTIModel.get_monthly_climate = _wrap(
        'climate', MonthlyTIModel.get_monthly_climate)
    SfcTypeTIModel._bucket_aging = _wrap('aging', SfcTypeTIModel._bucket_aging)
    SfcTypeTIModel._run_until = _wrap('run_until', SfcTypeTIModel._run_until)


def uninstrument():
    massbalance.floatyear_to_date = massbalance.floatyear_to_date._orig
    massbalance.date_to_floatyear = massbalance.date_to_floatyear._orig
    MonthlyTIModel.get_monthly_climate = \
        MonthlyTIModel.get_monthly_climate._orig
    SfcTypeTIModel._bucket_aging = SfcTypeTIModel._bucket_aging._orig
    SfcTypeTIModel._run_until = SfcTypeTIModel._run_until._orig


def breakdown(total_wall):
    """Split of the step time. Note: the climate call itself calls
    floatyear_to_date once, these calls are counted in 'climate' and
    removed from 'dates'."""
    n = COUNTS['step_total']
    ncl = COUNTS['climate']
    # floatyear_to_date inside get_monthly_climate, estimate per-call cost
    per_date = TIMERS['dates'] / max(COUNTS['dates'], 1)
    dates_in_step = TIMERS['dates'] - ncl * per_date
    step = TIMERS['step_total']
    parts = {
        'climate lookup (get_monthly_climate)': TIMERS['climate'],
        'date bookkeeping (floatyear/date conversions)': dates_in_step,
        'aging (_bucket_aging)': TIMERS['aging'],
    }
    parts['bucket update + python overhead (rest of step)'] = \
        step - sum(parts.values())
    res = {
        'n_steps': n,
        'wall_total_s': total_wall,
        'steps_total_s': step,
        'us_per_step': 1e6 * step / max(n, 1),
        'run_until_total_s': TIMERS['run_until'],
        'parts_s': parts,
        'parts_us_per_step': {k: 1e6 * v / max(n, 1) for k, v in parts.items()},
    }
    return res


def instrumented(name, func, out):
    TIMERS.clear()
    COUNTS.clear()
    instrument()
    try:
        t0 = time.perf_counter()
        func()
        wall = time.perf_counter() - t0
    finally:
        uninstrument()
    res = breakdown(wall)
    out[name] = res
    print(f'\n== {name} (instrumented) ==')
    print(f"  wall: {wall:.3f} s, steps: {res['n_steps']}, "
          f"time in steps: {res['steps_total_s']:.3f} s "
          f"({100 * res['steps_total_s'] / wall:.0f}% of wall), "
          f"{res['us_per_step']:.1f} us/step")
    for k, v in res['parts_us_per_step'].items():
        print(f'    {k:50s} {v:7.1f} us/step '
              f'({100 * v / res["us_per_step"]:.0f}%)')


def profiled(name, func, outdir, out):
    pr = cProfile.Profile()
    t0 = time.perf_counter()
    pr.enable()
    func()
    pr.disable()
    wall = time.perf_counter() - t0
    fn = os.path.join(outdir, f'cprofile_{name}.prof')
    pr.dump_stats(fn)
    st = pstats.Stats(pr)
    # cumulative time of a few interesting functions
    want = {
        '_apply_climate_step_and_aging_to_buckets': 'sfc step',
        '_init_buckets': 'sfc _init_buckets (spinup)',
        'get_monthly_mb': 'get_monthly_mb (all classes)',
        'get_annual_mb': 'get_annual_mb (all classes)',
        'run_until_and_store': 'flowline run_until_and_store',
        'step': 'flowline step (all classes)',
    }
    cum = defaultdict(float)
    for (file, line, fname), (cc, nc, tt, ct, callers) in st.stats.items():
        if fname in want and 'oggm' in file:
            # avoid double counting recursive / nested calls of same name
            cum[want[fname]] = max(cum[want[fname]], ct)
    out[name] = {'wall_profiled_s': wall, 'cumtime_s': dict(cum)}
    print(f'\n== {name} (cProfile, {wall:.2f} s with profiler overhead) ==')
    for k, v in cum.items():
        print(f'    {k:40s} {v:7.3f} s ({100 * v / wall:.0f}%)')
    s = pstats.Stats(fn)
    s.sort_stats('tottime')
    with open(os.path.join(outdir, f'cprofile_{name}_top.txt'), 'w') as f:
        s.stream = f
        s.print_stats(30)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--out',
                   default=os.path.join(os.path.dirname(__file__), 'results'))
    args = p.parse_args()
    os.makedirs(args.out, exist_ok=True)
    gdir = common.get_hef_gdir()

    calib = lambda: common.calibrate(gdir, InstrumentedSfc)  # noqa: E731
    dyn_m = lambda: common.dynamic_run(gdir, InstrumentedSfc,  # noqa: E731
                                       mb_elev_feedback='monthly')
    dyn_a = lambda: common.dynamic_run(gdir, InstrumentedSfc,  # noqa: E731
                                       mb_elev_feedback='annual')
    # warm up caches (file reading, numba etc.)
    calib()
    dyn_a()

    # pure TI model for comparison
    t_calib_ti, _ = common.timeit(lambda: common.calibrate(gdir,
                                                           MonthlyTIModel))
    t_dyn_ti, _ = common.timeit(lambda: common.dynamic_run(gdir,
                                                           MonthlyTIModel))
    t_calib, _ = common.timeit(calib)
    t_dyn_m, _ = common.timeit(dyn_m)
    t_dyn_a, _ = common.timeit(dyn_a)
    out = {'wall_s': {'calibration MonthlyTIModel': t_calib_ti,
                      'calibration SfcTypeTIModel': t_calib,
                      'dynamic monthly-feedback MonthlyTIModel': t_dyn_ti,
                      'dynamic monthly-feedback SfcTypeTIModel': t_dyn_m,
                      'dynamic annual-feedback SfcTypeTIModel': t_dyn_a}}
    print('\n== wall times (best of 3) ==')
    for k, v in out['wall_s'].items():
        print(f'    {k:45s} {v:7.3f} s')

    for name, f in [('calibration', calib), ('dynamic_monthly', dyn_m),
                    ('dynamic_annual', dyn_a)]:
        instrumented(name, f, out)
    prof = {}
    for name, f in [('calibration', calib), ('dynamic_monthly', dyn_m)]:
        profiled(name, f, args.out, prof)
    out['cprofile'] = prof

    with open(os.path.join(args.out, 'profile_reference.json'), 'w') as f:
        json.dump(out, f, indent=1)


if __name__ == '__main__':
    main()
