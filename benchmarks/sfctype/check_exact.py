"""Check that FastSfcTypeTIModel reproduces SfcTypeTIModel (EXPERIMENTAL).

Compares all stored outputs (climatic_mb, ice_mb, melt split, stored buckets,
snowline, final buckets) for several configurations, a calibration and a
dynamic run. Raises if anything differs beyond floating point precision.

Usage: python benchmarks/sfctype/check_exact.py
"""
import os
import sys
import itertools

import numpy as np
from numpy.testing import assert_allclose

sys.path.insert(0, os.path.dirname(__file__))
import common  # noqa: E402

from oggm.utils import date_to_floatyear  # noqa: E402

from oggm.core.massbalance import (SfcTypeTIModel, MonthlyTIModel,  # noqa
                                   MultipleFlowlineMassBalance)
from oggm.sandbox.sfc_type_fast import FastSfcTypeTIModel, HAS_NUMBA  # noqa

RTOL = 1e-12
ATOL = 1e-9  # kg m-2


def compare(ref, new, label):
    n = ref._current_index
    assert n == new._current_index, label
    assert ref._year_to_index == new._year_to_index, label
    for name in ['_climatic_mb', '_ice_mb', '_snow_melt', '_firn_melt',
                 '_ice_melt', '_mb_heights']:
        assert_allclose(getattr(new, name)[:n], getattr(ref, name)[:n],
                        rtol=RTOL, atol=ATOL, err_msg=f'{label} {name}')
    assert_allclose(new.mb_buckets_np, ref.mb_buckets_np, rtol=RTOL,
                    atol=ATOL, err_msg=f'{label} buckets')
    assert ref.mb_buckets_year == new.mb_buckets_year
    if ref.store_buckets:
        assert ref.mb_buckets_stored.keys() == new.mb_buckets_stored.keys()
        for k in ref.mb_buckets_stored:
            assert_allclose(new.mb_buckets_stored[k].values,
                            ref.mb_buckets_stored[k].values,
                            rtol=RTOL, atol=ATOL, err_msg=f'{label} stored')
    if ref.store_snowline:
        assert_allclose(new.snowline, ref.snowline, err_msg=f'{label} sl')
        assert_allclose(new.snowline_year, ref.snowline_year)
    bitwise = all(np.array_equal(getattr(new, a)[:n], getattr(ref, a)[:n])
                  for a in ['_climatic_mb', '_ice_mb'])
    maxdiff = np.max(np.abs(new._climatic_mb[:n] - ref._climatic_mb[:n]))
    print(f'  OK {label:70s} bitwise cmb/ice_mb: {bitwise!s:5s} '
          f'max|d cmb|={maxdiff:.1e} kg m-2')


def run_direct(cls, gdir, fl, **kw):
    mb = cls(gdir, fl=fl, mb_model_class=MonthlyTIModel, ys=1970, **kw)
    for yr in range(1970, 1990):
        mb.get_annual_mb(fl.surface_h, yr)
    # some monthly calls with changing heights (like dynamics)
    h = fl.surface_h.copy()
    for yr in range(1990, 2002):
        for m in range(1, 13):
            h = h - 0.05
            mb.get_monthly_mb(h, date_to_floatyear(yr, m))
    return mb


def main():
    gdir = common.get_hef_gdir()
    fl = common.main_model_flowline(gdir)
    kernels = ['numpy'] + (['numba'] if HAS_NUMBA else [])

    print('direct mb calls on the HEF main flowline:')
    configs = [
        dict(),
        dict(store_buckets='monthly', store_snowline=True),
        dict(aging_frequency='annual'),
        dict(aging_frequency='annual', store_buckets='annual',
             store_snowline=True, store_snowline_start_month='Jun'),
        dict(save_spinup_mbs=False, melt_f_change='linear',
             density_change='linear', spinup_years=3),
    ]
    for cfg_kw, kernel, batch, window in itertools.product(
            configs, kernels, [True, False], [True, False]):
        if kernel == 'numba' and not window:
            continue
        ref = run_direct(SfcTypeTIModel, gdir, fl, **cfg_kw)
        new = run_direct(FastSfcTypeTIModel, gdir, fl, kernel=kernel,
                         batch_climate=batch, window_aging=window,
                         debug_checks=True, **cfg_kw)
        compare(ref, new, f'{cfg_kw} {kernel} batch={batch} win={window}')

    # annual climate resolution
    for kernel in kernels:
        mbr = SfcTypeTIModel(gdir, fl=fl, mb_model_class=MonthlyTIModel,
                             climate_resolution='annual',
                             aging_frequency='annual', ys=1970)
        mbn = FastSfcTypeTIModel(gdir, fl=fl, mb_model_class=MonthlyTIModel,
                                 climate_resolution='annual',
                                 aging_frequency='annual', ys=1970,
                                 kernel=kernel)
        for yr in range(1970, 2002):
            mbr.get_annual_mb(fl.surface_h, yr)
            mbn.get_annual_mb(fl.surface_h, yr)
        compare(mbr, mbn, f'annual climate resolution {kernel}')

    print('calibration:')
    df_ref, mb_ref = common.calibrate(gdir, SfcTypeTIModel)
    for kernel in kernels:
        df_new, mb_new = common.calibrate(
            gdir, common.sfc_class(FastSfcTypeTIModel, kernel=kernel))
        assert_allclose(df_new['melt_f'], df_ref['melt_f'], rtol=1e-10)
        for a, b in zip(mb_ref.flowline_mb_models, mb_new.flowline_mb_models):
            compare(a, b, f'calibration flowline {kernel}')
        print(f"  melt_f ref={df_ref['melt_f']:.10f} "
              f"new={df_new['melt_f']:.10f}")

    print('dynamic run:')
    for feedback in ['monthly', 'annual']:
        r = common.dynamic_run(gdir, SfcTypeTIModel, mb_elev_feedback=feedback)
        for kernel in kernels:
            n = common.dynamic_run(
                gdir, common.sfc_class(FastSfcTypeTIModel, kernel=kernel),
                mb_elev_feedback=feedback)
            assert_allclose(n.volume_m3, r.volume_m3, rtol=1e-10)
            assert_allclose(n.length_m, r.length_m, rtol=1e-10)
            for a, b in zip(r.mb_model.flowline_mb_models,
                            n.mb_model.flowline_mb_models):
                compare(a, b, f'dynamic {feedback} flowline {kernel}')
    print('ALL EXACT CHECKS PASSED')


if __name__ == '__main__':
    main()
