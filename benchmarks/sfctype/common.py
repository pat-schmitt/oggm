"""Shared setup for the SfcTypeTIModel performance study (EXPERIMENTAL).

All benchmarks use Hintereisferner (HEF, RGI50-11.00897) from the OGGM test
sample data with the HISTALP monthly climate (the only climate available
offline). See ``benchmarks/sfctype/README.md`` for how to get the data.
"""
import time
import logging
import warnings
from functools import partial

import numpy as np

warnings.filterwarnings('ignore')

HEF_YS_CALIB = 1970
HEF_YE_CALIB = 2001  # inclusive (ref_mb_years=(1970, 2001))
HEF_YS_DYN = 1960
HEF_YE_DYN = 2003


def get_hef_gdir():
    """Initialised HEF glacier directory (same as the test fixture)."""
    from oggm.tests.funcs import init_hef
    from oggm.core.flowline import init_present_time_glacier
    gdir = init_hef(logging_level='CRITICAL')
    logging.getLogger('oggm').setLevel(logging.CRITICAL)
    init_present_time_glacier(gdir)
    return gdir


def main_model_flowline(gdir):
    return gdir.read_store('model_flowlines')[-1]


def resampled_flowline(gdir, nx=None, dx_factor=None, length_factor=1.0):
    """A single rectangular flowline built from the HEF main flowline.

    Either ``nx`` (number of grid points, same glacier length) or
    ``dx_factor`` (new dx = dx_factor * dx) can be given. ``length_factor``
    stretches the glacier along the flow (synthetic longer glacier with the
    same elevation range, i.e. a flatter glacier).
    """
    from oggm.core.flowline import RectangularBedFlowline
    fl = main_model_flowline(gdir)
    x = np.arange(fl.nx) * fl.dx_meter * length_factor
    if nx is None:
        nx = int(round(fl.nx * length_factor / dx_factor))
    xn = np.linspace(0, x[-1], nx)
    dx_meter = xn[1] - xn[0]
    surface_h = np.interp(xn, x, fl.surface_h)
    bed_h = np.interp(xn, x, fl.bed_h)
    widths_m = np.interp(xn, x, fl.widths_m)
    new = RectangularBedFlowline(dx=dx_meter / fl.map_dx, map_dx=fl.map_dx,
                                 surface_h=surface_h, bed_h=bed_h,
                                 widths=widths_m / fl.map_dx,
                                 rgi_id=fl.rgi_id)
    return new


def synthetic_glacier(nx, length_m=14e3, hmax=3700., hmin=2100.,
                      width_m=600., max_thick=180., ice_length_m=8e3):
    """Constant slope rectangular glacier over the HEF elevation range.

    Used for the dynamic scaling tests (the resampled HEF flowline is not a
    valid stand-alone dynamical glacier). nx sets the resolution.
    """
    from oggm.tests.funcs import dummy_constant_bed
    map_dx = length_m / nx
    fl = dummy_constant_bed(hmax=hmax, hmin=hmin, nx=nx, map_dx=map_dx,
                            widths=width_m / map_dx)[0]
    x = np.arange(nx) * map_dx
    thick = max_thick * np.sqrt(np.clip(1 - x / ice_length_m, 0, None))
    fl.thick = thick
    return fl


def calibrate(gdir, mb_model_class, ref_mb=-1000., **kwargs):
    """The calibration used in the tests (melt_f, brentq on 1970-2001)."""
    from oggm.core import massbalance
    if mb_model_class is not massbalance.MonthlyTIModel:
        kwargs.setdefault('ys', HEF_YS_CALIB)
        kwargs.setdefault('use_previous_mbs', True)
    df, mbmod = massbalance.mb_calibration_from_scalar_mb(
        gdir, observations_filesuffix='_bench',
        calibrate_param1='melt_f',
        ref_mb=ref_mb, ref_mb_years=(HEF_YS_CALIB, HEF_YE_CALIB),
        write_to_gdir=False, return_mb_model=True,
        mb_model_class=mb_model_class, overwrite_observations=True,
        **kwargs)
    return df, mbmod


def dynamic_run(gdir, mb_model_class, ys=HEF_YS_DYN, ye=HEF_YE_DYN,
                mb_elev_feedback='monthly', output_filesuffix='_bench',
                **kwargs):
    """run_from_climate_data on the HEF model flowlines."""
    from oggm import tasks
    return tasks.run_from_climate_data(
        gdir, mb_model_class=mb_model_class, ys=ys, ye=ye,
        mb_elev_feedback=mb_elev_feedback,
        output_filesuffix=output_filesuffix,
        **kwargs)


def single_fl_dynamic_run(gdir, fl, mb_model, ys=HEF_YS_DYN, ye=HEF_YE_DYN,
                          mb_elev_feedback='monthly'):
    """Dynamic run on one (resampled) flowline, returns the model."""
    from oggm.core.flowline import FluxBasedModel
    import copy
    model = FluxBasedModel([copy.deepcopy(fl)], mb_model=mb_model, y0=ys,
                           mb_elev_feedback=mb_elev_feedback)
    model.run_until(ye)
    return model


def timeit(func, repeat=3, number=1):
    """Best-of-`repeat` wall time of `number` calls (seconds per call)."""
    best = np.inf
    out = None
    for _ in range(repeat):
        t0 = time.perf_counter()
        for _ in range(number):
            out = func()
        best = min(best, (time.perf_counter() - t0) / number)
    return best, out


def sfc_class(cls=None, **kwargs):
    """partial of an SfcType class with default monthly settings."""
    from oggm.core.massbalance import SfcTypeTIModel, MonthlyTIModel
    cls = SfcTypeTIModel if cls is None else cls
    return partial(cls, mb_model_class=MonthlyTIModel, **kwargs)
