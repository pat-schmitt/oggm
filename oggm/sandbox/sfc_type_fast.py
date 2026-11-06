"""EXPERIMENTAL faster variants of :class:`oggm.core.massbalance.SfcTypeTIModel`.

This module is a prototype from a performance study (see
``benchmarks/sfctype/REPORT.md``). It is not part of the public API, it is not
tested beyond the benchmark scripts and it may change or disappear at any
time.

Two classes are provided:

:class:`FastSfcTypeTIModel`
    Exact re-implementation of the bucket update. It reproduces the reference
    model to floating point precision and combines several optimisations:

    - all climate steps requested with the same heights (e.g. the 12 months
      of a ``get_annual_mb`` call or the whole spinup) are collected and
      processed as one block;
    - the climate of a block is computed in one vectorised call;
    - the calendar (next bucket date, aging and storing flags, number of days
      per month) is precomputed once and cached;
    - aging is done by moving a window over a padded buffer instead of
      shifting the whole array (numpy kernel), or inside the per-point loop
      (Numba kernel);
    - no per-step copies, no np.clip, sanity checks only with
      ``debug_checks=True``;
    - optional Numba kernel (``kernel='numba'``, the default if Numba is
      installed) which loops over grid points and stops melting at the first
      bucket which is not fully melted. Numba is optional: without it the
      numpy kernel is used.

:class:`FixedHeightSfcTypeTIModel`
    APPROXIMATE variant. The buckets are not tracked per grid point but on
    fixed elevation levels (spacing ``dz``) covering the whole elevation range
    of the glacier. Because the levels never move, the full bucket evolution
    (spinup + all years until the end of the climate data) is computed once,
    in one block, and stored as a table (time x level). Mass balance requests
    are then a simple linear interpolation of the table rows to the requested
    heights. The table is shared between the flowlines of a glacier (and
    between calibration calls with the same parameters).

    Physics change: the snow/firn state belongs to elevation levels instead of
    grid points. A grid point which thins "inherits" the snow and firn of the
    lower level, i.e. snow and firn mass is not conserved per grid point.
"""
import warnings
from collections import OrderedDict

import numpy as np
import pandas as pd

from oggm.core.massbalance import SfcTypeTIModel, MonthlyTIModel
from oggm.utils import (floatyear_to_date, date_to_floatyear,
                        float_years_timeseries)
from oggm.exceptions import InvalidParamsError

try:
    import numba
    HAS_NUMBA = True
except ImportError:  # pragma: no cover
    numba = None
    HAS_NUMBA = False

__all__ = ['FastSfcTypeTIModel', 'FixedHeightSfcTypeTIModel', 'HAS_NUMBA']


# ---------------------------------------------------------------------------
# Calendar
# ---------------------------------------------------------------------------

_CALENDAR_CACHE = {}


class _Calendar:
    """Precomputed per-step date bookkeeping.

    For each climate step ``yr`` (as produced by float_years_timeseries) we
    store the date of the buckets after the step (``next``, the value of
    ``mb_buckets_year`` after the step), the month and day of this date (used
    for aging and storing) and the climate year/month of the step. The values
    are computed with exactly the same scalar functions as the reference
    model, so they are bitwise identical.
    """

    def __init__(self, y0, y1, climate_resolution):
        self.climate_resolution = climate_resolution
        if climate_resolution == 'annual':
            steps = list(range(y0, y1 + 1))
        elif climate_resolution == 'monthly':
            steps = list(float_years_timeseries(y0=y0, y1=y1,
                                                include_last_year=True))
        elif climate_resolution == 'daily':
            steps = list(float_years_timeseries(y0=y0, y1=y1,
                                                include_last_year=True,
                                                daily=True))
        else:
            raise NotImplementedError(climate_resolution)
        n = len(steps)
        self.steps = steps
        self.index = {yr: i for i, yr in enumerate(steps)}
        self.next = np.empty(n)
        self.bmonth = np.empty(n, dtype=np.int64)
        self.bday = np.empty(n, dtype=np.int64)
        self.cyear = np.empty(n, dtype=np.int64)
        self.cmonth = np.empty(n, dtype=np.int64)
        for i, yr in enumerate(steps):
            nxt, bm, bd, cy, cm = self._compute(yr)
            self.next[i] = nxt
            self.bmonth[i] = bm
            self.bday[i] = bd
            self.cyear[i] = cy
            self.cmonth[i] = cm
        self.next_list = self.next.tolist()
        self._days = {}

    def _compute(self, year):
        # same logic as in SfcTypeTIModel._apply_climate_step_and_aging...
        if self.climate_resolution == 'annual':
            return int(year) + 1, 1, 1, int(year), 1
        elif self.climate_resolution == 'monthly':
            cy, cm = floatyear_to_date(float(year))
            if cm == 12:
                by, bm = cy + 1, 1
            else:
                by, bm = cy, cm + 1
            return date_to_floatyear(by, bm), bm, 1, cy, cm
        else:
            cy, cm, cd = floatyear_to_date(float(year), return_day=True)
            nxt = date_to_floatyear(cy, cm, cd + 1)
            by, bm, bd = floatyear_to_date(nxt, return_day=True)
            return nxt, bm, bd, cy, cm

    def days_in_month(self, mbmod):
        """Days per month of each step, as computed by mbmod (the leap year
        convention of the climate model is used)."""
        key = bool(mbmod.use_leap_years)
        out = self._days.get(key)
        if out is None:
            out = np.array([mbmod.days_in_month(year=yr) for yr in self.steps],
                           dtype=np.float64)
            self._days[key] = out
        return out

    def sec_in_month(self, model, c):
        """Seconds of the month of step c, as computed by model."""
        key = ('sm', bool(model.use_leap_years))
        out = self._days.get(key)
        if out is None:
            out = [model.sec_in_month(year=yr) for yr in self.steps]
            self._days[key] = out
        return out[c]

    def sec_in_year(self, model, year):
        key = ('sy', bool(model.use_leap_years))
        out = self._days.get(key)
        if out is None:
            out = {}
            self._days[key] = out
        v = out.get(year)
        if v is None:
            v = model.sec_in_year(year)
            out[year] = v
        return v

    def month_index(self, year, month):
        """Index of the step (year, month) in a monthly calendar."""
        c = (int(year) - int(np.floor(self.steps[0]))) * 12 + int(month) - 1
        if c < 0 or c >= len(self.steps):
            return None
        return c


def get_calendar(y0, y1, climate_resolution):
    key = (int(y0), int(y1), climate_resolution)
    cal = _CALENDAR_CACHE.get(key)
    if cal is None:
        cal = _Calendar(*key)
        _CALENDAR_CACHE[key] = cal
    return cal


# ---------------------------------------------------------------------------
# Kernels
# ---------------------------------------------------------------------------

def _bucket_kernel_py(mb, prcpsol, tmelt, mf, nsnow, age, save,
                      cmb, icemb, smelt, fmelt, imelt,
                      record, rho, nsl, colm, colt, sig):
    """Loop version of the bucket update for several climate steps.

    Compiled with Numba if available (pure python would be far too slow and
    is never used, the numpy kernel is the fallback).

    mb: (n, nb + 1) buckets, modified in place
    prcpsol, tmelt: (nt, n)
    mf: (nb + 1,) melt factors, last one is ice
    age: (nt,) bool, aging after the step
    save: write cmb, icemb, smelt, fmelt, imelt (nt, n)
    record: additionally write the column mass and thickness after the step
        (colm, colt) and, if nsl[t] >= 0, the snowline signal (sig), used by
        the fixed height model
    """
    n, nbp1 = mb.shape
    nb = nbp1 - 1
    nt = prcpsol.shape[0]
    nage = 0
    for t in range(nt):
        if age[t]:
            nage += 1
    buf = np.empty(nbp1 + nage)
    for i in range(n):
        off = nage
        for j in range(nbp1):
            buf[off + j] = mb[i, j]
        for t in range(nt):
            ps = prcpsol[t, i]
            tm = tmelt[t, i]
            buf[off] += ps
            c = 0.0
            sm = 0.0
            fm = 0.0
            im = 0.0
            left = 0.0
            all_melted = True
            for j in range(nb):
                b = buf[off + j]
                c += b / mf[j]
                left = (c - tm) * mf[j]
                if left < 0:
                    # fully melted
                    if j < nsnow:
                        sm += b
                    else:
                        fm += b
                    buf[off + j] = 0.0
                else:
                    # partly melted (or untouched), all older buckets are
                    # untouched as the cumsum is monotonic
                    d = b - left
                    if d > 0:
                        if j < nsnow:
                            sm += d
                        else:
                            fm += d
                    buf[off + j] = left
                    all_melted = False
                    break
            if all_melted:
                im = -(left / mf[nb - 1]) * mf[nb]
                buf[off + nb] -= im
            if save:
                cmb[t, i] = ps - sm - fm - im
                icemb[t, i] = buf[off + nb]
                smelt[t, i] = sm
                fmelt[t, i] = fm
                imelt[t, i] = im
            buf[off + nb] = 0.0
            if age[t]:
                off -= 1
                buf[off] = 0.0
            if record:
                cm = 0.0
                ct = 0.0
                for j in range(nb):
                    v = buf[off + j]
                    cm += v
                    ct += v / rho[j]
                colm[t, i] = cm + buf[off + nb]
                colt[t, i] = ct
                k = nsl[t]
                if k >= 0:
                    s = 0.0
                    for j in range(min(k + 1, nb)):
                        s += buf[off + j]
                    if s <= 0:
                        s = -tm * mf[0]
                    sig[t, i] = s
        for j in range(nbp1):
            mb[i, j] = buf[off + j]


if HAS_NUMBA:
    _bucket_kernel_numba = numba.njit(cache=True, nogil=True)(_bucket_kernel_py)
else:  # pragma: no cover
    _bucket_kernel_numba = None


def _bucket_kernel_numpy(mb, prcpsol, tmelt, mf, nsnow, age, save,
                         cmb, icemb, smelt, fmelt, imelt,
                         record, rho, nsl, colm, colt, sig,
                         window_aging=True, debug_checks=False):
    """Vectorised (over grid points) version of the same bucket update.

    Same arguments as _bucket_kernel_py. Loops over the time steps.
    """
    n, nbp1 = mb.shape
    nb = nbp1 - 1
    nt = prcpsol.shape[0]
    mf_firn = mf[:-1]
    mf_last = mf_firn[-1]
    mf_ice = mf[-1]
    rows = np.arange(n)

    if window_aging:
        nage = int(np.count_nonzero(age))
        buf = np.empty((n, nbp1 + nage))
        off = nage
        buf[:, off:] = mb
        w = buf[:, off:off + nbp1]
    else:
        w = mb

    q = np.empty((n, nb))
    melted = np.empty((n, nb), dtype=bool)
    for t in range(nt):
        snow = w[:, :-1]
        ps = prcpsol[t]
        tm = tmelt[t]
        snow[:, 0] += ps
        # cumulative tfm needed to melt each bucket minus available tmelt,
        # back in kg m-2 (the same operations as the reference)
        np.divide(snow, mf_firn, out=q)
        np.cumsum(q, axis=1, out=q)
        q -= tm[:, None]
        q *= mf_firn
        np.less(q, 0, out=melted)
        nmelt = melted.sum(axis=1)
        partly = nmelt < nb
        r_p = rows[partly]
        b_p = nmelt[partly]
        left_p = q[r_p, b_p]
        if save:
            # melt of the fully melted buckets
            full = np.where(melted, snow, 0.)
            sm = full[:, :nsnow].sum(axis=1)
            fm = full[:, nsnow:].sum(axis=1)
            # melt of the partly melted bucket
            d = snow[r_p, b_p] - left_p
            d[d < 0] = 0.
            is_snow = b_p < nsnow
            sm[r_p[is_snow]] += d[is_snow]
            fm[r_p[~is_snow]] += d[~is_snow]
        snow[melted] = 0.
        snow[r_p, b_p] = left_p
        all_m = ~partly
        im = np.zeros(n)
        if np.any(all_m):
            im[all_m] = -(q[all_m, -1] / mf_last) * mf_ice
            w[all_m, -1] -= im[all_m]
        if save:
            cmb[t] = ps - sm - fm - im
            icemb[t] = w[:, -1]
            smelt[t] = sm
            fmelt[t] = fm
            imelt[t] = im
        w[:, -1] = 0.
        if age[t]:
            if debug_checks:
                if w[:, :-1].min() < 0:
                    raise ValueError("Bucket values must all be positive.")
            if window_aging:
                off -= 1
                w = buf[:, off:off + nbp1]
                w[:, 0] = 0.
            else:
                w[:, 1:] = w[:, :-1]
                w[:, 0] = 0.
        if record:
            colm[t] = w.sum(axis=1)
            colt[t] = (w[:, :-1] / rho).sum(axis=1)
            k = nsl[t]
            if k >= 0:
                s = w[:, :min(k + 1, nb)].sum(axis=1)
                nos = s <= 0
                s[nos] = -tm[nos] * mf[0]
                sig[t] = s
    if window_aging:
        mb[:] = w


# ---------------------------------------------------------------------------
# Exact fast model
# ---------------------------------------------------------------------------

class FastSfcTypeTIModel(SfcTypeTIModel):
    """EXPERIMENTAL: exact, faster implementation of SfcTypeTIModel.

    Same parameters as :class:`SfcTypeTIModel`, plus:

    Parameters
    ----------
    kernel : str, default 'auto'
        'numba', 'numpy' or 'auto' (numba if installed, numpy otherwise).
    batch_climate : bool, default True
        Compute the climate of all steps of a block in one vectorised call
        (only for MonthlyTIModel and monthly climate resolution, other cases
        fall back to one call per step).
    window_aging : bool, default True
        numpy kernel only: age by moving a window over a padded buffer
        instead of shifting the array.
    debug_checks : bool, default False
        numpy kernel only: check for negative buckets at aging (the reference
        model always does this).
    """

    def __init__(self, gdir, fl=None, kernel='auto', batch_climate=True,
                 window_aging=True, debug_checks=False, **kwargs):
        # fl is explicit as MultipleFlowlineMassBalance inspects the signature
        if kernel == 'auto':
            kernel = 'numba' if HAS_NUMBA else 'numpy'
        if kernel == 'numba' and not HAS_NUMBA:
            warnings.warn('Numba not available, using the numpy kernel.')
            kernel = 'numpy'
        if kernel not in ['numba', 'numpy']:
            raise InvalidParamsError(f'kernel: {kernel}')
        self.kernel = kernel
        self.batch_climate = batch_climate
        self.window_aging = window_aging
        self.debug_checks = debug_checks
        self._collecting = False
        self._pending = []
        self._calendar = None
        self._snowline_year_set = None
        # used by FixedHeightSfcTypeTIModel's level model
        self._record = False
        super().__init__(gdir, fl=fl, **kwargs)

    # --- calendar -----------------------------------------------------------
    def _get_calendar(self):
        # one calendar for the whole climate period, shared by all instances
        if self._calendar is None:
            self._calendar = get_calendar(self.mbmod.ys, self.ye,
                                          self.climate_resolution)
        return self._calendar

    def _cal_index(self, year):
        cal = self._get_calendar()
        i = cal.index.get(year)
        if i is None:
            raise KeyError(f'Timestep {year} not in the precomputed calendar.')
        return i

    # --- collect / flush ----------------------------------------------------
    def _apply_climate_step_and_aging_to_buckets(self, heights, year=None,
                                                 save_mbs=True):
        # Steps are collected and processed later in blocks (see _flush).
        # The bucket date is advanced already, such that the bookkeeping of
        # the calling methods works as in the reference model.
        self._pending.append((heights, year, save_mbs))
        cal = self._get_calendar()
        self.mb_buckets_year = cal.next_list[self._cal_index(year)]
        if not self._collecting:
            self._flush()

    def _flush(self):
        pending, self._pending = self._pending, []
        if not pending:
            return
        cal = self._get_calendar()
        i = 0
        while i < len(pending):
            h, _, save = pending[i]
            j = i + 1
            while (j < len(pending) and pending[j][0] is h and
                   pending[j][2] == save):
                j += 1
            years = [p[1] for p in pending[i:j]]
            self._process_block(np.asarray(h, dtype=np.float64), years, save)
            i = j
        assert self.mb_buckets_year == cal.next_list[
            self._cal_index(pending[-1][1])]

    def _run_until(self, heights, year, mb_resolution):
        self._collecting = True
        try:
            super()._run_until(heights, year, mb_resolution)
        except BaseException:
            self._pending = []
            raise
        finally:
            self._collecting = False
        self._flush()

    def _init_buckets(self):
        self._collecting = True
        try:
            super()._init_buckets()
        except BaseException:
            self._pending = []
            raise
        finally:
            self._collecting = False
        self._flush()
        self._snowline_year_set = None

    # --- lean getters (same results as the parent, less overhead) ---------
    def _lean_possible(self, add_climate):
        return (not add_climate and self.climate_resolution == 'monthly' and
                self.ys is not None)

    def _lean_run(self, heights, c_end):
        """Run all steps from the current bucket date up to calendar index
        c_end (inclusive)."""
        cal = self._get_calendar()
        if self.mb_buckets_year > cal.steps[c_end]:
            if not self.use_previous_mbs:
                return False  # let the parent raise
            return True
        c0 = cal.index.get(self.mb_buckets_year)
        if c0 is None:
            return False
        if c0 <= c_end:
            self._process_block(np.asarray(heights, dtype=np.float64),
                                cal.steps[c0:c_end + 1], True)
        return True

    def get_monthly_mb(self, heights, year, add_climate=False,
                       climatic_mb_or_ice_mb='climatic_mb',
                       include_mb_model_heights=False, **kwargs):
        self._ensure_buckets_initialized(year)
        cal = self._get_calendar()
        c = cal.index.get(year) if self._lean_possible(add_climate) else None
        if c is None or climatic_mb_or_ice_mb not in ['climatic_mb',
                                                      'ice_mb']:
            return super().get_monthly_mb(
                heights, year, add_climate=add_climate,
                climatic_mb_or_ice_mb=climatic_mb_or_ice_mb,
                include_mb_model_heights=include_mb_model_heights, **kwargs)
        if include_mb_model_heights:
            heights = heights + self.columns_thickness_m
        if not self._lean_run(heights, c):
            return super().get_monthly_mb(
                heights, year, climatic_mb_or_ice_mb=climatic_mb_or_ice_mb)
        i = self._year_to_index.get(year)
        if i is None:
            # let the parent raise the error
            self._stored_timestep_indices(year, mb_resolution='monthly')
        mbs = (self._climatic_mb if climatic_mb_or_ice_mb == 'climatic_mb'
               else self._ice_mb)
        monthly_mb = np.sum(mbs[[i]], axis=0)
        return ((monthly_mb / cal.sec_in_month(self, c) -
                 self.mbmod.bias / cal.sec_in_year(self, year)) /
                self.mbmod.ice_density)

    def get_annual_mb(self, heights, year, add_climate=False,
                      climatic_mb_or_ice_mb='climatic_mb',
                      include_mb_model_heights=False, **kwargs):
        self._ensure_buckets_initialized(year)
        cal = self._get_calendar()
        c = None
        if self._lean_possible(add_climate) and year == np.floor(year):
            c = cal.month_index(year, 12)
        if c is None or climatic_mb_or_ice_mb not in ['climatic_mb',
                                                      'ice_mb']:
            return super().get_annual_mb(
                heights, year, add_climate=add_climate,
                climatic_mb_or_ice_mb=climatic_mb_or_ice_mb,
                include_mb_model_heights=include_mb_model_heights, **kwargs)
        if include_mb_model_heights:
            heights = heights + self.columns_thickness_m
        if not self._lean_run(heights, c):
            return super().get_annual_mb(
                heights, year, climatic_mb_or_ice_mb=climatic_mb_or_ice_mb)
        idx = self._stored_timestep_indices(year, mb_resolution='annual')
        mbs = (self._climatic_mb if climatic_mb_or_ice_mb == 'climatic_mb'
               else self._ice_mb)
        annual_mb = np.sum(mbs[idx], axis=0)
        return ((annual_mb - self.mbmod.bias) / cal.sec_in_year(self, year) /
                self.mbmod.ice_density)

    # --- climate ------------------------------------------------------------
    def _block_climate(self, heights, years, cidx):
        """tmelt and prcpsol, shape (nt, n)."""
        mbmod = self.mbmod
        cal = self._get_calendar()
        if (self.batch_climate and self.climate_resolution == 'monthly' and
                type(mbmod) is MonthlyTIModel):
            if len(cidx) == 1:
                # lean path for single steps (dynamics with monthly feedback)
                c = int(cidx[0])
                vy = mbmod.validate_year(year=int(cal.cyear[c]))
                cm = int(cal.cmonth[c])
                pok = (vy - int(mbmod.years[0])) * 12 + (cm - 1)
                if mbmod.years[pok] != vy or mbmod.months[pok] != cm:
                    raise RuntimeError('Unexpected climate data layout.')
                temp = mbmod.temp[pok] + mbmod.grad[pok] * (heights -
                                                            mbmod.ref_hgt)
                tmelt = mbmod._get_tempformelt(temp)
                prcp = np.ones(len(heights)) * mbmod.prcp[pok]
                prcpsol = mbmod._get_prcpsol(prcp, temp)
                tmelt *= cal.days_in_month(mbmod)[c]
                return tmelt[None, :], prcpsol[None, :]
            cy = cal.cyear[cidx]
            cm = cal.cmonth[cidx]
            uy = np.unique(cy)
            vy = np.array([mbmod.validate_year(year=int(y)) for y in uy])
            vy = vy[np.searchsorted(uy, cy)]
            pok = (vy - mbmod.years[0]) * 12 + (cm - 1)
            if not (np.all(mbmod.years[pok] == vy) and
                    np.all(mbmod.months[pok] == cm)):
                raise RuntimeError('Unexpected climate data layout.')
            itemp = mbmod.temp[pok][:, None]
            igrad = mbmod.grad[pok][:, None]
            iprcp = mbmod.prcp[pok][:, None]
            temp = itemp + igrad * (heights[None, :] - mbmod.ref_hgt)
            tmelt = mbmod._get_tempformelt(temp)
            prcp = np.ones(len(heights))[None, :] * iprcp
            prcpsol = mbmod._get_prcpsol(prcp, temp)
            tmelt *= cal.days_in_month(mbmod)[cidx][:, None]
            return tmelt, prcpsol
        # fallback, one call per step
        func = {'annual': mbmod.get_annual_climate,
                'monthly': mbmod.get_monthly_climate,
                'daily': getattr(mbmod, 'get_daily_climate', None)
                }[self.climate_resolution]
        tm = np.empty((len(years), len(heights)))
        ps = np.empty((len(years), len(heights)))
        for k, yr in enumerate(years):
            _, tm[k], _, ps[k] = func(heights, yr)
        return tm, ps

    # --- hooks (storing, snowline) ------------------------------------------
    def _needs_hook(self, cidx):
        cal = self._get_calendar()
        need = np.zeros(len(cidx), dtype=bool)
        if self.store_buckets:
            if self.store_buckets_dates is not None:
                dates = set(self.store_buckets_dates)
                need |= np.array([cal.next_list[c] in dates for c in cidx])
            elif self.store_buckets == 'annual':
                need |= (cal.bmonth[cidx] == 1) & (cal.bday[cidx] == 1)
            elif self.store_buckets == 'monthly':
                need |= cal.bday[cidx] == 1
            elif self.store_buckets == 'daily':
                need[:] = True
            else:
                raise NotImplementedError(f"store_buckets: {self.store_buckets}")
        if self.store_snowline:
            need[:] = True
        return need

    def _snowline_years(self):
        if (self._snowline_year_set is None or
                len(self._snowline_year_set) != len(self._snowline_year)):
            self._snowline_year_set = set(self._snowline_year)
        return self._snowline_year_set

    def _hook(self, heights, c, buckets):
        """Storing and snowline after step with calendar index c, same logic
        as the reference model. ``buckets`` is the (n, nb+1) state."""
        cal = self._get_calendar()
        mb_year = cal.next_list[c]
        bm = int(cal.bmonth[c])
        bd = int(cal.bday[c])
        if self.store_buckets:
            store = False
            if self.store_buckets_dates is not None:
                store = mb_year in self.store_buckets_dates
            elif self.store_buckets == 'annual':
                store = bm == 1 and bd == 1
            elif self.store_buckets == 'monthly':
                store = bd == 1
            elif self.store_buckets == 'daily':
                store = True
            if store:
                self.mb_buckets_stored[mb_year] = pd.DataFrame(
                    buckets[:, :-1].copy(),
                    index=self.buckets_grid_point_label,
                    columns=self.buckets[:-1], )
        if self.store_snowline:
            if mb_year not in self._snowline_years():
                number_buckets = bm + self._snowline_start_month
                not_melted = np.any(buckets[:, :number_buckets + 1] > 0, axis=1)
                self._append_snowline(not_melted, heights, mb_year)

    def _append_snowline(self, not_melted_layers, heights, mb_year):
        if not np.any(not_melted_layers):
            self._snowline.append(np.inf)
            if np.inf not in self.snowline_inf_values:
                self.snowline_inf_values[np.inf] = np.max(heights) + 50
        elif np.all(not_melted_layers):
            self._snowline.append(-np.inf)
            if -np.inf not in self.snowline_inf_values:
                self.snowline_inf_values[-np.inf] = np.min(heights) - 50
        else:
            self._snowline.append(heights[not_melted_layers][-1])
        self._snowline_year.append(mb_year)
        self._snowline_years().add(mb_year)

    # --- the block ----------------------------------------------------------
    def _check_heights(self, heights):
        if len(heights) != len(self.buckets_grid_point_label):
            raise InvalidParamsError(
                f'Number of heights ({len(heights)}) does not match number '
                'of grid points provided on initialization '
                f'({len(self.buckets_grid_point_label)}).')

    def _aging_flags(self, cidx):
        cal = self._get_calendar()
        if self.aging_frequency == 'annual':
            if self.climate_resolution == 'annual':
                return np.ones(len(cidx), dtype=bool)
            return (cal.bmonth[cidx] == 1) & (cal.bday[cidx] == 1)
        if self.climate_resolution == 'monthly':
            return np.ones(len(cidx), dtype=bool)
        return cal.bday[cidx] == 1

    def _snowline_nr_buckets(self, cidx):
        """For the fixed height model: number of buckets of the snowline."""
        cal = self._get_calendar()
        if not self.store_snowline:
            return np.full(len(cidx), -1, dtype=np.int64)
        return (cal.bmonth[cidx] + self._snowline_start_month).astype(np.int64)

    def _run_kernel(self, tmelt, prcpsol, age, save, out, rec):
        mf = np.ascontiguousarray(self._melt_f_buckets_np[0])
        rho = np.ascontiguousarray(self._density_buckets_np[0, :-1])
        args = (self.mb_buckets_np, prcpsol, tmelt, mf,
                int(self._nr_snow_buckets), age, save) + out + \
               (self._record, rho) + rec
        if self.kernel == 'numba':
            _bucket_kernel_numba(*args)
        else:
            _bucket_kernel_numpy(*args, window_aging=self.window_aging,
                                 debug_checks=self.debug_checks)

    def _process_block(self, heights, years, save):
        self._check_heights(heights)
        cidx = np.array([self._cal_index(y) for y in years], dtype=np.int64)
        nt = len(cidx)
        n = len(heights)
        tmelt, prcpsol = self._block_climate(heights, years, cidx)
        tmelt = np.ascontiguousarray(tmelt)
        prcpsol = np.ascontiguousarray(prcpsol)
        age = self._aging_flags(cidx)
        need = self._needs_hook(cidx)
        c0 = self._current_index
        if save:
            sl = slice(c0, c0 + nt)
            out = (self._climatic_mb[sl], self._ice_mb[sl],
                   self._snow_melt[sl], self._firn_melt[sl],
                   self._ice_melt[sl])
        else:
            dummy = np.empty((0, 0))
            out = (dummy,) * 5
        if self._record:
            rec_full = (self._snowline_nr_buckets(cidx),
                        np.empty((nt, n)), np.empty((nt, n)),
                        np.empty((nt, n)))
        # segments end at steps which need a hook
        ends = list(np.nonzero(need)[0] + 1)
        if not ends or ends[-1] != nt:
            ends.append(nt)
        s = 0
        for e in ends:
            seg = slice(s, e)
            o = tuple(a[seg] for a in out) if save else out
            if self._record:
                rec = tuple(a[seg] for a in rec_full)
            else:
                rec = (np.empty(0, dtype=np.int64), np.empty((0, 0)),
                       np.empty((0, 0)), np.empty((0, 0)))
            self._run_kernel(tmelt[seg], prcpsol[seg], age[seg], save, o, rec)
            if need[e - 1]:
                self._hook(heights, cidx[e - 1], self.mb_buckets_np)
            s = e
        if self._record:
            self._record_block(cidx, save, rec_full)
        if save:
            self._mb_heights[c0:c0 + nt] = heights
            for k, yr in enumerate(years):
                self._year_to_index[yr] = c0 + k
            self._current_index += nt
        self.mb_buckets_year = self._get_calendar().next_list[cidx[-1]]

    def _record_block(self, cidx, save, rec):
        pass


# ---------------------------------------------------------------------------
# Fixed height model (approximate)
# ---------------------------------------------------------------------------

_TABLE_CACHE = OrderedDict()
_TABLE_CACHE_SIZE = 16
# level models are reused (their construction reads the climate file and the
# settings), they get the climate model of the caller at each table build
_LEVEL_MODEL_CACHE = OrderedDict()


class _LevelModel(FastSfcTypeTIModel):
    """Fast model on fixed levels which records per step column mass and
    thickness and the snowline signal."""

    def __init__(self, gdir, **kwargs):
        super().__init__(gdir, **kwargs)
        self._record = True
        self.rec = {}

    def _init_buckets(self):
        self.rec = {'colm': [], 'colt': [], 'sig': [], 'cidx': []}
        super()._init_buckets()

    def _needs_hook(self, cidx):
        # the snowline is derived from the recorded signal, only storing
        ss = self.store_snowline
        self.store_snowline = False
        try:
            return super()._needs_hook(cidx)
        finally:
            self.store_snowline = ss

    def _hook(self, heights, c, buckets):
        ss = self.store_snowline
        self.store_snowline = False
        try:
            return super()._hook(heights, c, buckets)
        finally:
            self.store_snowline = ss

    def _record_block(self, cidx, save, rec):
        _, colm, colt, sig = rec
        self.rec['cidx'].append(cidx)
        self.rec['colm'].append(colm)
        self.rec['colt'].append(colt)
        self.rec['sig'].append(sig)


class _Table:
    pass


class FixedHeightSfcTypeTIModel(FastSfcTypeTIModel):
    """EXPERIMENTAL and APPROXIMATE: SfcTypeTIModel on fixed height levels.

    Same parameters as :class:`SfcTypeTIModel` (and
    :class:`FastSfcTypeTIModel`), plus:

    Parameters
    ----------
    dz : float, default 25
        Spacing of the fixed elevation levels (m).
    z_range : (float, float), default None
        Elevation range covered by the levels. Default: from the lowest bed
        (model flowlines) or surface (inversion flowlines) of all flowlines of
        the glacier minus ``z_margin[0]`` to the highest surface plus
        ``z_margin[1]``. Heights outside the range are clamped.
    z_margin : (float, float), default (100, 300)
    share_table : bool, default True
        Share the precomputed table between instances with the same glacier,
        parameters and levels (e.g. the flowlines of a glacier).
    add_mb_model_heights : bool, default True
        If False, ``include_mb_model_heights=True`` (the default of the
        flowline models) is ignored, i.e. the table is always read at the
        ice surface height. On HEF honouring it (True) reproduces the
        reference much better (volume within 0.4% instead of 1.8%), because
        the reference evaluates the climate on top of the snow/firn column.
    """

    def __init__(self, gdir, fl=None, dz=25., z_range=None,
                 z_margin=(100., 300.), share_table=True,
                 add_mb_model_heights=True, **kwargs):
        self.add_mb_model_heights = add_mb_model_heights
        if kwargs.get('spinup_buckets') is not None:
            raise NotImplementedError('spinup_buckets are not supported by '
                                      'FixedHeightSfcTypeTIModel.')
        self.dz = float(dz)
        self.share_table = share_table
        self._z_range_in = z_range
        self.z_margin = z_margin
        self._table = None
        self._level_model = None
        self._last_row = None
        self._last_heights = None
        self._level_kwargs = dict(kwargs)
        self._gdir_for_levels = gdir
        self.n_clamped = 0
        self.z_levels = None
        super().__init__(gdir, fl=fl, **kwargs)
        if self.z_levels is None:
            self.z_levels = self._make_levels(gdir)

    def _make_levels(self, gdir):
        if self._z_range_in is not None:
            zmin, zmax = self._z_range_in
        else:
            zmin, zmax = np.inf, -np.inf
            try:
                fls = gdir.read_store('model_flowlines')
                for fl in fls:
                    zmin = min(zmin, np.min(fl.bed_h))
                    zmax = max(zmax, np.max(fl.surface_h))
                zmin -= self.z_margin[0]
            except Exception:
                fls = gdir.read_store('inversion_flowlines')
                for fl in fls:
                    zmin = min(zmin, np.min(fl.surface_h))
                    zmax = max(zmax, np.max(fl.surface_h))
                zmin -= self.z_margin[0] + 300.
            zmin = min(zmin, np.min(self.spinup_heights))
            zmax = max(zmax, np.max(self.spinup_heights)) + self.z_margin[1]
        z0 = np.floor(zmin / self.dz) * self.dz
        z1 = np.ceil(zmax / self.dz) * self.dz
        return np.arange(z0, z1 + self.dz / 2, self.dz)

    def get_monthly_mb(self, heights, year, include_mb_model_heights=False,
                       **kwargs):
        return super().get_monthly_mb(
            heights, year, include_mb_model_heights=(
                include_mb_model_heights and self.add_mb_model_heights),
            **kwargs)

    def get_annual_mb(self, heights, year, include_mb_model_heights=False,
                      **kwargs):
        return super().get_annual_mb(
            heights, year, include_mb_model_heights=(
                include_mb_model_heights and self.add_mb_model_heights),
            **kwargs)

    # buckets per grid point are not tracked in this model
    @property
    def mb_buckets_np(self):
        raise NotImplementedError('FixedHeightSfcTypeTIModel does not track '
                                  'the buckets per grid point (use '
                                  'store_buckets, or level_buckets_np).')

    @mb_buckets_np.setter
    def mb_buckets_np(self, value):
        pass

    @property
    def level_buckets_np(self):
        """Current buckets on the fixed levels (end of the table period)."""
        return self._level_model.mb_buckets_np

    def save_to_file(self, *args, **kwargs):
        raise NotImplementedError('Not supported by the prototype.')

    # --- levels and table ---------------------------------------------------
    def _table_key(self):
        m = self.mbmod
        return (self.gdir.dir, m.filename, m.input_filesuffix, float(m.melt_f),
                float(m.prcp_fac), float(m.temp_bias), float(m.temp_melt),
                bool(getattr(m, 'repeat', False)), int(m.ys), int(m.ye),
                self.ys, float(self.z_levels[0]), self.dz,
                len(self.z_levels), self.climate_resolution,
                self.aging_frequency, self.spinup_years, self.save_spinup_mbs,
                self.melt_f_ratio, self.melt_f_change, self.tau_e,
                self.snow_density, self.density_change,
                str(self.store_buckets),
                None if self.store_buckets_dates is None else
                tuple(self.store_buckets_dates),
                self.store_snowline,
                getattr(self, '_snowline_start_month', None),
                self.kernel, type(m).__name__)

    def _get_level_model(self):
        lkey = (self.gdir.dir, self.settings_filesuffix,
                float(self.z_levels[0]), self.dz, len(self.z_levels),
                self.kernel, self.batch_climate, self.window_aging,
                repr(sorted((k, repr(v)) for k, v in
                            self._level_kwargs.items())))
        if self._level_model is None and self.share_table:
            self._level_model = _LEVEL_MODEL_CACHE.get(lkey)
        if self._level_model is None:
            kw = dict(self._level_kwargs)
            for k in ['fl', 'use_main_fl_from', 'hbins', 'ys',
                      'use_previous_mbs', 'kernel', 'batch_climate',
                      'window_aging', 'debug_checks', 'save_spinup_mbs']:
                kw.pop(k, None)
            lm = _LevelModel(self._gdir_for_levels, hbins=self.z_levels,
                             kernel=self.kernel,
                             batch_climate=self.batch_climate,
                             window_aging=self.window_aging,
                             save_spinup_mbs=True, **kw)
            self._level_model = lm
            if self.share_table:
                _LEVEL_MODEL_CACHE[lkey] = lm
                while len(_LEVEL_MODEL_CACHE) > _TABLE_CACHE_SIZE:
                    _LEVEL_MODEL_CACHE.popitem(last=False)
        lm = self._level_model
        # share the climate model (and thus the parameters)
        lm.mbmod = self.mbmod
        lm.set_melt_f_buckets()
        return lm

    def _ensure_table(self):
        if self._table is not None:
            return
        key = self._table_key()
        if self.share_table and key in _TABLE_CACHE:
            self._table = _TABLE_CACHE[key]
            _TABLE_CACHE.move_to_end(key)
            return
        lm = self._get_level_model()
        lm.ys = self.ys
        lm._ys_is_dynamic = False
        lm._init_buckets()
        # the whole period until the end of the climate data in one block
        lm._run_until(self.z_levels, self.ye, 'annual')
        cal = lm._get_calendar()
        T = _Table()
        nt = lm._current_index
        T.year_to_row = dict(lm._year_to_index)
        T.cmb = lm._climatic_mb[:nt].copy()
        T.icemb = lm._ice_mb[:nt].copy()
        T.smelt = lm._snow_melt[:nt].copy()
        T.fmelt = lm._firn_melt[:nt].copy()
        T.imelt = lm._ice_melt[:nt].copy()
        # recorded state after each step (also for spinup steps)
        cidx = np.concatenate(lm.rec['cidx'])
        T.cidx_to_rec = {int(c): k for k, c in enumerate(cidx)}
        T.colm = np.concatenate(lm.rec['colm'])
        T.colt = np.concatenate(lm.rec['colt'])
        T.sig = np.concatenate(lm.rec['sig'])
        T.stored = dict(lm.mb_buckets_stored) if lm.store_buckets else {}
        T.end_buckets = lm.mb_buckets_np.copy()
        T.calendar = cal
        self._table = T
        if self.share_table:
            _TABLE_CACHE[key] = T
            while len(_TABLE_CACHE) > _TABLE_CACHE_SIZE:
                _TABLE_CACHE.popitem(last=False)

    def _weights(self, heights):
        z = self.z_levels
        f = (heights - z[0]) / self.dz
        i0 = np.floor(f).astype(np.int64)
        out = (i0 < 0) | (i0 > len(z) - 2)
        if np.any(out):
            # clamp (also catches the exact upper boundary)
            self.n_clamped += int(np.count_nonzero(
                (heights < z[0]) | (heights > z[-1])))
            i0 = np.clip(i0, 0, len(z) - 2)
        w = np.clip(f - i0, 0., 1.)
        return i0, w

    @staticmethod
    def _interp(A, i0, w):
        """A: (nt, nlev) -> (nt, n)."""
        return A[:, i0] * (1. - w) + A[:, i0 + 1] * w

    # --- state --------------------------------------------------------------
    def _init_buckets(self):
        if self.z_levels is None:
            # called from the parent __init__, levels not ready yet
            self.z_levels = self._make_levels(self._gdir_for_levels)
        self._table = None
        self._last_row = None
        n = len(self.buckets_grid_point_label)
        first_year = self.ys
        if self.save_spinup_mbs:
            first_year -= self.spinup_years
        if self.climate_resolution == 'annual':
            self.nr_timesteps = len(range(first_year, self.ye + 1))
        else:
            self.nr_timesteps = len(float_years_timeseries(
                y0=first_year, y1=self.ye, include_last_year=True,
                daily=self.climate_resolution == 'daily'))
        shape = (self.nr_timesteps, n)
        self._climatic_mb = np.empty(shape)
        self._ice_mb = np.empty(shape)
        self._mb_heights = np.empty(shape)
        self._snow_melt = np.empty(shape)
        self._firn_melt = np.empty(shape)
        self._ice_melt = np.empty(shape)
        self._year_to_index = {}
        self._current_index = 0
        self._snowline_year_set = None
        if self.store_buckets:
            self.mb_buckets_stored = {}
        spinup_start_year = self.ys - self.spinup_years
        if self.climate_resolution == 'annual':
            steps = range(spinup_start_year, self.ys)
        else:
            steps = float_years_timeseries(
                y0=spinup_start_year, y1=self.ys,
                daily=self.climate_resolution == 'daily')[:-1]
        self._collecting = True
        try:
            for yr in steps:
                self._apply_climate_step_and_aging_to_buckets(
                    heights=self.spinup_heights, year=yr,
                    save_mbs=self.save_spinup_mbs)
        finally:
            self._collecting = False
        self._flush()
        assert self.mb_buckets_year == self.ys

    def _process_block(self, heights, years, save):
        self._check_heights(heights)
        self._ensure_table()
        T = self._table
        cal = self._get_calendar()
        cidx = [self._cal_index(y) for y in years]
        # the level calendar can start earlier, map via the float years
        rec_rows = np.array([T.cidx_to_rec[c] for c in cidx])
        i0, w = self._weights(heights)
        nt = len(years)
        if save:
            rows = np.array([T.year_to_row[y] for y in years])
            c0 = self._current_index
            sl = slice(c0, c0 + nt)
            for name, A in [('_climatic_mb', T.cmb), ('_ice_mb', T.icemb),
                            ('_snow_melt', T.smelt), ('_firn_melt', T.fmelt),
                            ('_ice_melt', T.imelt)]:
                getattr(self, name)[sl] = self._interp(A[rows], i0, w)
            self._mb_heights[sl] = heights
            for k, yr in enumerate(years):
                self._year_to_index[yr] = c0 + k
            self._current_index += nt
        # hooks
        if self.store_buckets or self.store_snowline:
            for k, yr in enumerate(years):
                nxt = cal.next_list[cidx[k]]
                if self.store_buckets and nxt in T.stored:
                    lev = T.stored[nxt].values
                    self.mb_buckets_stored[nxt] = pd.DataFrame(
                        lev[i0] * (1 - w[:, None]) + lev[i0 + 1] * w[:, None],
                        index=self.buckets_grid_point_label,
                        columns=self.buckets[:-1])
                if self.store_snowline and nxt not in self._snowline_years():
                    s = self._interp(T.sig[rec_rows[k:k + 1]], i0, w)[0]
                    self._append_snowline(s > 0, heights, nxt)
        self._last_row = rec_rows[-1]
        self._last_heights = heights
        self.mb_buckets_year = cal.next_list[cidx[-1]]

    def _current_level_values(self, A):
        if self._last_row is None:
            return None
        i0, w = self._weights(self._last_heights)
        return self._interp(A[self._last_row:self._last_row + 1], i0, w)[0]

    @property
    def columns_thickness_m(self):
        if self._table is None or self._last_row is None:
            return np.zeros(len(self.buckets_grid_point_label))
        return self._current_level_values(self._table.colt)

    @property
    def columns_mass_kg_per_sqm(self):
        if self._table is None or self._last_row is None:
            return np.zeros(len(self.buckets_grid_point_label))
        return self._current_level_values(self._table.colm)

    @property
    def buckets_thickness_m(self):
        raise NotImplementedError('Use store_buckets with '
                                  'FixedHeightSfcTypeTIModel.')


def clear_caches(calendar=False):
    """Clear the table cache (and the calendar cache), e.g. for benchmarks."""
    if calendar:
        _CALENDAR_CACHE.clear()
        _LEVEL_MODEL_CACHE.clear()
    _TABLE_CACHE.clear()
