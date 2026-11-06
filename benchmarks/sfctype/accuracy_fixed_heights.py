"""Accuracy of FixedHeightSfcTypeTIModel vs. the reference (EXPERIMENTAL).

HEF, HISTALP monthly climate. For several level spacings dz:

static   main flowline, fixed heights (calibration-like), 1970-2002, monthly
         calls, buckets stored monthly, snowline stored. Errors of monthly and
         annual climatic_mb / ice_mb, melt split, buckets, column thickness and
         density, snowline, error by elevation.
calib    calibrated melt_f (all flowlines)
dynamic  run_from_climate_data 1960-2003 (monthly feedback): volume, area,
         length

Writes results/accuracy.json and figures to results/figures/.
Usage: python benchmarks/sfctype/accuracy_fixed_heights.py
"""
import os
import sys
import json
import time

import numpy as np
import xarray as xr

sys.path.insert(0, os.path.dirname(__file__))
import common  # noqa: E402
import plotstyle  # noqa: E402
from plotstyle import plt, REF, SERIES  # noqa: E402

from oggm.utils import date_to_floatyear  # noqa: E402
from oggm.core.massbalance import SfcTypeTIModel, MonthlyTIModel  # noqa: E402
from oggm.sandbox import sfc_type_fast as sf  # noqa: E402
from oggm.sandbox.sfc_type_fast import FixedHeightSfcTypeTIModel  # noqa

OUT = os.path.join(os.path.dirname(__file__), 'results')
FIG = os.path.join(OUT, 'figures')
DZS = [10, 25, 50, 100]
Y0, Y1 = 1970, 2002


def stats(a, b):
    d = np.asarray(a) - np.asarray(b)
    d = d[np.isfinite(d)]
    return {'bias': float(np.mean(d)), 'rmse': float(np.sqrt(np.mean(d ** 2))),
            'maxabs': float(np.max(np.abs(d)))}


def static_run(cls, gdir, fl, **kw):
    sf.clear_caches()
    mb = cls(gdir, fl=fl, mb_model_class=MonthlyTIModel, ys=Y0,
             store_buckets='monthly', store_snowline=True, **kw)
    for y in range(Y0, Y1 + 1):
        for m in range(1, 13):
            mb.get_monthly_mb(fl.surface_h, float(date_to_floatyear(y, m)))
    return mb


def annual(mb, arr):
    """(n_years, n) annual sums of a stored array, years Y0..Y1."""
    return np.array([np.sum(arr[mb._stored_timestep_indices(
        y, mb_resolution='annual')], axis=0) for y in range(Y0, Y1 + 1)])


def monthly(mb, arr):
    idx = [mb._year_to_index[float(date_to_floatyear(y, m))]
           for y in range(Y0, Y1 + 1) for m in range(1, 13)]
    return arr[idx]


def static_metrics(ref, new, fl, glacier):
    w = fl.widths_m[glacier]
    res = {}
    for name in ['_climatic_mb', '_ice_mb', '_snow_melt', '_firn_melt',
                 '_ice_melt']:
        ra, na = annual(ref, getattr(ref, name)), annual(new, getattr(new, name))
        rm, nm = monthly(ref, getattr(ref, name)), monthly(new,
                                                           getattr(new, name))
        res[name[1:]] = {
            'monthly': stats(nm[:, glacier], rm[:, glacier]),
            'annual': stats(na[:, glacier], ra[:, glacier]),
            'annual_glacier_wide': stats(
                np.average(na[:, glacier], weights=w, axis=1),
                np.average(ra[:, glacier], weights=w, axis=1)),
        }
    # buckets (monthly stored)
    rho = ref._density_buckets_np[0, :-1]
    keys = sorted(ref.mb_buckets_stored)
    rb = np.array([ref.mb_buckets_stored[k].values[glacier] for k in keys])
    nb = np.array([new.mb_buckets_stored[k].values[glacier] for k in keys])
    res['bucket_mass_kg_m2'] = stats(nb, rb)
    res['bucket_thickness_m'] = stats(nb / rho, rb / rho)
    rt, nt = (rb / rho).sum(axis=2), (nb / rho).sum(axis=2)
    rm, nm = rb.sum(axis=2), nb.sum(axis=2)
    res['column_thickness_m'] = stats(nt, rt)
    ok = (rt > 0.05) & (nt > 0.05)
    res['column_mean_density_kg_m3'] = stats((nm / np.where(ok, nt, 1))[ok],
                                             (rm / np.where(ok, rt, 1))[ok])

    # snowline: replace +-inf by the height values used by the model
    def sl(mb):
        s = mb.snowline.copy()
        s[s == np.inf] = np.max(fl.surface_h) + 50
        s[s == -np.inf] = np.min(fl.surface_h) - 50
        return s
    rs, ns = sl(ref), sl(new)
    assert np.all(ref.snowline_year == new.snowline_year)
    res['snowline_m'] = stats(ns, rs)
    res['snowline_m']['frac_identical'] = float(np.mean(rs == ns))
    # per elevation (annual climatic mb)
    ra = annual(ref, ref._climatic_mb)
    na = annual(new, new._climatic_mb)
    res['annual_cmb_rmse_by_point'] = np.sqrt(
        np.mean((na - ra) ** 2, axis=0)).tolist()
    return res


def main():
    os.makedirs(FIG, exist_ok=True)
    plotstyle.setup()
    gdir = common.get_hef_gdir()
    fl = common.main_model_flowline(gdir)
    glacier = fl.thick > 0
    out = {'dzs': DZS, 'heights': fl.surface_h.tolist(),
           'glacier': glacier.tolist(), 'static': {}, 'calib': {},
           'dynamic': {}}

    # ---- static
    ref = static_run(SfcTypeTIModel, gdir, fl)
    runs = {}
    for dz in DZS:
        new = static_run(FixedHeightSfcTypeTIModel, gdir, fl, dz=dz)
        runs[dz] = new
        out['static'][dz] = static_metrics(ref, new, fl, glacier)
        s = out['static'][dz]
        print(f"dz={dz:4d}: annual cmb rmse "
              f"{s['climatic_mb']['annual']['rmse']:7.1f} kg m-2, glacier-"
              f"wide bias {s['climatic_mb']['annual_glacier_wide']['bias']:6.1f}"
              f" rmse {s['climatic_mb']['annual_glacier_wide']['rmse']:6.1f}; "
              f"snowline rmse {s['snowline_m']['rmse']:6.1f} m; column thick "
              f"rmse {s['column_thickness_m']['rmse']:.3f} m", flush=True)

    # ---- calibration
    t0 = time.perf_counter()
    df_ref, _ = common.calibrate(gdir, SfcTypeTIModel)
    out['calib']['reference'] = {'melt_f': df_ref['melt_f'],
                                 's': time.perf_counter() - t0}
    for dz in DZS:
        sf.clear_caches()
        t0 = time.perf_counter()
        df, _ = common.calibrate(gdir, common.sfc_class(
            FixedHeightSfcTypeTIModel, dz=dz))
        out['calib'][dz] = {'melt_f': df['melt_f'],
                            's': time.perf_counter() - t0}
        print(f"calib dz={dz}: melt_f {df['melt_f']:.4f} "
              f"(ref {df_ref['melt_f']:.4f})", flush=True)

    # ---- dynamic
    def diag(cls_partial, suffix):
        sf.clear_caches()
        common.dynamic_run(gdir, cls_partial, output_filesuffix=suffix)
        with xr.open_dataset(gdir.get_filepath('model_diagnostics',
                                               filesuffix=suffix)) as ds:
            return ds[['volume_m3', 'area_m2', 'length_m']].load()
    dref = diag(SfcTypeTIModel, '_acc_ref')
    dyn = {}
    for dz in DZS:
        d = diag(common.sfc_class(FixedHeightSfcTypeTIModel, dz=dz),
                 f'_acc_{dz}')
        dyn[dz] = d
        r = {}
        for v in ['volume_m3', 'area_m2', 'length_m']:
            rel = (d[v].values - dref[v].values) / dref[v].values
            r[v] = {'max_abs_rel': float(np.max(np.abs(rel))),
                    'end_rel': float(rel[-1])}
        out['dynamic'][dz] = r
        print(f"dyn dz={dz}: volume max rel diff "
              f"{100 * r['volume_m3']['max_abs_rel']:.2f}%, end "
              f"{100 * r['volume_m3']['end_rel']:.2f}%; length end "
              f"{100 * r['length_m']['end_rel']:.2f}%", flush=True)
    # same with the mb model heights added to the lookup height
    dyn_add = {}
    for dz in [25]:
        d = diag(common.sfc_class(FixedHeightSfcTypeTIModel, dz=dz,
                                  add_mb_model_heights=False),
                 f'_acc_add_{dz}')
        dyn_add[dz] = d
        rel = (d.volume_m3.values - dref.volume_m3.values) / \
            dref.volume_m3.values
        out['dynamic'][f'{dz}_ignore_mb_model_heights'] = {'volume_m3': {
            'max_abs_rel': float(np.max(np.abs(rel))),
            'end_rel': float(rel[-1])}}
        print(f"dyn dz={dz} ignore mb model heights: volume max rel diff "
              f"{100 * np.max(np.abs(rel)):.2f}%", flush=True)
    # dynamic run of the reference without mb model heights (to show the
    # sensitivity to this option)
    common_kw = dict(output_filesuffix='_acc_ref_noh')
    sf.clear_caches()
    common.dynamic_run(gdir, SfcTypeTIModel, include_mb_model_heights=False,
                       **common_kw)
    with xr.open_dataset(gdir.get_filepath(
            'model_diagnostics', filesuffix='_acc_ref_noh')) as ds:
        dnoh = ds[['volume_m3']].load()
    rel = (dnoh.volume_m3.values - dref.volume_m3.values) / \
        dref.volume_m3.values
    out['dynamic']['reference_without_mb_model_heights'] = {'volume_m3': {
        'max_abs_rel': float(np.max(np.abs(rel))), 'end_rel': float(rel[-1])}}
    print(f"dyn reference without mb model heights: volume max rel diff "
          f"{100 * np.max(np.abs(rel)):.2f}%", flush=True)
    out['dynamic']['reference'] = {
        v: dref[v].values.tolist() for v in ['volume_m3', 'area_m2',
                                             'length_m']}
    out['dynamic']['time'] = dref.time.values.tolist()

    with open(os.path.join(OUT, 'accuracy.json'), 'w') as f:
        json.dump(out, f, indent=1)

    # ---- figures
    years = np.arange(Y0, Y1 + 1)
    w = fl.widths_m[glacier]
    gw = lambda mb: np.average(annual(mb, mb._climatic_mb)[:, glacier],  # noqa
                               weights=w, axis=1)
    fig, axs = plt.subplots(2, 1, figsize=(8, 6), sharex=True,
                            gridspec_kw={'height_ratios': [2, 1]})
    axs[0].plot(years, gw(ref), color=REF, label='reference')
    for c, dz in zip(SERIES, DZS):
        axs[0].plot(years, gw(runs[dz]), color=c, lw=1.5, label=f'dz={dz} m')
        axs[1].plot(years, gw(runs[dz]) - gw(ref), color=c, lw=1.5)
    axs[0].set_ylabel('glacier-wide climatic MB\n(kg m$^{-2}$ yr$^{-1}$)')
    axs[1].set_ylabel('fixed heights - reference\n(kg m$^{-2}$ yr$^{-1}$)')
    axs[0].legend(ncol=5, loc='lower left')
    axs[0].set_title('HEF main flowline, fixed heights, 1970-2002',
                     loc='left')
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, 'acc_annual_cmb.png'), dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 4.5))
    h = fl.surface_h
    for c, dz in zip(SERIES, DZS):
        e = np.array(out['static'][dz]['annual_cmb_rmse_by_point'])
        ax.plot(e[glacier], h[glacier], color=c, label=f'dz={dz} m')
    ax.set_xlabel('RMSE of annual climatic MB (kg m$^{-2}$ yr$^{-1}$)')
    ax.set_ylabel('surface elevation (m)')
    ax.set_title('Error by elevation (HEF main flowline)', loc='left')
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, 'acc_rmse_by_elevation.png'), dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 3.5))
    sy = ref.snowline_year
    sel = sy >= 1997
    ax.plot(sy[sel], ref.snowline[sel], color=REF, label='reference',
            drawstyle='steps-post')
    for c, dz in zip(SERIES[1:4:2], [25, 100]):
        ax.plot(sy[sel], runs[dz].snowline[sel], color=c, lw=1.5,
                label=f'dz={dz} m', drawstyle='steps-post')
    ax.set_ylabel('snowline (m)')
    ax.set_title('Snowline (start month Oct), finite values only', loc='left')
    ax.legend(ncol=3)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, 'acc_snowline.png'), dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4))
    key = float(date_to_floatyear(2000, 10))
    rho = ref._density_buckets_np[0, :-1]
    x = fl.dx_meter * np.arange(fl.nx) / 1000
    ax.plot(x[glacier], (ref.mb_buckets_stored[key].values / rho
                         ).sum(axis=1)[glacier], color=REF, label='reference')
    for c, dz in zip(SERIES, DZS):
        ax.plot(x[glacier], (runs[dz].mb_buckets_stored[key].values / rho
                             ).sum(axis=1)[glacier], color=c, lw=1.5,
                label=f'dz={dz} m')
    ax.set_xlabel('distance along flowline (km)')
    ax.set_ylabel('snow + firn thickness (m)')
    ax.set_title('Snow and firn column on 1 Oct 2000', loc='left')
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, 'acc_column_thickness.png'), dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 3.5))
    t = dref.time.values
    for c, dz in zip(SERIES, DZS):
        ax.plot(t, 100 * (dyn[dz].volume_m3 - dref.volume_m3) /
                dref.volume_m3, color=c, label=f'dz={dz} m')
    ax.plot(t, 100 * (dyn_add[25].volume_m3 - dref.volume_m3) /
            dref.volume_m3, color=SERIES[1], ls='--', lw=1.5,
            label='dz=25 m, add_mb_model_heights=False')
    ax.axhline(0, color=REF, lw=1)
    ax.set_ylabel('volume difference (%)')
    ax.set_title('Dynamic run HEF 1960-2003 (monthly MB feedback): fixed '
                 'heights vs reference', loc='left')
    ax.legend(ncol=3, fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, 'acc_dynamic_volume.png'), dpi=150)
    plt.close(fig)


if __name__ == '__main__':
    main()
