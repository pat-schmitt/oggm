"""Speed figures from results/bench_speed.json (EXPERIMENTAL)."""
import os
import sys
import json

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
import plotstyle  # noqa: E402
from plotstyle import plt, REF, SERIES, MUTED  # noqa: E402

OUT = os.path.join(os.path.dirname(__file__), 'results')
FIG = os.path.join(OUT, 'figures')

SHOW = [  # (name in json, label, color)
    ('reference', 'reference', REF),
    ('+window aging (5)', 'exact numpy (3+4+5+6)', SERIES[0]),
    ('+numba kernel (2)', 'exact numba (2-6)', SERIES[1]),
    ('fixed heights dz=25 (1)', 'fixed heights dz=25 m', SERIES[2]),
    ('fixed heights dz=25, cached table',
     'fixed heights dz=25 m, table cached', SERIES[3]),
    ('MonthlyTIModel (no buckets)', 'MonthlyTIModel (no buckets)', MUTED),
]


def main():
    plotstyle.setup()
    os.makedirs(FIG, exist_ok=True)
    with open(os.path.join(OUT, 'bench_speed.json')) as f:
        d = json.load(f)

    fig, axs = plt.subplots(1, 2, figsize=(11, 4.5), sharey=True)
    titles = {'annual': 'get_annual_mb, 12 steps per call\n(calibration-like)',
              'monthly': 'get_monthly_mb, 1 step per call\n(dynamics-like)'}
    for ax, pat in zip(axs, ['annual', 'monthly']):
        res = d['scaling'][pat]
        nxs = sorted(int(k) for k in res)
        for name, label, c in SHOW:
            y = [res[str(n)][name]['us_per_step'] for n in nxs]
            ls = ':' if 'Monthly' in name else '-'
            ax.plot(nxs, y, ls, color=c, marker='o', ms=4, label=label)
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_xlabel('number of grid points')
        ax.set_title(titles[pat], loc='left')
    axs[0].set_ylabel('time per climate step (µs)\nincl. spinup and getters')
    axs[1].legend(fontsize=8, loc='upper left')
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, 'speed_scaling.png'), dpi=150)
    plt.close(fig)

    # HEF workflows
    hef = d['hef']
    names = [n for n in hef]
    fig, axs = plt.subplots(1, 3, figsize=(12, 4.2), sharey=True)
    for ax, wf in zip(axs, ['calibration', 'dynamic monthly',
                            'dynamic annual']):
        vals = [hef[n][wf] for n in names]
        cols = [REF if n == 'reference' else MUTED if 'Monthly' in n else
                SERIES[0] for n in names]
        y = np.arange(len(names))
        ax.barh(y, vals, color=cols, height=0.6)
        for yi, v in zip(y, vals):
            ax.text(v, yi, f' {v:.2f} s ({hef["reference"][wf] / v:.1f}x)',
                    va='center', fontsize=8, color=REF)
        ax.set_title(f'HEF {wf}', loc='left')
        ax.set_xlabel('wall time (s)')
        ax.set_xlim(0, max(vals) * 1.45)
        ax.grid(axis='y', visible=False)
    axs[0].set_yticks(np.arange(len(names)))
    axs[0].set_yticklabels(names, fontsize=8)
    axs[0].invert_yaxis()
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, 'speed_hef.png'), dpi=150)
    plt.close(fig)


if __name__ == '__main__':
    main()
