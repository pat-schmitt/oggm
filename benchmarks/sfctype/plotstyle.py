"""Shared matplotlib style for the figures of the study."""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402

# categorical slots in fixed order (validated default palette)
SERIES = ['#2a78d6', '#eb6834', '#1baf7a', '#eda100', '#e87ba4', '#008300',
          '#6250d6', '#e34948']
REF = '#0b0b0b'      # the reference model is drawn in primary ink
MUTED = '#52514e'
SURFACE = '#fcfcfb'


def setup():
    plt.rcParams.update({
        'figure.facecolor': SURFACE, 'axes.facecolor': SURFACE,
        'savefig.facecolor': SURFACE, 'axes.edgecolor': MUTED,
        'axes.labelcolor': REF, 'xtick.color': MUTED, 'ytick.color': MUTED,
        'axes.spines.top': False, 'axes.spines.right': False,
        'axes.grid': True, 'grid.color': '#e4e3df', 'grid.linewidth': 0.6,
        'lines.linewidth': 2, 'font.size': 10, 'legend.frameon': False,
        'axes.prop_cycle': matplotlib.cycler(color=SERIES),
        'axes.axisbelow': True,
    })
