'''
Speed plot for the fast rank network, in the same style as the earlier plot: how long does it take to build the friendship
network for different numbers of agents? It times three things:
    random                              Covasim's make_random_contacts()
    rank, fast (default)                the fast rank network as it runs by default: one core
    rank, fast, use_workers=True        the same function, letting it choose how many cores to use for this computer

Run from network_rank_model/tests:   python test_timing_fast.py
(about 8 minutes on a big server: most of it is the one-core run at 200,000 agents, which takes about 6 minutes)

Output:
    - a table printed to the terminal and saved as ../results_simulations/timing_table_fast.csv
    - a plot saved as ../results_simulations/timing_plot_fast.png   (normal axes, time in minutes)

Only the network build is timed (not placing agents on the map, not running Covasim).
'''
import os
import sys
import time
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
from covasim.population import make_random_contacts   # Covasim's own function

sys.path.append(str(Path(__file__).resolve().parent.parent / 'src'))
from rank_network      import load_density_raster, sample_coords_from_density, lonlat_to_km
from rank_network_fast import make_rank_based_contacts_fast as rank_fast       # src/rank_network_fast.py
from rank_network_fast import _choose_workers                                  # how many workers use_workers=True picks


# ---------------------------------------------------------------------------
# Settings
# ---------------------------------------------------------------------------
TIF            = '../data/san_diego_county_ppp_2020.tif'            # people-per-cell raster
N_LIST         = [1000, 5000, 10000, 25000, 50000, 100000, 200000]  # numbers of agents to time
DEG            = 20                                                 # average friends per person
DEFAULT_MAX    = 200000    # time the default (one core) version only up to here. At 200,000 it takes about 6 minutes.
                           # Lower this to save time: beyond it the one-core time is ESTIMATED (dashed line).
OUTDIR         = Path('../results_simulations')                     # where the table and plot are saved

np.random.seed(0)
OUTDIR.mkdir(exist_ok=True)
n_reported = os.cpu_count()
n_allowed = len(os.sched_getaffinity(0)) if hasattr(os, 'sched_getaffinity') else n_reported
print(f'This machine reports {n_reported} cores; this job may use {n_allowed}.')
print('With use_workers=True the workers used at each size would be: ' + ', '.join(f'{n:,}: {_choose_workers(n, use_workers=True)}' for n in N_LIST) + '\n')


def timed(function, *args, **kwargs):
    t = time.time()
    function(*args, **kwargs)
    return time.time() - t


# ---------------------------------------------------------------------------
# 1. Place the largest number of agents once; smaller runs use the first n of them
# ---------------------------------------------------------------------------
raster = load_density_raster(TIF)
xy_all = lonlat_to_km(sample_coords_from_density(raster, max(N_LIST)))   # (max N, 2) km


# ---------------------------------------------------------------------------
# 2. Time everything at each size
# ---------------------------------------------------------------------------
rows = []
for n in N_LIST:
    xy = xy_all[:n]
    row = dict(agents=n, random_s=timed(make_random_contacts, n, DEG), rank_default_s=np.nan,
               rank_use_workers_s=timed(rank_fast, xy, DEG, use_workers=True), workers_used=_choose_workers(n, use_workers=True))
    if n <= DEFAULT_MAX:
        row['rank_default_s'] = timed(rank_fast, xy, DEG)
    rows.append(row)
    f = lambda v: '       -' if pd.isna(v) else f'{v:8.2f}'
    print(f'{n:>8,} agents:  random {row["random_s"]:7.3f} s   rank fast (default) {f(row["rank_default_s"])} s   '
          f'rank fast, use_workers=True {f(row["rank_use_workers_s"])} s ({row["workers_used"]} workers)', flush=True)

df = pd.DataFrame(rows)

# Default (one-core) times beyond DEFAULT_MAX are estimated as (use_workers time) x (the speedup measured at the largest size timed both ways)
measured = df[df['rank_default_s'].notna()]
speedup_at_largest = (measured['rank_default_s']/measured['rank_use_workers_s']).iloc[-1] if len(measured) else np.nan
df['rank_default_estimated_s'] = np.where(df['rank_default_s'].isna(), df['rank_use_workers_s']*speedup_at_largest, np.nan)
df['rank_default_min']     = df['rank_default_s']/60
df['rank_use_workers_min'] = df['rank_use_workers_s']/60
df['speedup_use_workers']  = df['rank_default_s']/df['rank_use_workers_s']
df['rank_over_random']     = df['rank_use_workers_s']/df['random_s']       # how many times slower rank (use_workers=True) is than random


# ---------------------------------------------------------------------------
# 3. Table
# ---------------------------------------------------------------------------
shown = ['agents', 'random_s', 'rank_default_s', 'rank_default_estimated_s', 'rank_use_workers_s', 'workers_used', 'rank_use_workers_min',
         'speedup_use_workers', 'rank_over_random']
print('\n', df[shown].round({'random_s': 3, 'rank_default_s': 1, 'rank_default_estimated_s': 1, 'rank_use_workers_s': 1,
                             'rank_use_workers_min': 2, 'speedup_use_workers': 1, 'rank_over_random': 0}).to_string(index=False))
df.to_csv(OUTDIR / 'timing_table_fast.csv', index=False)


# ---------------------------------------------------------------------------
# 4. Plot (normal axes, time in minutes)
# ---------------------------------------------------------------------------
series = {'random': ('random_s', 'tab:gray'),
          'rank, fast (default: one core)': ('rank_default_s', 'tab:orange'),
          f'rank, fast, use_workers=True (up to {int(df["workers_used"].max())} workers)': ('rank_use_workers_s', 'tab:green')}
fig, ax = plt.subplots(figsize=(8, 5.5))
for label, (col, color) in series.items():
    d = df[df[col].notna()]
    ax.plot(d['agents'], d[col]/60, 'o-', color=color, label=label)
    if label != 'random':     # label the largest point of each rank line with its time
        last = d.iloc[-1]
        ax.annotate(f'{last[col]/60:.1f} min', (last['agents'], last[col]/60), textcoords='offset points', xytext=(-8, -16), ha='right', fontsize=9, color=color)
est = df[df['rank_default_estimated_s'].notna()]
if len(est):      # dashed continuation of the default line, from the last measured point through the estimates
    last = measured.iloc[-1]
    ax.plot([last['agents']] + list(est['agents']), [last['rank_default_s']/60] + list(est['rank_default_estimated_s']/60), 'o--',
            color='tab:orange', mfc='white', label=f'rank, fast (default), estimated beyond {int(last["agents"]):,}')
ymax = np.nanmax([df['random_s'].max(), df['rank_default_s'].max(), df['rank_use_workers_s'].max(), df['rank_default_estimated_s'].max()])/60
ax.set_xlim(0, df['agents'].max()*1.05); ax.set_ylim(0, ymax*1.15)      # room for every line and its label
ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f'{int(v):,}'))
ax.set_xlabel('number of agents'); ax.set_ylabel('time to build network (minutes)')
ax.set_title(f'Network build time vs number of agents ({DEG} friends per person)')
ax.grid(True, alpha=0.3); ax.legend(fontsize=9, loc='upper left')
plt.tight_layout(); plt.savefig(OUTDIR / 'timing_plot_fast.png', dpi=150)
print(f'\nSaved {OUTDIR / "timing_table_fast.csv"} and {OUTDIR / "timing_plot_fast.png"}')