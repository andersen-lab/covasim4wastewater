'''
Timing test for the fast rank network: how long does it take to build the friendship network for different numbers of
agents, compared with Covasim's make_random_contacts()? It times three things:
    random              Covasim's make_random_contacts()
    rank, fast, 1 core  the fast rank network on one core (src/rank_network_fast.py)
    rank, fast, all     the fast rank network on every core of this machine (n_workers=-1)

Run from network_rank_model/tests:   python test_timing_fast.py

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
from covasim.population import make_random_contacts   # Covasim's own function

sys.path.append(str(Path(__file__).resolve().parent.parent / 'src'))
from rank_network      import load_density_raster, sample_coords_from_density, lonlat_to_km
from rank_network_fast import make_rank_based_contacts_fast as rank_fast       # src/rank_network_fast.py


# ---------------------------------------------------------------------------
# Settings
# ---------------------------------------------------------------------------
TIF            = '../data/san_diego_county_ppp_2020.tif'            # people-per-cell raster
N_LIST         = [1000, 5000, 10000, 25000, 50000, 100000, 200000]  # numbers of agents to time
DEG            = 20                                                 # average friends per person
FAST_1CORE_MAX = 200000    # time the fast version on one core only up to here (about 3.5 minutes at 100,000)
OUTDIR         = Path('../results_simulations')                     # where the table and plot are saved

np.random.seed(0)
OUTDIR.mkdir(exist_ok=True)
n_cores = os.cpu_count()
print(f'This machine reports {n_cores} cores.\n')


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
    row = dict(agents=n, random_s=timed(make_random_contacts, n, DEG), rank_fast_1core_s=np.nan,
               rank_fast_allcores_s=timed(rank_fast, xy, DEG, n_workers=-1))
    if n <= FAST_1CORE_MAX:
        row['rank_fast_1core_s'] = timed(rank_fast, xy, DEG, n_workers=1)
    rows.append(row)
    f = lambda v: '       -' if pd.isna(v) else f'{v:8.2f}'
    print(f'{n:>8,} agents:  random {row["random_s"]:7.3f} s   rank fast, 1 core {f(row["rank_fast_1core_s"])} s   '
          f'rank fast, all cores {f(row["rank_fast_allcores_s"])} s', flush=True)

df = pd.DataFrame(rows)
df['rank_fast_1core_min']    = df['rank_fast_1core_s']/60
df['rank_fast_allcores_min'] = df['rank_fast_allcores_s']/60
df['allcores_speedup_vs_1core'] = df['rank_fast_1core_s']/df['rank_fast_allcores_s']
df['rank_over_random']          = df['rank_fast_allcores_s']/df['random_s']       # how many times slower rank is than random (all cores)


# ---------------------------------------------------------------------------
# 3. Table
# ---------------------------------------------------------------------------
shown = ['agents', 'random_s', 'rank_fast_1core_s', 'rank_fast_allcores_s', 'rank_fast_allcores_min', 'allcores_speedup_vs_1core', 'rank_over_random']
print('\n', df[shown].round({'random_s': 3, 'rank_fast_1core_s': 1, 'rank_fast_allcores_s': 1, 'rank_fast_allcores_min': 2,
                             'allcores_speedup_vs_1core': 1, 'rank_over_random': 0}).to_string(index=False))
df.to_csv(OUTDIR / 'timing_table_fast.csv', index=False)

# Growth rate: slope of log(time) against log(agents), using sizes of 5,000 or more (smaller runs are too fast to time reliably).
# A slope near 1 means time doubles when agents double; a slope near 2 means time quadruples (N squared).
series = {'random': 'random_s', 'rank, fast, 1 core': 'rank_fast_1core_s', f'rank, fast, all {n_cores} cores': 'rank_fast_allcores_s'}
slopes = {}
for label, col in series.items():
    d = df[(df['agents'] >= 5000) & df[col].notna()]
    slopes[label] = np.polyfit(np.log(d['agents']), np.log(d[col].clip(lower=1e-3)), 1)[0] if len(d) >= 3 else np.nan
print('\nGrowth rate (time ~ agents^slope): ' + ', '.join(f'{k} {v:.1f}' for k, v in slopes.items() if not np.isnan(v)))


# ---------------------------------------------------------------------------
# 4. Plot (normal axes, time in minutes)
# ---------------------------------------------------------------------------
from matplotlib.ticker import FuncFormatter
fig, ax = plt.subplots(figsize=(8, 5.5))
styles = {'random': 'tab:gray', 'rank, fast, 1 core': 'tab:orange', f'rank, fast, all {n_cores} cores': 'tab:green'}
for label, col in series.items():
    d = df[df[col].notna()]
    ax.plot(d['agents'], d[col]/60, 'o-', color=styles[label], label=label)
    if label != 'random':     # label the largest point of each rank line with its time
        last = d.iloc[-1]
        ax.annotate(f'{last[col]/60:.1f} min', (last['agents'], last[col]/60), textcoords='offset points', xytext=(-6, 8), ha='right', fontsize=9, color=styles[label])
ax.set_xlim(left=0); ax.set_ylim(bottom=0)
ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f'{int(v):,}'))
ax.set_xlabel('number of agents'); ax.set_ylabel('time to build network (minutes)')
ax.set_title(f'Network build time vs number of agents ({DEG} friends per person)')
ax.grid(True, alpha=0.3); ax.legend(fontsize=9, loc='upper left')
plt.tight_layout(); plt.savefig(OUTDIR / 'timing_plot_fast.png', dpi=150)
print(f'Saved {OUTDIR / "timing_table_fast.csv"} and {OUTDIR / "timing_plot_fast.png"}')