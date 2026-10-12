'''
Very simple timing test: how long does it take to build the friendship network for
different numbers of agents, for the rank-based network vs Covasim's make_random_contacts?

Run from network_rank_model/tests:   python test_timing.py

Output:
    - a table printed to the terminal and saved as ../results_simulations/timing_table.csv
    - a plot saved as ../results_simulations/timing_plot.png   (log-log axes)

Only the network build is timed (not placing agents on the map, not running Covasim).
'''
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from covasim.population import make_random_contacts   # Covasim's own function

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent / 'src'))
from rank_network import load_density_raster, sample_coords_from_density, lonlat_to_km, make_rank_based_contacts


# ---------------------------------------------------------------------------
# Settings
# ---------------------------------------------------------------------------
TIF    = '../data/san_diego_county_ppp_2020.tif'    # people-per-cell raster
N_LIST = [1000, 5000, 10000, 25000, 50000, 100000]    # numbers of agents to time (rank at 40,000 takes about 1.5 minutes)
DEG    = 20                                         # average friends per person
OUTDIR = Path('../results_simulations')             # where the table and plot are saved

np.random.seed(0)
OUTDIR.mkdir(exist_ok=True)


# ---------------------------------------------------------------------------
# 1. Place the largest number of agents once; smaller runs use the first n of them
# ---------------------------------------------------------------------------
raster = load_density_raster(TIF)
xy_all = lonlat_to_km(sample_coords_from_density(raster, max(N_LIST)))   # (max N, 2) km


# ---------------------------------------------------------------------------
# 2. Time both networks at each size
# ---------------------------------------------------------------------------
rows = []
for n in N_LIST:
    xy = xy_all[:n]

    t = time.time()
    make_random_contacts(n, DEG)
    t_random = time.time() - t

    t = time.time()
    make_rank_based_contacts(xy, DEG)
    t_rank = time.time() - t

    rows.append(dict(agents=n, random_s=t_random, rank_s=t_rank))
    print(f'{n:>8,} agents:  random {t_random:8.3f} s   rank {t_rank:8.3f} s', flush=True)

df = pd.DataFrame(rows)
df['rank_over_random'] = df['rank_s'] / df['random_s']       # how many times slower rank is


# ---------------------------------------------------------------------------
# 3. Table
# ---------------------------------------------------------------------------
print('\n', df.round({'random_s': 4, 'rank_s': 2, 'rank_over_random': 0}).to_string(index=False))
df.to_csv(OUTDIR / 'timing_table.csv', index=False)

# Growth rate: slope of log(time) vs log(agents), using sizes of 5,000 or more (smaller runs are too fast to time reliably).
# A slope near 1 means time doubles when agents double; a slope near 2 means time quadruples (N squared).
# The random times are only milliseconds at these sizes, so its slope is a rough estimate.
big = df[df['agents'] >= 5000]
slope_random = np.polyfit(np.log(big['agents']), np.log(big['random_s'].clip(lower=1e-3)), 1)[0]
slope_rank   = np.polyfit(np.log(big['agents']), np.log(big['rank_s']), 1)[0]
print(f'\nGrowth rate (time ~ agents^slope): random slope = {slope_random:.1f}, rank slope = {slope_rank:.1f}')


# ---------------------------------------------------------------------------
# 4. Plot
# ---------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(7, 5))
ax.plot(df['agents'], df['random_s'], 'o-', color='tab:gray', label=f'random (slope {slope_random:.1f})')
ax.plot(df['agents'], df['rank_s'],   'o-', color='tab:blue', label=f'rank-based (slope {slope_rank:.1f})')
ax.set_xscale('log'); ax.set_yscale('log')
ax.set_xlabel('number of agents'); ax.set_ylabel('time to build network (seconds)')
ax.set_title(f'Network build time vs number of agents ({DEG} friends per person)')
ax.grid(True, which='both', alpha=0.3); ax.legend()
plt.tight_layout(); plt.savefig(OUTDIR / 'timing_plot.png', dpi=150)
print(f'Saved {OUTDIR / "timing_table.csv"} and {OUTDIR / "timing_plot.png"}')

