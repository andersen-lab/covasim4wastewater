'''
Recreate the friendship plot (random vs rank), but with the rank population built by Covasim itself:
cv.Sim(pop_type='rank', density_raster=..., boundary_shapefile=...). This checks the integrated code still behaves
as a rank-based model: rank friendships mostly short, and a slower epidemic than on the random network.

Same figure as before:
    Left : number of friendships against friend distance, averaged over the runs
    Right: epidemic curves, one thin line per run and a thick mean line, for each network

For each of N_RUNS replicates it runs one epidemic on a random population (pop_type='random') and one on a rank
population (pop_type='rank'), with the same settings and epidemic seed. Each replicate rebuilds the populations,
so the spread between runs includes new agent locations and new networks as well as epidemic randomness.
Friend distances for the random network are measured on the rank population's agent locations: random friends do not
depend on where people live, so this gives the distances you would get between random pairs of these agents.

The boundary file is the San Diego one in network_rank_model/data if it exists (san_diego_boundaries.geojson,
with columns region_code, region_name, population), otherwise the small example in tests/ (which covers only part of the
county, so friend distances will be shorter than in the county-wide plot).

Run from any folder:  python compare_random_vs_rank_integrated.py
Try N = 5000 and N_RUNS = 2 first to see how long it takes.
Output: network_rank_model/results_simulations/multi_sim_compare_integrated_<N>_<runs>runs.png
'''
import os
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import geopandas as gpd
import covasim as cv
from covasim.geography import convert_points, lonlat_to_km

CURRENT_FOLDER = Path(__file__).resolve().parent            # network_rank_model/tests
REPO = CURRENT_FOLDER.parents[1]                            # the repo root
TIF      = str(REPO/'network_rank_model'/'data'/'san_diego_county_ppp_2020.tif')
COUNTY   = REPO/'network_rank_model'/'data'/'san_diego_boundaries.geojson'
BOUNDARY = str(COUNTY if COUNTY.exists() else REPO/'tests'/'example_boundaries.geojson')
os.chdir(REPO/'tests')                            # the default sewershed file is looked up from the repo's tests folder
print('Boundary file:', BOUNDARY)

# ---------------------------------------------------------------------------
# Settings (the same as the earlier plot)
# ---------------------------------------------------------------------------
N         = 20000      # people
DEG       = 20         # target average friends per person (the default for both population types)
N_DAYS    = 60         # length of the epidemic
SEED      = 1          # first random seed (replicate r uses SEED + r)
N_RUNS    = 10         # replicates per network
N_REAL    = 3_400_000  # real county population
POP_SCALE = N_REAL / N # each agent stands for this many real people
HIST_BINS = np.linspace(0, 50, 100)                          # friend-distance bins: 0-50 km in 0.5 km steps
OUT       = REPO/'network_rank_model'/'results_simulations'/f'multi_sim_compare_integrated_{N//1000}k_{N_RUNS}runs.png'
OUT.parent.mkdir(exist_ok=True)

COLORS = {'random': 'tab:gray', 'rank': 'tab:blue'}
BOUNDARY_CRS = gpd.read_file(BOUNDARY).crs


def run_population(pop_type, seed):
    ''' Build a population with Covasim itself and run the epidemic on it '''
    sim = cv.Sim(pop_type=pop_type, pop_size=N, pop_scale=POP_SCALE, pop_infected=2, n_days=N_DAYS, rand_seed=seed, verbose=0,
                 density_raster=TIF, boundary_shapefile=BOUNDARY)
    sim.run()
    return sim


# ---------------------------------------------------------------------------
# Replicates
# ---------------------------------------------------------------------------
infections = {name: [] for name in COLORS}     # new infections per day, one array per run
cum_final  = {name: [] for name in COLORS}     # total infected at the end, one number per run
dist_hist  = {name: [] for name in COLORS}     # friend-distance histogram counts, one array per run
n_edges    = {name: [] for name in COLORS}     # number of friendships, one number per run
for r in range(N_RUNS):
    sims = {name: run_population(name, SEED + r) for name in COLORS}
    # Agent locations in km, from the rank population (lon/lat in the boundary file's coordinate system)
    xy = lonlat_to_km(convert_points(sims['rank'].pars['people_coords'], BOUNDARY_CRS, 'EPSG:4326'))
    for name, sim in sims.items():
        layer = sim.people.contacts['a']
        p1, p2 = np.array(layer['p1']), np.array(layer['p2'])
        dist = np.linalg.norm(xy[p1] - xy[p2], axis=1)
        dist_hist[name].append(np.histogram(dist, bins=HIST_BINS)[0])
        n_edges[name].append(len(p1))
        infections[name].append(np.array(sim.results['new_infections'].values))
        cum_final[name].append(int(sim.results['cum_infections'][-1]))
        days = np.array(sim.results['t'])
        degree = np.bincount(np.concatenate([p1, p2]), minlength=N)
        print(f'replicate {r+1}/{N_RUNS} {name:6s}: edges={len(p1):,}, mean degree={degree.mean():.1f}, '
              f'median friend distance={np.median(dist):.2f} km, infected by day {N_DAYS}: {cum_final[name][-1]:,}', flush=True)

# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------
fig, ax = plt.subplots(1, 2, figsize=(11, 5))

# Left: friendship lengths, averaged over the runs (bar height = mean count per bin)
centers = (HIST_BINS[:-1] + HIST_BINS[1:]) / 2
width   = HIST_BINS[1] - HIST_BINS[0]
for name in COLORS:
    ax[0].bar(centers, np.mean(dist_hist[name], axis=0), width=width, alpha=0.6, label=name, color=COLORS[name])
ax[0].set_xlabel('friend distance (km)'); ax[0].set_ylabel('number of friendships (mean of runs)')
ax[0].set_title(f"Num friendships vs friend distance\nAverage total edges over {N_RUNS} runs: "
                f"random = {np.mean(n_edges['random']):,.0f}, rank = {np.mean(n_edges['rank']):,.0f}", fontsize=10)
ax[0].legend()

# Right: each thin line is one epidemic, the thick line is the mean over runs
for name in COLORS:
    arr = np.array(infections[name])
    for a_ in arr:
        ax[1].plot(days, a_, color=COLORS[name], alpha=0.2, lw=0.8)
    ax[1].plot(days, arr.mean(axis=0), color=COLORS[name], lw=2.5, label=f'{name} (mean of {N_RUNS})')
ax[1].set_xlabel('day'); ax[1].set_ylabel('new infections per day')
ax[1].set_title(f'Epidemic: {N_RUNS} runs per network', fontsize=10)
ax[1].legend()

summary = (f'N = {N:,}, target contacts = {DEG}, beta = {cv.make_pars()["beta"]}\n'
           f'initial infections = 2, {N_RUNS} runs per network\n'
           + '\n'.join(f'{name}: mean {np.mean(v):,.0f} infected ({np.mean(v)/N:.0%}) by day {N_DAYS}' for name, v in cum_final.items()))
ax[1].text(0.97, 0.97, summary, transform=ax[1].transAxes, ha='right', va='top', fontsize=7, family='monospace',
           bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

plt.tight_layout(); plt.savefig(OUT, dpi=150)
print(f'Saved {OUT}')