'''
Aim: what happens to the outbreak when BOTH negative binomial dispersions are varied together?

THE TWO DISPERSIONS (each is a negative binomial with a mean that stays fixed; k is the dispersion, smaller = more uneven)
    1. k for the NUMBER OF FRIENDS (friend-count distribution). Passed to the rank network through sim.initialize(dispersion=k).
       Default None = Poisson. A small k gives hubs (a few people with very many friends) and many people with very few.
       Variance of the friendships each person starts = mean + mean^2/k.
    2. k for INFECTIOUSNESS (beta_dist). The Sim parameter dict(dist='neg_binomial', par1=1.0, par2=k, step=0.01); Covasim
       default k = 0.45. A small k gives super-spreaders (a few people far more infectious than the rest) and many people
       who are hardly infectious at all. The variance of a person's infectiousness is about 0.01 + 1/k (mean 1).
    How each is built: see the docstrings of check_dispersion.py (friend counts) and check_superspreading_outbreaks.py
    (infectiousness: numpy draws a whole-number count with mean rate/step = 100 and the code multiplies it by step = 0.01).
    A person's friend count and infectiousness are drawn separately, so they are independent of each other: a super-spreader
    can have many friends or few.

WHAT THE SCRIPT DOES
    Runs every combination of CONTACT_KS (rows) and INFECT_KS (columns), N_RUNS epidemics each, on the rank network.
    Within a replicate every combination uses the same seed, so the agents match and only the two k values differ.
    For example (Poisson, 10) is near-uniform on both counts; (0.1, 0.1) has hubs AND super-spreaders; (5, 0.1) has
    nearly even friend counts but super-spreaders.
    Prints a table, one row per combination: friends std (a check the friend-count setting took effect), how many runs
    took off, final size, the peak per 1,000 agents, the peak day, and the days to reach 1% infected.
    Saves two figures in network_rank_model/results_simulations/:
        two_dispersions_spaghetti_<tag>.png : one panel per combination (rows: friend-count k, columns: infectiousness k),
                                              a thin line per run, a thick mean of all runs, a dashed mean of runs that took off
        two_dispersions_summary_<tag>.png   : three grids over the same rows and columns: share of runs that took off,
                                              mean final size, and mean days to reach 1% infected (runs that took off)

Data: the San Diego files in docs/tutorials/san-diego-data, as in tut_wastewater.py. Run  git pull  on network_rank_model if missing.
Time: each simulation takes seconds at 20,000 agents, and there are len(CONTACT_KS) x len(INFECT_KS) x N_RUNS of them.
Run from anywhere:  python check_two_dispersions_outbreaks.py       Try N_RUNS = 3 first.
'''
import os
import time
import logging
import warnings
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import covasim as cv
import geopandas as gpd

warnings.filterwarnings('ignore', message='.*no "population" column.*')   # harmless: with a raster the regions come from the raster
logging.getLogger('rasterio._env').setLevel(logging.ERROR)                # the repeated PROJ version message

# ---------------------------------------------------------------------------
# Data and checks
# ---------------------------------------------------------------------------
HERE = Path(__file__).resolve().parent            # network_rank_model/<folder>
REPO = HERE.parents[1]                            # the repo root
DATA       = REPO/'docs'/'tutorials'/'san-diego-data'
TIF        = DATA/'san_diego_county_ppp_2020.tif'            # people per ~100 m cell: where agents are placed
BOUNDARY   = DATA/'sandiego_zip_boundaries.geojson'          # one region per ZIP code
SEWERSHEDS = DATA/'sandiego_sewersheds.geojson'              # one catchment per treatment plant
OUT_DIR    = REPO/'network_rank_model'/'results_simulations'
OUT_DIR.mkdir(exist_ok=True)

for f in (TIF, BOUNDARY, SEWERSHEDS):
    if not f.exists():
        raise FileNotFoundError(f'{f} not found. Run  git pull  on the network_rank_model branch to get docs/tutorials/san-diego-data.')
default_pars = cv.make_pars()
if 'sewershed_file' not in default_pars:
    raise KeyError('This Covasim has no sewershed_file setting. Run  git pull  on the network_rank_model branch.')
print('Covasim default beta_dist:', default_pars['beta_dist'])     # expect par2 = 0.45
print(f'Boundary file: {BOUNDARY.name}, {len(gpd.read_file(BOUNDARY))} regions')

# ---------------------------------------------------------------------------
# Settings
# ---------------------------------------------------------------------------
N         = 20000      # agents
N_DAYS    = 120        # length of the epidemic
SEED      = 1          # first seed (replicate r uses SEED + r)
N_RUNS    = 10         # replicates per combination
BETA      = 0.016      # transmission per contact (the Covasim default); try 0.008 for a less saturated epidemic
N_INITIAL = 5          # agents infected on day 0. Few, so that early spread can decide whether an outbreak takes off
CONTACT_KS = [None, 5.0, 0.1]   # dispersion k of the number of friends (rows). None = Poisson; smaller k = more uneven (hubs)
INFECT_KS  = [0.1, 0.45, 10.0]  # dispersion k of infectiousness (columns). 0.45 is the Covasim default; smaller k = more super-spreading
DIED_OUT  = 0.01       # a run counts as died out if fewer than this share of the agents were infected
EARLY     = 0.01       # the time to reach this share of the agents infected is the "time to take off"
PER       = 1000/N     # new infections are shown per 1,000 agents, so runs with different N can be compared
TAG       = f'{N//1000}k_beta{BETA}_seeds{N_INITIAL}'      # in the file names, so runs with other settings do not overwrite each other


def contact_name(k):
    return 'Poisson' if k is None else f'k={k:g}'


def run_one(k_contacts, k_infect, seed):
    ''' Build the rank population with the two dispersions, run the epidemic, and return the sim and each agent's friend count '''
    sim = cv.Sim(pop_type='rank', pop_size=N, pop_infected=N_INITIAL, n_imports=0, beta=BETA, n_days=N_DAYS, rand_seed=seed, verbose=0,
                 beta_dist=dict(dist='neg_binomial', par1=1.0, par2=k_infect, step=0.01),   # infectiousness dispersion: all four entries are needed
                 density_raster=str(TIF), boundary_shapefile=str(BOUNDARY), sewershed_file=str(SEWERSHEDS))
    extra = dict(use_workers=True)                 # the workers change the speed of building the network, not the network
    if k_contacts is not None:
        extra['dispersion'] = k_contacts           # friend-count dispersion: passed through sim.initialize to make_rank_based_contacts
    sim.initialize(**extra)
    layer = sim.people.contacts['a']
    degree = np.bincount(np.concatenate([layer['p1'], layer['p2']]), minlength=N)
    sim.run()
    return sim, degree


# ---------------------------------------------------------------------------
# Replicates
# ---------------------------------------------------------------------------
combos     = [(kc, ki) for kc in CONTACT_KS for ki in INFECT_KS]
infections = {c: [] for c in combos}      # new infections per day, one array per run
final      = {c: [] for c in combos}      # agents infected at least once by the end, one number per run
take_off   = {c: [] for c in combos}      # the first day the share infected reached EARLY (NaN if it never did)
degree_std = {c: [] for c in combos}      # std of friends per person, one number per run
for r in range(N_RUNS):
    for kc, ki in combos:
        t0 = time.time()
        sim, degree = run_one(kc, ki, SEED + r)
        new = np.array(sim.results['new_infections'].values)
        days = np.array(sim.results['t'])
        ever = int(np.sum(~np.isnan(np.array(sim.people.date_exposed))))     # a reinfection does not count twice
        reached = np.flatnonzero(np.cumsum(new) >= EARLY*N)
        infections[(kc, ki)].append(new)
        final[(kc, ki)].append(ever)
        take_off[(kc, ki)].append(days[reached[0]] if len(reached) else np.nan)
        degree_std[(kc, ki)].append(degree.std())
        print(f'replicate {r+1}/{N_RUNS} friends {contact_name(kc):8s} infectiousness k={sim["beta_dist"]["par2"]:<5g} '
              f'friends std {degree.std():5.1f}, infected: {ever:,} agents ({time.time()-t0:.0f} s)', flush=True)

# ---------------------------------------------------------------------------
# Summary table
# ---------------------------------------------------------------------------
print(f'\n{N:,} agents, {N_INITIAL} initial infections, beta {BETA}, {N_RUNS} runs per combination, {N_DAYS} days\n')
print(f'{"friends":>9} {"infect k":>9} {"friends std":>11} {"took off":>9} {"final size":>11} {"peak/1,000":>11} {"peak day":>14} {"days to 1%":>15}')
for kc, ki in combos:
    f, arr, t1 = np.array(final[(kc, ki)]), np.array(infections[(kc, ki)]), np.array(take_off[(kc, ki)], float)
    ok = f >= DIED_OUT*N                                         # the runs that did not die out
    if ok.any():
        peak_day = days[arr[ok].argmax(axis=1)]
        extra = f'{arr[ok].max(axis=1).mean()*PER:>11.1f} {peak_day.mean():>7.0f} +/-{peak_day.std():>3.0f} {np.nanmean(t1[ok]):>8.1f} +/-{np.nanstd(t1[ok]):>4.1f}'
    else:
        extra = f'{"-":>11} {"-":>14} {"-":>15}'
    print(f'{contact_name(kc):>9} {ki:>9g} {np.mean(degree_std[(kc, ki)]):>11.1f} {ok.sum():>6}/{N_RUNS} {f.mean()/N:>11.0%} {extra}')
print('\ntook off = runs that infected at least 1% of the agents. final size is the mean over all runs; the last three columns are for runs that took off.')
print('friends std shows the friend-count setting took effect: it should rise as its k falls, and not change with the infectiousness k.')

# ---------------------------------------------------------------------------
# Figure 1: one panel per combination (rows: friend-count k, columns: infectiousness k)
# ---------------------------------------------------------------------------
top = PER*max(np.max(a) for arrs in infections.values() for a in arrs)
fig, ax = plt.subplots(len(CONTACT_KS), len(INFECT_KS), figsize=(4.2*len(INFECT_KS), 3.2*len(CONTACT_KS)), sharex=True, sharey=True, squeeze=False)
k_colors = {0.1: 'tab:red', 0.45: 'tab:orange', 1.0: 'tab:green', 10.0: 'tab:gray'}      # the same colours for each k as in check_superspreading_outbreaks.py
colors = [k_colors.get(k, 'tab:blue') for k in INFECT_KS]
for i, kc in enumerate(CONTACT_KS):
    for j, ki in enumerate(INFECT_KS):
        a, arr = ax[i, j], np.array(infections[(kc, ki)])
        for line in arr:
            a.plot(days, line*PER, color=colors[j], alpha=0.3, lw=0.8)
        a.plot(days, arr.mean(axis=0)*PER, color=colors[j], lw=2.5)                   # the mean of ALL runs, those that died out included
        took = np.array(final[(kc, ki)]) >= DIED_OUT*N
        if took.any():
            a.plot(days, arr[took].mean(axis=0)*PER, color='black', lw=1.5, ls='--')   # the mean of the runs that took off
        a.set_title(f'friends: {contact_name(kc)}, infectiousness k={ki:g}\n{took.sum()}/{N_RUNS} took off', fontsize=9)
        a.set_ylim(0, top*1.05)
for a in ax[-1]:
    a.set_xlabel('day')
for a in ax[:, 0]:
    a.set_ylabel('new infections per day\nper 1,000 agents')
fig.suptitle(f'Rank network, {N:,} agents, {N_INITIAL} initial infections (thin = one run, thick = mean of all runs, dashed = mean of runs that took off)', fontsize=10)
plt.tight_layout(); plt.savefig(OUT_DIR/f'two_dispersions_spaghetti_{TAG}.png', dpi=150)

# ---------------------------------------------------------------------------
# Figure 2: grids of the main outcomes
# ---------------------------------------------------------------------------
shape = (len(CONTACT_KS), len(INFECT_KS))
took_off_share, final_size, days_to_1pct = np.full(shape, np.nan), np.full(shape, np.nan), np.full(shape, np.nan)
for i, kc in enumerate(CONTACT_KS):
    for j, ki in enumerate(INFECT_KS):
        f, t1 = np.array(final[(kc, ki)]), np.array(take_off[(kc, ki)], float)
        ok = f >= DIED_OUT*N
        took_off_share[i, j], final_size[i, j] = ok.mean(), f.mean()/N
        if ok.any():
            days_to_1pct[i, j] = np.nanmean(t1[ok])

fig, ax = plt.subplots(1, 3, figsize=(15, 4.5))
grids = [(took_off_share, 'Share of runs that took off', '{:.0%}', 'viridis'),
         (final_size, 'Mean final size (share of agents infected)', '{:.0%}', 'viridis'),
         (days_to_1pct, 'Mean days to reach 1% infected (runs that took off)', '{:.0f}', 'viridis_r')]
for a, (grid, title, fmt, cmap) in zip(ax, grids):
    im = a.imshow(grid, cmap=cmap, aspect='auto')
    for i in range(shape[0]):
        for j in range(shape[1]):
            if np.isnan(grid[i, j]):
                a.text(j, i, '-', ha='center', va='center')
            else:
                dark = (grid[i, j] - np.nanmin(grid))/(np.ptp(grid[~np.isnan(grid)]) + 1e-9) < 0.5 if cmap == 'viridis' else (grid[i, j] - np.nanmin(grid))/(np.ptp(grid[~np.isnan(grid)]) + 1e-9) > 0.5
                a.text(j, i, fmt.format(grid[i, j]), ha='center', va='center', color='white' if dark else 'black', fontsize=11)
    a.set_xticks(range(shape[1])); a.set_xticklabels([f'{k:g}' for k in INFECT_KS])
    a.set_yticks(range(shape[0])); a.set_yticklabels([contact_name(k) for k in CONTACT_KS])
    a.set_xlabel('infectiousness k (smaller = more super-spreading)'); a.set_ylabel('friend-count k (smaller = more hubs)')
    a.set_title(title, fontsize=10)
plt.tight_layout(); plt.savefig(OUT_DIR/f'two_dispersions_summary_{TAG}.png', dpi=150)
print(f'Saved {OUT_DIR/f"two_dispersions_spaghetti_{TAG}.png"}\nSaved {OUT_DIR/f"two_dispersions_summary_{TAG}.png"}')