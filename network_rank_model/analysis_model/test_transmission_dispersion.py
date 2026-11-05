'''
Aim: what does super-spreading (overdispersed individual infectiousness) do to the outbreak?

THE PARAMETER
    Covasim gives every person a random infectiousness, drawn from a negative binomial distribution. It is the Sim parameter
    beta_dist, whose default is
        dict(dist='neg_binomial', par1=1.0, par2=0.45, step=0.01)
    par1 is the mean (1.0: an average person is as infectious as the base beta says) and par2 is the dispersion k.
    The generic sampler cv.sample() in covasim/utils.py, with the line  dist == 'neg_binomial': n_neg_binomial(rate=par1,
    dispersion=par2, ...),  is what turns these numbers into draws. We do not edit utils.py: changing par2 in beta_dist
    is the supported way, and editing the sampler would change every distribution that uses it.

    HOW THE NEGATIVE BINOMIAL IS IMPLEMENTED (n_neg_binomial in covasim/utils.py, called by cv.sample)
        The problem: numpy's negative binomial gives whole-number counts (0, 1, 2, 3, ...), but Covasim wants an
        infectiousness multiplier that can be fractional (0, 0.01, 0.02, ...). The step = 0.01 is a workaround for this
        and is not needed to use a negative binomial as such.
        The trick, in three steps (rate = par1 = 1.0 the wanted mean, dispersion = par2 = k, step = 0.01):
            1. Divide the wanted mean by the step: rate/step = 1.0/0.01 = 100, the mean of the whole-number count.
            2. Draw a negative binomial count with that mean, using numpy with n = k and p = k/(100 + k).
               The count has mean 100 and variance 100 + 100^2/k.
            3. Multiply the count by the step: infectiousness = count x 0.01, so the values are 0, 0.01, 0.02, ...
        The code, in n_neg_binomial:
            nbn_n = dispersion
            nbn_p = dispersion/(rate/step + dispersion)                            # step 1 (rate/step), and p for step 2
            samples = np.random.negative_binomial(n=nbn_n, p=nbn_p, size=n)*step   # step 2 (the draw), then step 3 (*step)
        and cv.sample reaches it with  samples = n_neg_binomial(rate=par1, dispersion=par2, n=size, **kwargs),
        where step arrives in **kwargs from the beta_dist dictionary.
        Result: one random infectiousness per agent, with mean 1.0 (= rate) and variance 0.01^2 x (100 + 100^2/k) =
        0.01 + 1/k. (The 0.01 is a small side effect of using whole-number counts; the 1/k term is the overdispersion.)
        A person's infectiousness multiplies the transmission from them, and the mean stays 1, so the average transmission
        is the same and only the spread changes:
            k = 0.1 -> std 3.2     k = 0.45 -> std 1.5 (the default)     k = 1 -> std 1.0     k = 10 -> std 0.3
        and as k grows without limit the std falls to 0.1 (the Poisson limit): everyone is about equally infectious.
        A small k means a few people are far more infectious than the rest (super-spreaders) and many are hardly
        infectious at all. For example, 66% of people have an infectiousness below 0.1 at k = 0.1, against 27% at
        k = 0.45 and 0% at k = 10 (simulated draws from the function above).

    This is a different thing from the friend-count dispersion we compared before (make_rank_based_contacts, dispersion=...),
    which changes how many friends a person has. Here every person has the same kind of friends, and what varies is how
    infectious each person is. The friend-count setting stays at its default (Poisson).

WHAT THE SCRIPT DOES
    For each k it runs N_RUNS epidemics on the rank network (same agents and seeds for every k within a replicate) and
    prints, for each k, how many runs died out, the final size, the peak, and how long the epidemic took to reach 1% infected.
    Super-spreading is expected to matter most early on, when a few people decide whether an outbreak starts, so the
    defaults use few initial infections (N_INITIAL = 5) and many runs (N_RUNS = 20) on a smaller population (N = 20,000).
    Saves two figures in network_rank_model/results_simulations/:
        superspreading_spaghetti_<tag>.png : one panel per k, a thin line per run and a thick mean line
        superspreading_summary_<tag>.png   : the final size of every run, and the day each run reached 1% infected

Data: the San Diego files in docs/tutorials/san-diego-data, as in tut_wastewater.py. Run  git pull  on network_rank_model if missing.
Run from anywhere:  python check_superspreading_outbreaks.py
Try N_RUNS = 3 first to see how long it takes.
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
N         = 100000      # agents
N_DAYS    = 120        # length of the epidemic
SEED      = 1          # first seed (replicate r uses SEED + r)
N_RUNS    = 20         # replicates per setting
BETA      = 0.016      # transmission per contact (the Covasim default); try 0.008 for a less saturated epidemic
N_INITIAL = 5          # agents infected on day 0. Few, so that early super-spreading can decide whether an outbreak takes off
# The dispersion k of individual infectiousness, as (label, k). Smaller k = more super-spreading; the default is 0.45.
SETTINGS  = [('k = 0.1 (extreme super-spreading)', 0.1), ('k = 0.45 (Covasim default)', 0.45),
             ('k = 1', 1.0), ('k = 10 (almost uniform)', 10.0)]
COLORS    = ['tab:red', 'tab:orange', 'tab:green', 'tab:gray']
DIED_OUT  = 0.01       # a run counts as died out if fewer than this share of the agents were infected
EARLY     = 0.01       # the time to reach this share of the agents infected is the "time to take off"
TAG       = f'{N//1000}k_beta{BETA}_seeds{N_INITIAL}'      # in the file names, so runs with other settings do not overwrite each other


def run_one(k, seed):
    ''' Build the rank population and run the epidemic, with the individual infectiousness dispersion set to k '''
    sim = cv.Sim(pop_type='rank', pop_size=N, pop_infected=N_INITIAL, n_imports=0, beta=BETA, n_days=N_DAYS, rand_seed=seed, verbose=0,
                 beta_dist=dict(dist='neg_binomial', par1=1.0, par2=k, step=0.01),     # the setting being varied: all four entries are needed
                 density_raster=str(TIF), boundary_shapefile=str(BOUNDARY), sewershed_file=str(SEWERSHEDS))
    sim.initialize(use_workers=True)               # the workers change the speed of building the network, not the network
    sim.run()
    return sim


# ---------------------------------------------------------------------------
# Replicates
# ---------------------------------------------------------------------------
infections = {label: [] for label, _ in SETTINGS}      # new infections per day, one array per run
final      = {label: [] for label, _ in SETTINGS}      # agents infected at least once by the end, one number per run
take_off   = {label: [] for label, _ in SETTINGS}      # the first day the share infected reached EARLY (NaN if it never did)
for r in range(N_RUNS):
    for label, k in SETTINGS:
        t0 = time.time()
        sim = run_one(k, SEED + r)
        new = np.array(sim.results['new_infections'].values)
        days = np.array(sim.results['t'])
        ever = int(np.sum(~np.isnan(np.array(sim.people.date_exposed))))     # a reinfection does not count twice
        reached = np.flatnonzero(np.cumsum(new) >= EARLY*N)
        infections[label].append(new)
        final[label].append(ever)
        take_off[label].append(days[reached[0]] if len(reached) else np.nan)
        print(f'replicate {r+1}/{N_RUNS} {label:36s} beta_dist k {sim["beta_dist"]["par2"]:<5} infected: {ever:,} agents ({time.time()-t0:.0f} s)', flush=True)

# ---------------------------------------------------------------------------
# Summary table
# ---------------------------------------------------------------------------
print(f'\n{N:,} agents, {N_INITIAL} initial infections, beta {BETA}, {N_RUNS} runs per setting, {N_DAYS} days\n')
print(f'{"":36s} {"took off":>9} {"final size":>11} {"peak/day":>9} {"peak day":>14} {"days to 1%":>15}')
for label, _ in SETTINGS:
    f, arr, t1 = np.array(final[label]), np.array(infections[label]), np.array(take_off[label], float)
    ok = f >= DIED_OUT*N                                         # the runs that did not die out
    if ok.any():
        peak_day = days[arr[ok].argmax(axis=1)]
        extra = f'{arr[ok].max(axis=1).mean():>9,.0f} {peak_day.mean():>7.0f} +/-{peak_day.std():>3.0f} {np.nanmean(t1[ok]):>8.1f} +/-{np.nanstd(t1[ok]):>4.1f}'
    else:
        extra = f'{"-":>9} {"-":>14} {"-":>15}'
    print(f'{label:36s} {ok.sum():>6}/{N_RUNS} {f.mean()/N:>11.0%} {extra}')
print('\ntook off = runs that infected at least 1% of the agents. final size is the mean over all runs; the last three columns are for runs that took off.')

# ---------------------------------------------------------------------------
# Figure 1: one panel per setting
# ---------------------------------------------------------------------------
top = max(np.max(a) for arrs in infections.values() for a in arrs)
fig, ax = plt.subplots(2, 2, figsize=(11, 8), sharex=True, sharey=True)
for a, (label, _), color in zip(ax.ravel(), SETTINGS, COLORS):
    arr = np.array(infections[label])
    for line in arr:
        a.plot(days, line, color=color, alpha=0.3, lw=0.8)
    a.plot(days, arr.mean(axis=0), color=color, lw=2.5)
    took = np.sum(np.array(final[label]) >= DIED_OUT*N)
    a.set_title(f'{label}\n{took}/{N_RUNS} runs took off', fontsize=10)
    a.set_ylim(0, top*1.05)
for a in ax[1]:
    a.set_xlabel('day')
for a in ax[:, 0]:
    a.set_ylabel('new infections per day (agents)')
fig.suptitle(f'Rank network, {N:,} agents, {N_INITIAL} initial infections (thin = one run, thick = mean of all runs)', fontsize=11)
plt.tight_layout(); plt.savefig(OUT_DIR/f'superspreading_spaghetti_{TAG}.png', dpi=150)

# ---------------------------------------------------------------------------
# Figure 2: final size and time to take off, run by run
# ---------------------------------------------------------------------------
fig, ax = plt.subplots(1, 2, figsize=(12, 5))
rng = np.random.RandomState(0)
for i, ((label, _), color) in enumerate(zip(SETTINGS, COLORS)):
    f, t1 = np.array(final[label]), np.array(take_off[label], float)
    jitter = rng.uniform(-0.15, 0.15, len(f))
    ax[0].scatter(np.full(len(f), i) + jitter, f/N, color=color, alpha=0.7)
    ax[0].hlines((f/N).mean(), i - 0.3, i + 0.3, color='black', lw=2)
    ax[1].scatter(np.full(len(f), i) + jitter, t1, color=color, alpha=0.7)
    ax[1].hlines(np.nanmean(t1), i - 0.3, i + 0.3, color='black', lw=2)
for a in ax:
    a.set_xticks(range(len(SETTINGS))); a.set_xticklabels([f'k = {k:g}' for _, k in SETTINGS])
ax[0].set_ylabel(f'share of agents infected by day {N_DAYS}'); ax[0].set_title('Final size of each run (black = mean)', fontsize=10)
ax[1].set_ylabel(f'day the share infected reached {EARLY:.0%}'); ax[1].set_title('Time to take off (runs that never reached it are missing)', fontsize=10)
plt.tight_layout(); plt.savefig(OUT_DIR/f'superspreading_summary_{TAG}.png', dpi=150)
print(f'Saved {OUT_DIR/f"superspreading_spaghetti_{TAG}.png"}\nSaved {OUT_DIR/f"superspreading_summary_{TAG}.png"}')