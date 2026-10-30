'''
CHECK: what do simulated outbreaks look like for Poisson and negative binomial friend counts?

For each dispersion setting it runs N_RUNS Covasim epidemics on the rank network and compares them:
    dispersion=None (Poisson, the default), 10, 1 and 0.5. A smaller dispersion gives a more uneven number of
    friends per person: more people with very few friends and a few with very many.

Every replicate uses the same random seed for all settings, so the agents and the epidemic seed match and the main difference
is the friend-count distribution. The population is built by Covasim, set up as in the tutorial (docs/tutorials/tut_wastewater.py): pop_type='rank', agents
placed from the San Diego density raster inside the ZIP-code boundary file, each agent's region the ZIP polygon it falls in, and
each agent's sewershed the catchment containing its home. All the data is in the repo, in docs/tutorials/san-diego-data
(san_diego_county_ppp_2020.tif, sandiego_zip_boundaries.geojson, sandiego_sewersheds.geojson). Run  git pull  on
network_rank_model if that folder or the sewershed_file setting is missing.
sim.initialize(dispersion=...) passes the setting to the rank network.

Settings: 100,000 agents, 20 contacts, 120 days, 3 runs per setting, 100 initial infections (0.1% of the agents, as in the tutorial). There is no pop_scale here, so
the results are counts of agents and the share infected is a share of the agents. This is a long run (each simulation
takes a few minutes), so run it in the background, e.g.
    nohup python network_rank_model/tests/check_dispersion_outbreaks.py > outbreaks.log 2>&1 &
and watch it with  tail -f outbreaks.log.  Try N = 5000 and N_RUNS = 2 first to see how it behaves.

Prints, for each setting: friends-per-person spread (a check that the setting took effect), the share infected by the end,
the peak of new infections and the day it happens, and how many runs died out.
Saves two figures in network_rank_model/results_simulations/ (the size and beta are in the file names):
    dispersion_outbreaks_spaghetti_<N>k_beta<beta>.png : one panel per setting, a thin line per run and a thick mean line
    dispersion_outbreaks_summary_<N>k_beta<beta>.png   : the mean curves together with the min-max range, and the final size of each run
    dispersion_outbreaks_hubs_wastewater_<N>k_beta<beta>.png : the mean friends of the agents infected each week (are hubs infected first?),
                                                              and the total viral shedding per day (the wastewater signal)
The curves are scaled to REPRESENTED_POP people (3.4 million), as if each agent stood for REPRESENTED_POP / N people.

Run from anywhere:  python check_dispersion_outbreaks.py
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

warnings.filterwarnings('ignore', message='.*no "population" column.*')   # harmless here: with a raster the regions come from the raster
logging.getLogger('rasterio._env').setLevel(logging.ERROR)                # the repeated PROJ version message

HERE = Path(__file__).resolve().parent            # network_rank_model/tests
REPO = HERE.parents[1]                            # the repo root
DATA       = REPO/'docs'/'tutorials'/'san-diego-data'        # the San Diego data in the repo
TIF        = DATA/'san_diego_county_ppp_2020.tif'            # people per ~100 m cell: where agents are placed
BOUNDARY   = DATA/'sandiego_zip_boundaries.geojson'          # one region per ZIP code (region_code, region_name)
SEWERSHEDS = DATA/'sandiego_sewersheds.geojson'              # one catchment per treatment plant (sid)
OUT_DIR    = REPO/'network_rank_model'/'results_simulations'
OUT_DIR.mkdir(exist_ok=True)

# Check the data and the settings first, so a missing piece gives a clear message
for f in (TIF, BOUNDARY, SEWERSHEDS):
    if not f.exists():
        raise FileNotFoundError(f'{f} not found. Run  git pull  on the network_rank_model branch to get docs/tutorials/san-diego-data.')
if 'sewershed_file' not in cv.make_pars():
    raise KeyError('This Covasim has no sewershed_file setting. Run  git pull  on the network_rank_model branch.')
boundary = gpd.read_file(BOUNDARY)
missing = {'region_code', 'region_name'} - set(boundary.columns)       # a population column is optional
if missing:
    raise ValueError(f'{BOUNDARY.name} is missing the columns {missing}; it has {boundary.columns.tolist()}')
print(f'Boundary file: {BOUNDARY.name}, {len(boundary)} regions, CRS {boundary.crs}')
print(f'Sewershed file: {SEWERSHEDS.name}')

# ---------------------------------------------------------------------------
# Settings (the same as the earlier plots)
# ---------------------------------------------------------------------------
N         = 20000     # agents
N_DAYS    = 120        # length of the epidemic
SEED      = 1          # first seed (replicate r uses SEED + r)
N_RUNS    = 3          # replicates per setting
BETA      = 0.016      # transmission per contact (the Covasim default). At this value everyone is infected within 120 days; try 0.008 for a slower, less saturated epidemic
REPRESENTED_POP = 3_400_000   # people in the county. Only used to scale the reported numbers and curves: the epidemic itself runs on N agents,
                        # and each agent stands for REPRESENTED_POP / N people. It is not the same as simulating all 3.4 million (see the notes)
N_INITIAL = N // 1000  # agents infected on day 0: 0.1% of the agents, as in the tutorial (20 of 20,000). Too few and runs can die out early
SETTINGS  = [('Poisson (default)', None), ('negative binomial, dispersion 10', 10.0),
             ('negative binomial, dispersion 1', 1.0), ('negative binomial, dispersion 0.5', 0.5)]
COLORS    = ['tab:gray', 'tab:green', 'tab:orange', 'tab:red']
SCALE     = REPRESENTED_POP / N        # people per agent, for the reported numbers and the curves
TAG       = f'{N//1000}k_beta{BETA}'    # in the file names, so runs with different sizes or beta do not overwrite each other
DIED_OUT  = 0.01       # a run counts as died out if fewer than this share of the agents were infected


def run_one(dispersion, seed):
    ''' Build the rank population with the given dispersion and run the epidemic '''
    sim = cv.Sim(pop_type='rank', pop_size=N, pop_infected=N_INITIAL, n_imports=0, beta=BETA, n_days=N_DAYS, rand_seed=seed, verbose=0,
                 density_raster=str(TIF), boundary_shapefile=str(BOUNDARY), sewershed_file=str(SEWERSHEDS))
    extra = dict(use_workers=True)                # the workers change the speed, not the network
    if dispersion is not None:
        extra['dispersion'] = dispersion          # passed through sim.initialize to make_rank_based_contacts
    sim.initialize(**extra)
    layer = sim.people.contacts['a']
    degree = np.bincount(np.concatenate([layer['p1'], layer['p2']]), minlength=N)
    sim.run()
    exposed = np.array(sim.people.date_exposed)                # the day each agent was infected (NaN if never; a reinfection overwrites it)
    ever = int(np.sum(~np.isnan(exposed)))                     # agents infected at least once (a reinfection does not count twice)
    shedding = np.array(sim.results['viral_shedding_hist']).sum(axis=0) if 'viral_shedding_hist' in sim.results else None   # total shedding per day: the wastewater signal
    return sim, degree, ever, exposed, shedding


# ---------------------------------------------------------------------------
# Replicates
# ---------------------------------------------------------------------------
infections = {label: [] for label, _ in SETTINGS}      # new infections per day, one array per run
final      = {label: [] for label, _ in SETTINGS}      # agents infected at least once by the end, one number per run
total_inf  = {label: [] for label, _ in SETTINGS}      # all infections by the end, reinfections included
degree_std = {label: [] for label, _ in SETTINGS}      # std of friends per person, one number per run
hub_by_week = {label: [] for label, _ in SETTINGS}     # mean friends of the agents infected in each week, one array per run
first_deg  = {label: [] for label, _ in SETTINGS}      # mean friends of the first 1% of agents infected, one number per run
waste      = {label: [] for label, _ in SETTINGS}      # total viral shedding per day, one array per run
pop_degree = {}                                        # mean friends per person, for reference
for r in range(N_RUNS):
    for label, d in SETTINGS:
        t0 = time.time()
        sim, degree, ever, exposed, shedding = run_one(d, SEED + r)
        infections[label].append(np.array(sim.results['new_infections'].values))
        final[label].append(ever)
        total_inf[label].append(int(sim.results['cum_infections'][-1]))
        degree_std[label].append(degree.std())
        pop_degree[label] = degree.mean()
        week = np.floor(exposed/7)                                                       # NaN (never infected) matches no week
        hub_by_week[label].append(np.array([degree[week == w].mean() if np.any(week == w) else np.nan for w in range(N_DAYS//7 + 1)]))
        got_it = np.flatnonzero(~np.isnan(exposed))
        first_deg[label].append(degree[got_it[np.argsort(exposed[got_it], kind='stable')[:max(N//100, 1)]]].mean())   # the first 1% infected
        if shedding is not None:
            waste[label].append(shedding)
        days = np.array(sim.results['t'])
        print(f'replicate {r+1}/{N_RUNS} {label:36s} friends std {degree.std():5.1f}, infected by day {N_DAYS}: {ever:,} agents, {total_inf[label][-1]:,} infections ({time.time()-t0:.0f} s)', flush=True)

# ---------------------------------------------------------------------------
# Summary table
# ---------------------------------------------------------------------------
print(f'\n{N:,} agents, {N_RUNS} runs per setting, {N_DAYS} days\n')
print(f'{"":36s} {"friends std":>11} {"agents infected":>16} {"share":>6} {"min":>8} {"max":>8} {"all infections":>15} {"peak/day":>9} {"scaled":>10} {"peak day":>9} {"died out":>9} {"1st 1% friends":>15}')
for label, _ in SETTINGS:
    arr, f = np.array(infections[label]), np.array(final[label])
    print(f'{label:36s} {np.mean(degree_std[label]):>11.1f} {f.mean():>16,.0f} {f.mean()/N:>6.0%} {f.min():>8,} {f.max():>8,} '
          f'{np.mean(total_inf[label]):>15,.0f} {arr.max(axis=1).mean():>9,.0f} {arr.max(axis=1).mean()*SCALE:>10,.0f} {days[arr.argmax(axis=1)].mean():>9.0f} {np.sum(f < DIED_OUT*N):>6}/{N_RUNS} {np.mean(first_deg[label]):>10.1f} ({pop_degree[label]:.1f})')
print('1st 1% friends is the average number of friends of the first 1% of agents infected, with the average for everyone in brackets: a bigger gap means hubs are infected first.')
print(f'friends std shows the setting took effect: it should rise as the dispersion falls. "scaled" is the peak per day for {REPRESENTED_POP:,} people ({SCALE:.0f} people per agent).')

# ---------------------------------------------------------------------------
# Figure 1: one panel per setting
# ---------------------------------------------------------------------------
top = SCALE*max(np.max(a) for arrs in infections.values() for a in arrs)
fig, ax = plt.subplots(2, 2, figsize=(11, 8), sharex=True, sharey=True)
for a, (label, _), color in zip(ax.ravel(), SETTINGS, COLORS):
    arr = np.array(infections[label])
    for line in arr:
        a.plot(days, line*SCALE, color=color, alpha=0.25, lw=0.8)
    a.plot(days, arr.mean(axis=0)*SCALE, color=color, lw=2.5)
    a.set_title(f'{label}\n{np.mean(final[label])/N:.0%} of agents infected', fontsize=10)
    a.set_ylim(0, top*1.05)
for a in ax[1]:
    a.set_xlabel('day')
for a in ax[:, 0]:
    a.set_ylabel(f'new infections per day ({REPRESENTED_POP/1e6:.1f} million people)')
fig.suptitle(f'Rank network, {N:,} agents scaled to {REPRESENTED_POP/1e6:.1f} million people, {N_RUNS} runs per setting (thin = one run, thick = mean)', fontsize=10)
plt.tight_layout(); plt.savefig(OUT_DIR/f'dispersion_outbreaks_spaghetti_{TAG}.png', dpi=150)

# ---------------------------------------------------------------------------
# Figure 2: means together, and the final size of every run
# ---------------------------------------------------------------------------
fig, ax = plt.subplots(1, 2, figsize=(12, 5))
for (label, _), color in zip(SETTINGS, COLORS):
    arr = np.array(infections[label])
    ax[0].plot(days, arr.mean(axis=0)*SCALE, color=color, lw=2.5, label=label)
    ax[0].fill_between(days, arr.min(axis=0)*SCALE, arr.max(axis=0)*SCALE, color=color, alpha=0.12)
ax[0].set_xlabel('day'); ax[0].set_ylabel(f'new infections per day ({REPRESENTED_POP/1e6:.1f} million people)')
ax[0].set_title('Mean over runs, with the min-max range', fontsize=10); ax[0].legend(fontsize=8)
for i, ((label, _), color) in enumerate(zip(SETTINGS, COLORS)):
    f = np.array(final[label])
    ax[1].scatter(np.full(len(f), i) + np.random.uniform(-0.12, 0.12, len(f)), f, color=color, alpha=0.7)
    ax[1].hlines(f.mean(), i - 0.3, i + 0.3, color='black', lw=2)
ax[1].set_xticks(range(len(SETTINGS))); ax[1].set_xticklabels(['Poisson', 'NB 10', 'NB 1', 'NB 0.5'])
ax[1].set_ylabel(f'agents infected by day {N_DAYS}'); ax[1].set_title('Final size of each run (black = mean)', fontsize=10)
plt.tight_layout(); plt.savefig(OUT_DIR/f'dispersion_outbreaks_summary_{TAG}.png', dpi=150)
# ---------------------------------------------------------------------------
# Figure 3: why the curves differ (who is infected when), and the wastewater signal
# ---------------------------------------------------------------------------
fig, ax = plt.subplots(1, 2, figsize=(12, 5))
weeks = np.arange(N_DAYS//7 + 1)
for (label, _), color in zip(SETTINGS, COLORS):
    ax[0].plot(weeks, np.nanmean(np.array(hub_by_week[label]), axis=0), color=color, lw=2.5, marker='o', ms=3, label=label)
    ax[0].axhline(pop_degree[label], color=color, ls=':', lw=1)
ax[0].set_xlabel('week of infection'); ax[0].set_ylabel('mean friends of the agents infected that week')
ax[0].set_title('Hubs are infected first (dotted = mean for everyone)', fontsize=10); ax[0].legend(fontsize=8)
if all(len(waste[label]) for label, _ in SETTINGS):
    for (label, _), color in zip(SETTINGS, COLORS):
        w = np.array(waste[label])
        ax[1].plot(np.arange(w.shape[1]), w.mean(axis=0), color=color, lw=2.5, label=label)
        ax[1].fill_between(np.arange(w.shape[1]), w.min(axis=0), w.max(axis=0), color=color, alpha=0.12)
    ax[1].set_xlabel('day'); ax[1].set_ylabel('total viral shedding (the wastewater signal)')
    ax[1].set_title('Total viral shedding per day, mean and min-max range', fontsize=10)
else:
    ax[1].text(0.5, 0.5, 'viral_shedding_hist not available', ha='center'); ax[1].set_axis_off()
plt.tight_layout(); plt.savefig(OUT_DIR/f'dispersion_outbreaks_hubs_wastewater_{TAG}.png', dpi=150)
print(f'Saved {OUT_DIR/f"dispersion_outbreaks_hubs_wastewater_{TAG}.png"}')
print(f'Saved {OUT_DIR/f"dispersion_outbreaks_spaghetti_{TAG}.png"}\nSaved {OUT_DIR/f"dispersion_outbreaks_summary_{TAG}.png"}')