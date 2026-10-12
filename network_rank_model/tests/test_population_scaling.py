'''
First Covasim test: does the rank-based layer behave like make_random_contacts
when handed to Covasim, and how does an epidemic differ on the two networks?

Run from network_rank_model/tests:   python test_covasim.py

What this script does, in order:
    1. Places N people on the San Diego map (same method as test_sandiego.py)
    2. Builds TWO friendship layers for the SAME people:
         a) random : cv.population.make_random_contacts(N, DEG)   (Covasim's own)
         b) rank   : make_rank_based_contacts(xy, DEG)            (ours)
    3. Compares them: output format, mean degree, spread of friend counts, friend distance
    4. Runs N_RUNS short Covasim epidemics on each network (same people, new network + seed each run)
    5. Saves a 2-panel figure: friend distances (averaged over runs) and infections over time

POPULATION SCALING
    We simulate N agents but the real county has N_REAL people. Covasim's pop_scale makes each
    agent stand for POP_SCALE = N_REAL / N real people, so infection counts are reported at
    county scale (millions), not agent scale (thousands). rescale=True means the scale starts at 1
    (so a seed of 2 infections is 2 real people) and ramps up as the epidemic grows.
    NOTE: pop_scale only scales the REPORTED RESULTS. The rank network is still built from the
    N-agent sample, so friend distances in km still depend on N (fixed separately in rank_network.py).

Nothing here edits population.py yet. The layers are passed to Covasim through
a hand-built popdict (the dictionary format make_people() accepts).

What to look for:
    - Format check: both give dict(p1, p2) with the same integer dtype
    - Mean degree: both a little below DEG
    - Friend distance: random ~ county-wide (tens of km); rank ~ a few km
    - Epidemic: the clustered rank network usually spreads differently from the random one
'''
import numpy as np
import matplotlib.pyplot as plt
import covasim as cv
from covasim.population import make_random_contacts   # Covasim's own function we are replicating

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent / 'src'))

from rank_network_scaled import load_density_raster, sample_coords_from_density, lonlat_to_km, make_rank_based_contacts


# ---------------------------------------------------------------------------
# Settings
# ---------------------------------------------------------------------------
TIF       = '../data/san_diego_county_ppp_2020.tif'   # people-per-cell raster, already clipped to the county
N         = 200000      # simulated agents; keep small while testing (rank network time grows with N squared)
N_REAL    = 3400000  # real county population (the raster total is about 3.40 million)
POP_SCALE = N_REAL / N # each agent stands for this many real people (N = 20,000 gives 170)
DEG       = 20         # target average friends per person. Changed from 10 to 20 to match make_random_contacts()
N_DAYS    = 60         # length of the epidemic
SEED      = 1          # first random seed for the epidemic runs (run r uses SEED + r)
N_RUNS    = 10         # replicate epidemics per network
INIT_INFECTED = 10      # people infected on day 0. With rescale=True each agent starts as ONE real person,
                       # so this is a small, realistic seed, not 2 x POP_SCALE people

np.random.seed(0)  # same people and networks every run


# ---------------------------------------------------------------------------
# 1. Place people on the map
# ---------------------------------------------------------------------------
raster = load_density_raster(TIF)
lonlat = sample_coords_from_density(raster, N)   # (N,2) lon/lat in degrees
xy     = lonlat_to_km(lonlat)                    # (N,2) km, so distances are real distances

# Ages and sexes are not part of the network test, so keep them simple
ages  = np.random.uniform(0, 90, N)
sexes = np.random.binomial(1, 0.5, N)


# ---------------------------------------------------------------------------
# 2. Build the two layers
# ---------------------------------------------------------------------------
layers = {
    'random': make_random_contacts(N, DEG),      # needs only N and DEG
    'rank':   make_rank_based_contacts(xy, DEG, n_real=N_REAL), # also needs where people live
}


# ---------------------------------------------------------------------------
# 3. Compare the layers
# ---------------------------------------------------------------------------
stats = {}
for name, e in layers.items():
    p1, p2 = e['p1'], e['p2']
    degree = np.bincount(np.concatenate([p1, p2]), minlength=N)   # friends per person
    dist   = np.linalg.norm(xy[p1] - xy[p2], axis=1)              # friendship lengths in km
    stats[name] = dict(degree=degree, dist=dist)
    print(f'{name:6s}: keys={sorted(e.keys())}, dtype={p1.dtype}, edges={len(p1):,}, '
          f'mean degree={degree.mean():.2f}, std={degree.std():.2f}, max={degree.max()}, '
          f'median friend distance={np.median(dist):.2f} km')


# ---------------------------------------------------------------------------
# 4. Run Covasim on each network
# ---------------------------------------------------------------------------
def run_sim(name, edges, seed):
    '''Build a popdict around one friendship layer ('f') and run a short epidemic.'''
    popdict = dict(
        uid    = np.arange(N),
        age    = ages,
        sex    = sexes,
        region = np.zeros(N, dtype=int),   # this fork's make_people() reads popdict['region']
        x      = xy[:, 0],                 # this fork forwards x,y to People if present
        y      = xy[:, 1],
        contacts   = {'f': edges},         # our single layer, called 'f' for friends
        layer_keys = ['f'],
    )
    pars = dict(
        pop_size     = N,                  # number of AGENTS actually simulated
        pop_type     = 'random',           # ignored when a popdict is supplied; kept so validation passes
        pop_scale    = POP_SCALE,          # each agent represents this many real people in the results
        rescale      = True,               # dynamic rescaling: start with 1 agent = 1 person, ramp up to POP_SCALE as the epidemic grows
        pop_infected = INIT_INFECTED,      # people infected on day 0 (the scale starts at 1, so agents = real people here)
        n_days       = N_DAYS,
        rand_seed    = seed,
        verbose      = 0,
    )
    sim = cv.Sim(pars, label=name)

    # Covasim MERGES layer parameters with its defaults, which hold a layer 'a'. That leaves the
    # layer keys as ['a', 'f'] while our popdict only has 'f'. Overwrite them so 'f' is the only layer.
    sim.pars['contacts']    = {'f': DEG}   # average friends per person
    sim.pars['beta_layer']  = {'f': 1.0}   # relative transmission risk on this layer
    sim.pars['quar_factor'] = {'f': 1.0}   # contact reduction under quarantine
    sim.pars['iso_factor']  = {'f': 1.0}   # contact reduction under isolation
    sim.pars['dynam_layer'] = {'f': 0}     # 0 = static layer (contacts do not change over time)
    sim.popdict = popdict   # this version of Sim has no popdict argument; make_people() picks it up from sim.popdict
    sim.run()
    return sim

# Epidemics are random, so a single run says little. Run N_RUNS replicates per network.
# Each replicate rebuilds BOTH networks (new random draw) and uses a new epidemic seed, so the
# spread between runs includes network-to-network variation as well as epidemic randomness.
runs      = {'random': [], 'rank': []}   # new infections per day (county scale), one array per run
cum_final = {'random': [], 'rank': []}   # cumulative infections at the end (county scale), one number per run
hist_bins = np.linspace(0, 50, 100)      # friend-distance bins: 0-50 km in 0.5 km steps
dist_hist = {'random': [], 'rank': []}   # edge-length histogram counts, one array per run
n_edges   = {'random': [], 'rank': []}   # total number of friendships, one number per run
for r in range(N_RUNS):
    np.random.seed(100 + r)              # new networks each replicate
    rep_layers = {'random': make_random_contacts(N, DEG),
                  'rank':   make_rank_based_contacts(xy, DEG, n_real=N_REAL)}
    for name, e in rep_layers.items():
        sim = run_sim(name, e, seed=SEED + r)
        runs[name].append(np.array(sim.results['new_infections'].values))
        cum_final[name].append(int(sim.results['cum_infections'][-1]))
        d = np.linalg.norm(xy[e['p1']] - xy[e['p2']], axis=1)       # friendship lengths (km) in this run's network
        dist_hist[name].append(np.histogram(d, bins=hist_bins)[0])  # counts per distance bin
        n_edges[name].append(len(e['p1']))
        days = np.array(sim.results['t'])
    print(f'replicate {r+1}/{N_RUNS} done (scale went from {sim.rescale_vec[0]:.0f} to {sim.rescale_vec[-1]:.0f})')

# Results are at county scale, so compare against N_REAL (not N)
for name, vals in cum_final.items():
    print(f'{name:6s}: cumulative infections after {N_DAYS} days = {np.mean(vals):,.0f} +/- {np.std(vals):,.0f} '
          f'(min {min(vals):,}, max {max(vals):,}) of {N_REAL:,}  ({np.mean(vals)/N_REAL:.0%})')


# ---------------------------------------------------------------------------
# 5. Plot
# ---------------------------------------------------------------------------
colors = {'random': 'tab:gray', 'rank': 'tab:blue'}
fig, ax = plt.subplots(1, 2, figsize=(11, 5))

# Left: friendship lengths, AVERAGED over the N_RUNS networks built above (bar height = mean count per bin).
# Random friends live anywhere in the county; rank friends are mostly close by.
# (Counts are for the N simulated agents, not scaled to county size.)
centers = (hist_bins[:-1] + hist_bins[1:]) / 2
width   = hist_bins[1] - hist_bins[0]
for name, h in dist_hist.items():
    ax[0].bar(centers, np.mean(h, axis=0), width=width, alpha=0.6, label=name, color=colors[name])
ax[0].set_xlabel('friend distance (km)'); ax[0].set_ylabel('number of friendships (mean of runs)')
ax[0].set_title(f"Num friendships vs friend distance\n"
                f"Average total edges over {N_RUNS} runs: random = {np.mean(n_edges['random']):,.0f}, "
                f"rank = {np.mean(n_edges['rank']):,.0f}", fontsize=10)
ax[0].legend()

# Right: "spaghetti plot". Each thin line is one epidemic; the thick line is the mean over runs.
# Same people and parameters for both networks; only the network changes. Counts are at county scale.
for name, arrs in runs.items():
    arr = np.array(arrs)                                   # shape (N_RUNS, days)
    for a_ in arr:
        ax[1].plot(days, a_, color=colors[name], alpha=0.2, lw=0.8)
    ax[1].plot(days, arr.mean(axis=0), color=colors[name], lw=2.5, label=f'{name} (mean of {N_RUNS})')
ax[1].set_xlabel('day'); ax[1].set_ylabel('new infections per day (county scale)')
ax[1].set_title(f'Epidemic: {N_RUNS} runs per network', fontsize=10) #\n(thin = one run, thick = mean)
ax[1].legend(loc='center right')   # keeps the legend clear of the text box in the top right

# Text box with the settings and mean outcome, top right of the epidemic plot
summary = (f'{N:,} agents, scaled up to {N_REAL:,} people\n'
           f'target contacts = {DEG}, beta = {sim.pars["beta"]}\n'
           f'initial infections = {INIT_INFECTED}, {N_RUNS} runs per network\n'
           + '\n'.join(f'{name}: mean {np.mean(v):,.0f} infected ({np.mean(v)/N_REAL:.0%}) by day {N_DAYS}'
                       for name, v in cum_final.items()))
ax[1].text(0.97, 0.97, summary, transform=ax[1].transAxes, ha='right', va='top',
           fontsize=7, family='monospace', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

plt.tight_layout(); plt.savefig('../data/multi_sim_compare_scaled.png', dpi=150)
print('Saved ../data/multi_sim_compare_scaled.png')