'''
Does the rank-based layer behave like make_random_contacts when handed to Covasim, and does the NEW faster rank function
give the same network and the same epidemics whether it uses one core or several?

Run from network_rank_model/tests:   python compare_covasim_pops_multi_sim.py

Three friendship layers for the SAME people:
    1) random              Covasim's make_random_contacts()
    2) rank, new           the fast rank function on one core        (use_workers=False, the default)
    3) rank, new, workers  the fast rank function with use_workers=True (the computer chooses how many cores)

What this script does, in order:
    1. Places N people on the San Diego map
    2. Builds the three layers and compares them (format, mean degree, friend distance, build time)
    3. Runs N_RUNS Covasim epidemics on each network: each replicate rebuilds the networks and uses a new epidemic seed
    4. Saves TWO figures, each with 2 panels (friend distances averaged over runs, and infections over time):
         random vs rank (one core)   and   random vs rank (use_workers=True)

Layers 2 and 3 are built from the same random seed, so they should be the SAME network, and so give the same epidemic.
The script checks that. Layer 3 should just be built faster. If the two networks are identical, the epidemic for layer 3
is not run again (REUSE_IF_IDENTICAL), because the same network and seed give exactly the same epidemic.

Nothing here edits population.py. The layers are passed to Covasim through a hand-built popdict.
'''
import sys
import time
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import covasim as cv
import covasim.utils as cvu
from covasim.population import make_random_contacts   # Covasim's own function

sys.path.append(str(Path(__file__).resolve().parent.parent / 'src'))
from rank_network      import load_density_raster, sample_coords_from_density, lonlat_to_km
from rank_network_fast import make_rank_based_contacts_fast as make_rank_contacts   # the new, faster rank function


# ---------------------------------------------------------------------------
# Settings
# ---------------------------------------------------------------------------
TIF       = '../data/san_diego_county_ppp_2020.tif'   # people-per-cell raster, already clipped to the county
N         = 20000      # people
DEG       = 20         # target average friends per person
N_DAYS    = 60         # length of the epidemic
SEED      = 1          # first random seed for the epidemic runs (run r uses SEED + r)
N_RUNS    = 10         # replicate epidemics per network
N_REAL    = 3_400_000  # real county population (the raster total is about 3.40 million)
POP_SCALE = N_REAL / N # each agent stands for this many real people (N = 20,000 gives 170)
REUSE_IF_IDENTICAL = True   # if layer 3 is exactly the same network as layer 2, reuse its epidemic instead of running it again
OUT_DIR   = Path('../results_simulations')
OUT_FILES = {'rank':         OUT_DIR/'multi_sim_compare_covasim_20k_10runs_fast_one_core.png',
             'rank_workers': OUT_DIR/'multi_sim_compare_covasim_20k_10runs_fast_use_workers.png'}

NAMES  = ['random', 'rank', 'rank_workers']
LABELS = {'random': 'random', 'rank': 'rank, new (one core)', 'rank_workers': 'rank, new (use_workers=True)'}
COLORS = {'random': 'tab:gray', 'rank': 'tab:blue', 'rank_workers': 'tab:green'}

OUT_DIR.mkdir(exist_ok=True)
np.random.seed(0)  # same people every run


def build_layers(seed, xy):
    '''The three layers for one replicate, with the time each took to build. The two rank layers use the same seed.'''
    layers, seconds = {}, {}
    cvu.set_seed(seed)
    t = time.time(); layers['random'] = make_random_contacts(N, DEG); seconds['random'] = time.time() - t
    cvu.set_seed(seed + 1)                                                    # same seed for both rank builds
    t = time.time(); layers['rank'] = make_rank_contacts(xy, DEG); seconds['rank'] = time.time() - t
    cvu.set_seed(seed + 1)
    t = time.time(); layers['rank_workers'] = make_rank_contacts(xy, DEG, use_workers=True); seconds['rank_workers'] = time.time() - t
    return layers, seconds


def same_network(a, b):
    return np.array_equal(a['p1'], b['p1']) and np.array_equal(a['p2'], b['p2'])


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
# 2. Build the three layers once and compare them
# ---------------------------------------------------------------------------
layers, seconds = build_layers(100, xy)
for name, e in layers.items():
    p1, p2 = e['p1'], e['p2']
    degree = np.bincount(np.concatenate([p1, p2]), minlength=N)   # friends per person
    dist   = np.linalg.norm(xy[p1] - xy[p2], axis=1)              # friendship lengths in km
    print(f'{LABELS[name]:28s}: dtype={p1.dtype}, edges={len(p1):,}, mean degree={degree.mean():.2f}, std={degree.std():.2f}, '
          f'max={degree.max()}, median friend distance={np.median(dist):.2f} km, built in {seconds[name]:.2f} s')
print(f'one-core and use_workers=True rank networks identical: {same_network(layers["rank"], layers["rank_workers"])}\n')


# ---------------------------------------------------------------------------
# 3. Run Covasim on each network
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
        pop_size     = N,
        pop_type     = 'random',           # ignored when a popdict is supplied; kept so validation passes
        pop_scale    = POP_SCALE,
        pop_infected = 2,                  # people infected on day 0
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


# Each replicate rebuilds ALL networks (new random draw) and uses a new epidemic seed, so the spread between runs
# includes network-to-network variation as well as epidemic randomness.
runs      = {n: [] for n in NAMES}   # new infections per day, one array per run
cum_final = {n: [] for n in NAMES}   # cumulative infections at the end, one number per run
build_s   = {n: [] for n in NAMES}   # seconds taken to build the network, one number per run
dist_hist = {n: [] for n in NAMES}   # edge-length histogram counts, one array per run
n_edges   = {n: [] for n in NAMES}   # total number of friendships, one number per run
hist_bins = np.linspace(0, 50, 100)  # friend-distance bins: 0-50 km in 0.5 km steps
all_identical = True

for r in range(N_RUNS):
    rep_layers, rep_seconds = build_layers(100 + 2*r, xy)
    identical = same_network(rep_layers['rank'], rep_layers['rank_workers'])
    all_identical &= identical
    for name in NAMES:
        e = rep_layers[name]
        if name == 'rank_workers' and identical and REUSE_IF_IDENTICAL:
            runs[name].append(runs['rank'][-1]); cum_final[name].append(cum_final['rank'][-1])   # same network and seed: same epidemic
        else:
            sim = run_sim(name, e, seed=SEED + r)
            runs[name].append(np.array(sim.results['new_infections'].values))
            cum_final[name].append(int(sim.results['cum_infections'][-1]))
            days = np.array(sim.results['t'])
        d = np.linalg.norm(xy[e['p1']] - xy[e['p2']], axis=1)       # friendship lengths (km) in this run's network
        dist_hist[name].append(np.histogram(d, bins=hist_bins)[0])  # counts per distance bin
        n_edges[name].append(len(e['p1']))
        build_s[name].append(rep_seconds[name])
    print(f'replicate {r+1}/{N_RUNS} done (rank networks identical: {identical})', flush=True)

print()
for name in NAMES:
    v = cum_final[name]
    print(f'{LABELS[name]:28s}: infected after {N_DAYS} days = {np.mean(v):,.0f} +/- {np.std(v):,.0f} (min {min(v):,}, max {max(v):,}) of {N:,};'
          f' network built in {np.mean(build_s[name]):.2f} s on average')
print('The two new rank functions gave the same network in every replicate.' if all_identical else
      'WARNING: the one-core and use_workers=True rank networks were NOT identical in every replicate.')


# ---------------------------------------------------------------------------
# 4. Plots: random against ONE rank network each time
# ---------------------------------------------------------------------------
hist_centers = (hist_bins[:-1] + hist_bins[1:]) / 2
hist_width   = hist_bins[1] - hist_bins[0]


def make_plot(rank_name, path):
    '''Two panels, random against one kind of rank network: friend distances (mean of runs) and the epidemics.'''
    pair = ['random', rank_name]
    fig, ax = plt.subplots(1, 2, figsize=(11, 5.5))

    # Left: friendship lengths, AVERAGED over the N_RUNS networks (bar height = mean count per bin)
    for name in pair:
        ax[0].bar(hist_centers, np.mean(dist_hist[name], axis=0), width=hist_width, alpha=0.6, label=LABELS[name], color=COLORS[name])
    ax[0].set_xlabel('friend distance (km)'); ax[0].set_ylabel('number of friendships (mean of runs)')
    ax[0].set_title(f"Num friendships vs friend distance\nAverage total edges over {N_RUNS} runs: random = {np.mean(n_edges['random']):,.0f}, "
                    f"rank = {np.mean(n_edges[rank_name]):,.0f}", fontsize=10)
    ax[0].legend()

    # Right: "spaghetti plot". Each thin line is one epidemic; the thick line is the mean over runs.
    for name in pair:
        arr = np.array(runs[name])                                   # shape (N_RUNS, days)
        for a_ in arr:
            ax[1].plot(days, a_, color=COLORS[name], alpha=0.2, lw=0.8)
        ax[1].plot(days, arr.mean(axis=0), color=COLORS[name], lw=2.5, label=f'{LABELS[name]} (mean of {N_RUNS})')
    ax[1].set_xlabel('day'); ax[1].set_ylabel('new infections per day')
    ax[1].set_title(f'Epidemic: {N_RUNS} runs per network', fontsize=10)
    ax[1].legend(loc='center left', fontsize=8)

    # Text box with the settings, the mean outcome and how long each network took to build
    summary = (f'N = {N:,}, target contacts = {DEG}, beta = {cv.make_pars()["beta"]}\n'
               f'initial infections = 2, {N_RUNS} runs per network\n'
               + '\n'.join(f'{LABELS[name]}: {np.mean(cum_final[name]):,.0f} infected ({np.mean(cum_final[name])/N:.0%}), built in {np.mean(build_s[name]):.2f} s'
                           for name in pair))
    ax[1].text(0.99, 0.99, summary, transform=ax[1].transAxes, ha='right', va='top', fontsize=7, family='monospace',
               bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    plt.tight_layout(); plt.savefig(path, dpi=150); plt.close(fig)
    print(f'Saved {path}')


for rank_name, path in OUT_FILES.items():
    make_plot(rank_name, path)