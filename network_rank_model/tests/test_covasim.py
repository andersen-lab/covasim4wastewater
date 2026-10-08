'''
First Covasim test: does the rank-based layer behave like make_random_contacts
when handed to Covasim, and how does an epidemic differ on the two networks?

Run from network_rank_model/src:   python test_covasim.py

What this script does, in order:
    1. Places N people on the San Diego map (same method as test_sandiego.py)
    2. Builds TWO friendship layers for the SAME people:
         a) random : cv.population.make_random_contacts(N, DEG)   (Covasim's own)
         b) rank   : make_rank_based_contacts(xy, DEG)            (ours)
    3. Compares them: output format, mean degree, spread of friend counts, friend distance
    4. Runs a short Covasim epidemic on each network (same people, same seed)
    5. Saves a 3-panel figure: degree distributions, friend distances, infections over time

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
from rank_network import load_density_raster, sample_coords_from_density, lonlat_to_km, make_rank_based_contacts


# ---------------------------------------------------------------------------
# Settings
# ---------------------------------------------------------------------------
TIF    = '../data/san_diego_county_ppp_2020.tif'   # people-per-cell raster, already clipped to the county
N      = 5000      # people; keep small while testing (rank network time grows with N squared)
DEG    = 10        # target average friends per person
N_DAYS = 60        # length of the epidemic
SEED   = 1         # random seed for the epidemic runs

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
    'rank':   make_rank_based_contacts(xy, DEG), # also needs where people live
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
def run_sim(name, edges):
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
        pop_infected = 10,                 # people infected on day 0
        n_days       = N_DAYS,
        rand_seed    = SEED,
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

sims = {name: run_sim(name, e) for name, e in layers.items()}
for name, sim in sims.items():
    print(f'{name:6s}: cumulative infections after {N_DAYS} days = {int(sim.results["cum_infections"][-1]):,} of {N:,}')


# ---------------------------------------------------------------------------
# 5. Plot v1
# ---------------------------------------------------------------------------
colors = {'random': 'tab:gray', 'rank': 'tab:blue'}
fig, ax = plt.subplots(1, 3, figsize=(17, 5))

# Left: how many friends each person has. Random is bell-shaped around DEG;
# the rank network is usually wider (people in dense areas get more links).
for name, s in stats.items():
    ax[0].hist(s['degree'], bins=np.arange(0, 40), alpha=0.6, label=name, color=colors[name])
ax[0].set_xlabel('friends per person'); ax[0].set_ylabel('number of people')
ax[0].set_title('Degree distribution'); ax[0].legend()

# Middle: friendship lengths. Random friends live anywhere in the county;
# rank friends are mostly close by.
for name, s in stats.items():
    ax[1].hist(s['dist'], bins=np.linspace(0, 50, 100), alpha=0.6, label=name, color=colors[name])
ax[1].set_xlabel('friend distance (km)'); ax[1].set_ylabel('number of friendships')
ax[1].set_title(f"Edge Length. Total num of edges (friendships): "
                f"random = {len(layers['random']['p1']):,}, rank = {len(layers['rank']['p1']):,}",
                fontsize=10)
ax[1].legend()

# Right: epidemic curves on each network (same people, same seed, same parameters)
for name, sim in sims.items():
    ax[2].plot(sim.results['t'], sim.results['new_infections'].values, label=name, color=colors[name])
ax[2].set_xlabel('day'); ax[2].set_ylabel('new infections per day')
ax[2].set_title('Epidemic on each network'); ax[2].legend()

#plt.tight_layout(); plt.savefig('../data/covasim_test.png', dpi=150)
#print('Saved ../data/covasim_test.png')

# ---------------------------------------------------------------------------
# 5. Plot saved
# ---------------------------------------------------------------------------
colors = {'random': 'tab:gray', 'rank': 'tab:blue'}
fig, ax = plt.subplots(1, 2, figsize=(11, 5))

# Middle: friendship lengths. Random friends live anywhere in the county;
# rank friends are mostly close by.
for name, s in stats.items():
    ax[0].hist(s['dist'], bins=np.linspace(0, 50, 100), alpha=0.6, label=name, color=colors[name])
ax[0].set_xlabel('friend distance (km)'); ax[0].set_ylabel('number of friendships')
#ax[0].set_title(f"Edge Length. Total num of edges (friendships): "
#                f"random = {len(layers['random']['p1']):,}, rank = {len(layers['rank']['p1']):,}",fontsize=10)
ax[0].set_title(f"Num friendships vs friend distance\n"
                f"Total edges: random = {len(layers['random']['p1']):,}, rank = {len(layers['rank']['p1']):,}",
                fontsize=10)
ax[0].legend()

# Right: epidemic curves on each network (same people, same seed, same parameters)
for name, sim in sims.items():
    ax[1].plot(sim.results['t'], sim.results['new_infections'].values, label=name, color=colors[name])
ax[1].set_xlabel('day'); ax[1].set_ylabel('new infections per day')
ax[1].set_title('Epidemic on each network'); ax[1].legend()

plt.tight_layout(); plt.savefig('../data/covasim_comparison_test.png', dpi=150)
print('Saved ../data/covasim_comparison_test.png')