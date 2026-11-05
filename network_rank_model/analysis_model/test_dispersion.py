'''
CHECK: Poisson against negative binomial friend counts in the rank network.

THE QUESTION
    The first step of building the rank network is to draw how many friendships each person starts. Everything after that
    (who the friends are) is the same whatever distribution is used. This script compares two ways of drawing that number:

        dispersion=None   Poisson (the default). The spread is fixed by the mean: variance = mean.
        dispersion=d      negative binomial with the same mean. Variance = mean + mean^2/d, so a smaller d gives more
                          spread: more people with very few friends and a few people with very many (hubs).

    The mean number of friendships a person starts is the same in both. What changes is the SHAPE of the distribution.
    The final average number of friends comes out a little lower for strong overdispersion: a person who starts many
    friendships keeps picking the same nearest neighbours, and repeated pairs are merged into one friendship.

WORDS USED BELOW
    agent / person     one simulated person
    friends per person ("degree") how many friendships a person ends up in: the ones they start plus the ones others
                       start with them, with repeated pairs counted once
    edge / friendship  one pair of friends; the network is a list of such pairs

WHAT THE SCRIPT DOES
    1. Builds the rank network for Poisson and for three dispersion values (10, 1, 0.5) on the SAME agents with the SAME
       random seed, and prints the friends-per-person statistics for each. Agents are placed from the San Diego
       population-density raster (more agents where more people live), or on made-up clustered points if the raster is missing.
       With San Diego it also prints the median friend distance on the dense coast and in the sparse east. The draw changes
       how many friends a person starts, not who they pick, so those distances should barely move.
    2. Saves a figure of the friends-per-person distributions (normal axis, and log axis to show the tail).
    3. Optionally (RUN_SIM_PIPELINE) repeats the comparison through Covasim itself: cv.Sim(pop_type='rank') followed by
       sim.initialize(dispersion=...), to check that the keyword reaches the rank function from a Sim.

HOW TO READ THE TABLE
    friendships      number of friend pairs (about agents x mean friends / 2, since every friendship has two ends)
    mean, std        average and spread of friends per person
    var/mean         variance divided by mean: near 1 for a Poisson-like spread (a little below, because merging repeated
                     pairs trims the tail) and larger the more uneven the friend counts are
    99th, max        the 99th percentile and the largest number of friends anyone has (the hubs)
    degree<=5, >=40  share of people with very few, and with very many, friends
    coast km, east km  median friend distance in the dense coast (lon < -117.1) and the sparse east (lon > -116.7)

Run:     python test_dispersion.py
Output:  network_rank_model/results_simulations/dispersion_poisson_vs_negbin_<N>k.png
'''
import os
import warnings
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')                          # no display on the server: the figure is saved to a file
import matplotlib.pyplot as plt
import covasim.utils as cvu
from covasim import population as cvpop
from covasim.geography import load_density_raster, sample_coords_from_density, lonlat_to_km

# ---------------------------------------------------------------------------
# Settings
# ---------------------------------------------------------------------------
N, DEG = 20000, 20       # number of agents, and the target friends per person (the mean of the draw)
N_SIM  = 5000            # agents in the full-pipeline check (part 3), which is slower because it builds a whole Sim
# The four ways of drawing each person's number of friendships: (label, dispersion). None means Poisson.
SETTINGS = [('Poisson (default)', None), ('negative binomial, dispersion 10', 10.0),
            ('negative binomial, dispersion 1', 1.0), ('negative binomial, dispersion 0.5', 0.5)]
RUN_SIM_PIPELINE = True  # part 3: check that sim.initialize(dispersion=...) reaches the rank function
USE_SAN_DIEGO    = True  # place agents from the San Diego raster (falls back to made-up points if the file is missing)

# Paths, worked out from where this file is, so the script runs from any folder.
# (This file sits in network_rank_model/<folder>/, so two levels up is the repo root.)
HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
DATA = REPO / 'docs' / 'tutorials' / 'san-diego-data'                  # the San Diego data that comes with the repo
RASTER = DATA / 'san_diego_county_ppp_2020.tif'                        # people per ~100 m cell: where people live
if not RASTER.exists():
    RASTER = REPO / 'network_rank_model' / 'data' / 'san_diego_county_ppp_2020.tif'   # the older location
OUT = REPO / 'network_rank_model' / 'results_simulations' / f'dispersion_poisson_vs_negbin_{N//1000}k.png'   # the size is in the name


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------
def make_points(n, seed=0):
    ''' Made-up agent locations (km) for when there is no raster: a dense core, a looser area and a sparse area '''
    rng = np.random.RandomState(seed)
    area = rng.choice(3, n, p=[0.6, 0.3, 0.1])                                  # which of the 3 areas each agent lives in
    centres = np.array([[0, 0], [15, 5], [40, 10]])[area]                       # the centre of that area
    spread = np.array([4, 8, 15])[area][:, None]                                # how widely agents scatter around it
    return centres + rng.randn(n, 2)*spread


def san_diego_points(n):
    ''' Place n agents from the density raster. Returns their (x, y) in km and their (longitude, latitude) '''
    np.random.seed(0)                                                           # the same agents every run
    raster = load_density_raster(str(RASTER))                                   # the grid of people per cell
    lonlat = sample_coords_from_density(raster, n)                              # a cell for each agent, in proportion to its people
    return lonlat_to_km(lonlat), lonlat                                         # km, so that distances are real distances


def friends_per_person(edges, n):
    ''' Count how many times each person appears in the friendship list (as either end of a pair) '''
    return np.bincount(np.concatenate([edges['p1'], edges['p2']]), minlength=n)


def distance_by_area(edges, xy, lonlat):
    ''' Median friend distance (km) for friendships in the dense coast and in the sparse east, as in test_sandiego.py '''
    p1, p2 = edges['p1'], edges['p2']
    length = np.linalg.norm(xy[p1] - xy[p2], axis=1)                            # how far apart each pair of friends lives
    lon_mid = (lonlat[p1, 0] + lonlat[p2, 0]) / 2                               # where the friendship sits: the mean longitude of its two ends
    coast = np.median(length[lon_mid < -117.1])
    east = np.median(length[lon_mid > -116.7]) if np.any(lon_mid > -116.7) else np.nan   # few agents live in the east: a noisy number
    return coast, east


def describe(label, deg, n_friendships, areas=None):
    ''' Print one row of the table (see HOW TO READ THE TABLE at the top) '''
    row = (f'{label:36s} {n_friendships:>9,} {deg.mean():>7.1f} {deg.std():>6.1f} {deg.var()/deg.mean():>11.2f} '
           f'{np.percentile(deg, 99):>6.0f} {deg.max():>5d} {np.mean(deg <= 5):>10.1%} {np.mean(deg >= 40):>10.1%}')
    if areas is not None:
        row += f' {areas[0]:>12.2f} {areas[1]:>12.2f}'
    print(row)


header = (f'{"":36s} {"friendships":>9} {"mean":>7} {"std":>6} {"var/mean":>11} {"99th":>6} {"max":>5} '
          f'{"degree<=5":>10} {"degree>=40":>10}')


# ---------------------------------------------------------------------------
# 1. Build the rank network directly, for each way of drawing the friend counts
# ---------------------------------------------------------------------------
have_sd = USE_SAN_DIEGO and RASTER.exists()
if have_sd:
    coords, lonlat = san_diego_points(N)
    where = 'San Diego raster'
else:
    coords, lonlat, where = make_points(N), None, 'made-up clustered points'
    if USE_SAN_DIEGO:
        print(f'(raster not found at {RASTER}: using made-up points instead)')

results = {}                                    # friends per person for each setting, kept for the figure
print(f'1. make_rank_based_contacts on {N:,} agents from the {where}, {DEG} target friends per person '
      f'(same agents and seed for each row)\n')
print(header + (f' {"coast km":>12} {"east km":>12}' if have_sd else ''))
for label, dispersion in SETTINGS:
    cvu.set_seed(1)                             # the same random numbers for every setting, so the draw is the only difference
    # use_workers=True builds the network on several CPU cores: it changes the speed, not the network
    edges = cvpop.make_rank_based_contacts(coords, DEG, dispersion=dispersion, use_workers=True)
    results[label] = friends_per_person(edges, N)
    describe(label, results[label], len(edges['p1']), distance_by_area(edges, coords, lonlat) if have_sd else None)
print('\nvar/mean is near 1 (a little below, because repeated pairs are merged) for Poisson, and grows as the dispersion gets smaller.')


# ---------------------------------------------------------------------------
# 2. Figure: the distribution of friends per person for each setting
# ---------------------------------------------------------------------------
fig, ax = plt.subplots(1, 2, figsize=(13, 5))
bins = np.arange(0, max(deg.max() for deg in results.values()) + 4, 2)    # bins reach the largest friend count, so the tail is not cut off
for label, deg in results.items():
    ax[0].hist(deg, bins=bins, histtype='step', lw=2, label=f'{label} (std {deg.std():.1f})')
    ax[1].hist(deg, bins=bins, histtype='step', lw=2, label=label)
ax[0].set_xlabel('friends per person')
ax[0].set_ylabel('number of people')
ax[0].set_title(f'Friend counts: Poisson vs negative binomial ({N:,} agents, {where})', fontsize=10)
ax[0].legend(fontsize=8)
ax[1].set_yscale('log')                         # a log axis makes the rare, very-high-friend-count people visible
ax[1].set_xlabel('friends per person')
ax[1].set_ylabel('number of people (log scale)')
ax[1].set_title('Same, log scale (shows the tail)')
plt.tight_layout()
OUT.parent.mkdir(exist_ok=True)
plt.savefig(OUT, dpi=150)
print(f'\nSaved {OUT}')


# ---------------------------------------------------------------------------
# 3. The whole pipeline: does sim.initialize(dispersion=...) reach the rank function from a Sim?
# ---------------------------------------------------------------------------
# cv.Sim only stores the settings; sim.initialize() builds the people and the network, and any extra keyword given to it
# (such as dispersion) is passed down to make_rank_based_contacts. Without it the network uses the Poisson default.
if RUN_SIM_PIPELINE:
    import covasim as cv
    warnings.filterwarnings('ignore', message='.*no "population" column.*')   # harmless: with a raster the regions come from the raster
    # The San Diego files from the repo, set up as in the tutorial. If the folder is missing, the Sim uses its defaults.
    sim_files = {}
    if DATA.exists():
        sim_files = dict(density_raster=str(RASTER), boundary_shapefile=str(DATA / 'sandiego_zip_boundaries.geojson'),
                         sewershed_file=str(DATA / 'sandiego_sewersheds.geojson'))
    print(f'\n3. Through cv.Sim(pop_type="rank") and sim.initialize(...), {N_SIM:,} agents\n')
    print(header)
    for label, dispersion in (SETTINGS[0], SETTINGS[2]):                       # Poisson, and negative binomial with dispersion 1
        sim = cv.Sim(pop_type='rank', pop_size=N_SIM, n_days=5, rand_seed=1, verbose=0, **sim_files)
        sim.initialize(**({} if dispersion is None else dict(dispersion=dispersion)))
        layer = sim.people.contacts['a']                                       # the single friendship layer
        edges = dict(p1=np.array(layer['p1']), p2=np.array(layer['p2']))
        describe(label, friends_per_person(edges, N_SIM), len(edges['p1']))
    print('\nIf the std and var/mean of the second row are clearly larger than the first, the keyword reaches the rank function.')