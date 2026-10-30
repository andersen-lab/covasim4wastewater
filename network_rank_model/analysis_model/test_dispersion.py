'''
CHECK: Poisson (the default) against negative binomial friend counts in the rank network.

make_rank_based_contacts(coords, n, dispersion=...) draws how many friendships each person starts.
    dispersion=None   Poisson. The spread is fixed by the mean: variance = mean. The default.
    dispersion=d      negative binomial with the same mean n. The variance is mean + mean^2/d, so a smaller d
                      gives more spread: more people with very few friends and more with very many.
The average number of friendships a person starts is the same; the SHAPE of the friend-count distribution changes.
The final average comes out a little lower for strong overdispersion: people who start many friendships keep picking the
same nearest neighbours, and repeated pairs are merged.

This script:
  1. Builds the rank network with Poisson and with several dispersion values, using the same random seed, and prints
     the friend-count (degree) statistics for each. The agents are placed from the San Diego population-density raster
     if it exists (USE_SAN_DIEGO), otherwise on made-up clustered points. With San Diego it also prints the median friend
     distance in the dense coast and the sparse east, as in test_sandiego.py: dispersion changes how many friends a
     person starts, not who they pick, so those distances should barely move.
  2. Saves a figure of the degree distributions.
  3. Optionally (RUN_SIM_PIPELINE) does the same through the whole pipeline, cv.Sim(pop_type='rank') followed by
     sim.initialize(dispersion=...), to check the keyword reaches the rank function from a Sim.

Run:  python check_dispersion.py
Output: network_rank_model/results_simulations/dispersion_poisson_vs_negbin.png
'''
import os
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import covasim.utils as cvu
from covasim import population as cvpop
from covasim.geography import load_density_raster, sample_coords_from_density, lonlat_to_km

#N, DEG = 20000, 20             # agents and target friends per person (the pipeline check in part 3 uses 5,000 agents)
N, DEG = 100000, 20             # agents and target friends per person (the pipeline check in part 3 uses 5,000 agents)
SETTINGS = [('Poisson (default)', None), ('negative binomial, dispersion 10', 10.0),
            ('negative binomial, dispersion 1', 1.0), ('negative binomial, dispersion 0.5', 0.5)]
RUN_SIM_PIPELINE = True        # needs the real Covasim and the example boundary file in the repo's tests folder
USE_SAN_DIEGO = True           # place the agents from the San Diego raster (falls back to made-up points if the file is missing)
HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
RASTER = REPO / 'network_rank_model' / 'data' / 'san_diego_county_ppp_2020.tif'
OUT = REPO / 'network_rank_model' / 'results_simulations' / 'dispersion_poisson_vs_negbin_100k.png'


def make_points(n, seed=0):
    ''' A dense core, a looser area and a sparse area '''
    rng = np.random.RandomState(seed)
    k = rng.choice(3, n, p=[0.6, 0.3, 0.1])
    return np.array([[0, 0], [15, 5], [40, 10]])[k] + rng.randn(n, 2)*np.array([4, 8, 15])[k][:, None]


def san_diego_points(n):
    ''' Agents placed from the density raster: more where more people live. Returns km coordinates and lon/lat. '''
    np.random.seed(0)
    lonlat = sample_coords_from_density(load_density_raster(str(RASTER)), n)
    return lonlat_to_km(lonlat), lonlat


def distance_by_area(edges, xy, lonlat):
    ''' Median friend distance (km) in the dense coast (lon < -117.1) and the sparse east (lon > -116.7), as in test_sandiego.py '''
    p1, p2 = edges['p1'], edges['p2']
    dist = np.linalg.norm(xy[p1] - xy[p2], axis=1)
    lon_mid = (lonlat[p1, 0] + lonlat[p2, 0]) / 2
    return np.median(dist[lon_mid < -117.1]), (np.median(dist[lon_mid > -116.7]) if np.any(lon_mid > -116.7) else np.nan)


def degrees(edges, n):
    return np.bincount(np.concatenate([edges['p1'], edges['p2']]), minlength=n)


def describe(label, deg, n_edges, areas=None):
    line = (f'{label:36s} {n_edges:>9,} {deg.mean():>7.1f} {deg.std():>6.1f} {deg.var()/deg.mean():>11.2f} '
            f'{np.percentile(deg, 99):>6.0f} {deg.max():>5d} {np.mean(deg <= 5):>10.1%} {np.mean(deg >= 40):>10.1%}')
    if areas is not None:
        line += f' {areas[0]:>12.2f} {areas[1]:>12.2f}'
    print(line)


header = f'{"":36s} {"friendships":>9} {"mean":>7} {"std":>6} {"var/mean":>11} {"99th":>6} {"max":>5} {"degree<=5":>10} {"degree>=40":>10}'

# ---- 1. the function, directly ----
have_sd = USE_SAN_DIEGO and RASTER.exists()
if have_sd:
    coords, lonlat = san_diego_points(N)
    where = 'San Diego raster'
else:
    coords, lonlat, where = make_points(N), None, 'made-up clustered points'
    if USE_SAN_DIEGO:
        print(f'(raster not found at {RASTER}: using made-up points instead)')
results = {}
print(f'1. make_rank_based_contacts on {N:,} agents from the {where}, {DEG} target friends per person (same seed for each row)\n')
print(header + (f' {"coast km":>12} {"east km":>12}' if have_sd else ''))
for label, d in SETTINGS:
    cvu.set_seed(1)
    edges = cvpop.make_rank_based_contacts(coords, DEG, dispersion=d, use_workers=True)   # the workers change the speed, not the network
    results[label] = degrees(edges, N)
    describe(label, results[label], len(edges['p1']), distance_by_area(edges, coords, lonlat) if have_sd else None)
print('\nvar/mean is near 1 (a little below, because duplicate pairs are merged) for Poisson, and grows as the dispersion gets smaller.')

# ---- 2. figure ----
fig, ax = plt.subplots(1, 2, figsize=(13, 5))
bins = np.arange(0, 91, 2)
for label, deg in results.items():
    ax[0].hist(deg, bins=bins, histtype='step', lw=2, label=f'{label} (std {deg.std():.1f})')
    ax[1].hist(deg, bins=bins, histtype='step', lw=2, label=label)
ax[0].set_xlabel('friends per person'); ax[0].set_ylabel('number of people'); ax[0].set_title(f'Friend counts: Poisson vs negative binomial ({where})', fontsize=10)
ax[0].legend(fontsize=8)
ax[1].set_yscale('log'); ax[1].set_xlabel('friends per person'); ax[1].set_ylabel('number of people (log scale)'); ax[1].set_title('Same, log scale (shows the tail)')
plt.tight_layout(); OUT.parent.mkdir(exist_ok=True); plt.savefig(OUT, dpi=150)
print(f'\nSaved {OUT}')

# ---- 3. the whole pipeline: does sim.initialize(dispersion=...) reach the rank function? ----
if RUN_SIM_PIPELINE:
    import covasim as cv
    os.chdir(REPO / 'tests')                      # the default boundary and sewershed files are looked up from here
    print('\n3. Through cv.Sim(pop_type="rank") and sim.initialize(...), 5,000 agents\n')
    print(header)
    for label, d in (SETTINGS[0], SETTINGS[2]):
        sim = cv.Sim(pop_type='rank', pop_size=N, n_days=5, rand_seed=1, verbose=0)
        sim.initialize(**({} if d is None else dict(dispersion=d)))
        layer = sim.people.contacts['a']
        describe(label, degrees(dict(p1=np.array(layer['p1']), p2=np.array(layer['p2'])), N), len(layer['p1']))
    print('\nIf the std and var/mean of the second row are clearly larger than the first, the keyword reaches the rank function.')