'''
Smoke test: rank-based friendship network on WorldPop density for San Diego County.
Run from network_rank_model/src:   python test_sandiego.py

What this script does, in order (and what it prints):
    1. Reads the population raster (a grid of "people per cell")       -> prints raster shape + total population
    2. Places N simulated people on the map, more where more people live
    3. Builds the friendship network with the rank rule               -> prints edge count, runtime, mean degree
    4. Measures how far apart friends live, west (dense) vs east (sparse) -> prints median friend distance for each
    5. Saves a two-panel figure: where the agents are, and how long the friendships are

What a good result looks like:
    - Total pop in bbox       : about 3.3-3.5 million (the county's real population is ~3.3 million)
    - Mean degree             : a little below DEG (about 9.2-9.5 for DEG = 10); duplicates are merged
    - Median friend distance  : short in the dense west, several times longer in the sparse east
'''
import time                                   # for timing how long the network takes to build
import numpy as np                            # arrays and maths
import matplotlib.pyplot as plt               # for the figure at the end
from rank_network import load_density_raster, sample_coords_from_density, lonlat_to_km, make_rank_based_contacts


# ---------------------------------------------------------------------------
# Settings
# ---------------------------------------------------------------------------
TIF  = '../data/san_diego_county_ppp_2020.tif'   # raster file: people per ~100 m cell, already clipped to the county
BBOX = None                                      # None = read the whole file (it is already clipped to the county)
#TIF  = '../data/usa_ppp_2020.tif'               # alternative: the whole-US raster (3.8 GB) ...
#BBOX = (-117.6, 32.53, -116.08, 33.51)          # ... read only this window: lon_min, lat_min, lon_max, lat_max
N    = 100000 #20000   # number of simulated people (was 5000). Runtime grows with N squared:
               # 5,000 = ~1 s, 20,000 = ~20 s, 50,000 = ~2 min
DEG  = 10      # target average number of friends per person (the actual average comes out a bit lower)

np.random.seed(0)   # fix the random numbers so reruns give the same population and network


# ---------------------------------------------------------------------------
# 1. Raster sanity check
# ---------------------------------------------------------------------------
# Read the raster. 'raster' is a dict holding the grid of people-per-cell ('counts')
# plus the longitude/latitude of each cell, so we know where each number sits on the map.
raster = load_density_raster(TIF, BBOX)

# Prints: (number of rows, number of columns) of the grid, and the sum of all cells
# = estimated total population in the file.
print(f'Raster shape {raster["counts"].shape}, total pop in bbox: {raster["counts"].sum():,.0f}')
# Expect roughly 3.3 million (county population) -- if it's wildly off, check bbox/nodata


# ---------------------------------------------------------------------------
# 2. Sample agents and project to km
# ---------------------------------------------------------------------------
# Pick N cells with probability proportional to the people in each cell, then place
# each agent at a random spot inside its cell.
# lonlat has shape (N, 2): column 0 = longitude, column 1 = latitude (in degrees).
lonlat = sample_coords_from_density(raster, N)

# Convert degrees to km so that straight-line distance between two rows is a real distance in km.
# xy has shape (N, 2): column 0 = east-west km, column 1 = north-south km.
xy = lonlat_to_km(lonlat)


# ---------------------------------------------------------------------------
# 3. Build network
# ---------------------------------------------------------------------------
t = time.time()                               # start the timer
edges = make_rank_based_contacts(xy, DEG)     # the rank-based friendship rule (the main function being tested)
p1, p2 = edges['p1'], edges['p2']             # edgelist: person p1[i] is friends with person p2[i]

# Prints: number of friendships (each pair counted once), seconds taken, and
# average friends per person = 2 * edges / people (each edge has two ends).
print(f'Built {len(p1):,} edges in {time.time()-t:.1f}s, mean degree {2*len(p1)/N:.1f}')


# ---------------------------------------------------------------------------
# 4. Friend distances: should be short in the dense coast, longer in the sparse east
# ---------------------------------------------------------------------------
# Length of every friendship in km: straight-line distance between the two friends' locations.
dist = np.linalg.norm(xy[p1] - xy[p2], axis=1)

# Where each friendship sits on the map: the average longitude of its two ends.
lon_mid = (lonlat[p1, 0] + lonlat[p2, 0]) / 2

# Split the friendships into the western (dense, urban) and eastern (sparse, backcountry)
# parts of the county. These longitude cut-offs are rough, chosen from geography:
# downtown/coast is near -117.2, mountain and desert towns are east of about -116.7.
summary_lines = []   # the same text is also drawn on the map in step 5
for name, mask in [('coast (lon < -117.1)', lon_mid < -117.1),
                   ('east  (lon > -116.7)', lon_mid > -116.7)]:
    if mask.sum():   # skip a region if it has no friendships
        # Prints: how many friendships fall in the region (n), and the median
        # friend distance in km (half of friendships are shorter, half longer).
        # The east number is noisy because few agents live there.
        line = f'{name}: n={mask.sum():5d}, median friend distance {np.median(dist[mask]):6.2f} km'
        print(line)
        summary_lines.append(line)


# ---------------------------------------------------------------------------
# 5. Plot
# ---------------------------------------------------------------------------
fig, ax = plt.subplots(1, 2, figsize=(12, 5))   # two panels side by side

# Left: one dot per agent (x = longitude, y = latitude). Should look like San Diego
# County: a dense coastal band and inland corridors, with sparse east county.
ax[0].scatter(*lonlat.T, s=1, alpha=0.4)
ax[0].set_xlabel('longitude (degrees)'); ax[0].set_ylabel('latitude (degrees)')
ax[0].set_title('Sampled agents')

# Right: histogram of friendship lengths on normal axes (0-50 km in 0.5 km bins).
# Most friendships should be short, with a long tail of far-apart friends.
# Friendships longer than 50 km fall off the right edge; the print below says how many.
ax[1].hist(dist, bins=np.linspace(0, 50, 100))
ax[1].set_xlabel('friend distance (km)'); ax[1].set_ylabel('number of friendships') 
ax[1].set_title(f'Edge Length. Total num of edges (friendships) = {len(p1):,}')
print(f'{(dist > 50).mean():.1%} of friendships are longer than 50 km (not shown in histogram)')

# Write the coast/east summary on the map, top right.
# transAxes = position as a fraction of the panel (0,0 = bottom left, 1,1 = top right), not data units.
# monospace font keeps the two lines lined up.
ax[0].text(0.97, 0.97, '\n'.join(summary_lines), transform=ax[0].transAxes,
           ha='right', va='top', fontsize=7, family='monospace',
           bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

plt.tight_layout(); plt.savefig('../data/sandiego_test_10_08_26_100k.png', dpi=150)   # save to the (gitignored) data folder
print('Saved ../data/sandiego_test_10_08_26_100k.png')