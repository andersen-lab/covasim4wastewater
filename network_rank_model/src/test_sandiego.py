'''
Smoke test: rank-based friendship network on WorldPop density for San Diego County.
Run from network_rank_model/src:   python test_sandiego.py
'''
import time
import numpy as np
import matplotlib.pyplot as plt
from rank_network import load_density_raster, sample_coords_from_density, lonlat_to_km, make_rank_based_contacts

TIF  = '../data/san_diego_county_ppp_2020.tif'
BBOX = None   # file is already clipped to the county
#TIF  = '../data/usa_ppp_2020.tif'
#BBOX = (-117.6, 32.53, -116.08, 33.51)  # lon_min, lat_min, lon_max, lat_max
N    = 20000 #5000   # start small; runtime scales as N^2
DEG  = 10     # target mean number of friends

np.random.seed(0)

# 1. Raster sanity check
raster = load_density_raster(TIF, BBOX)
print(f'Raster shape {raster["counts"].shape}, total pop in bbox: {raster["counts"].sum():,.0f}')
# Expect roughly 3.3 million (county population) -- if it's wildly off, check bbox/nodata

# 2. Sample agents and project to km
lonlat = sample_coords_from_density(raster, N)
xy = lonlat_to_km(lonlat)

# 3. Build network
t = time.time()
edges = make_rank_based_contacts(xy, DEG)
p1, p2 = edges['p1'], edges['p2']
print(f'Built {len(p1):,} edges in {time.time()-t:.1f}s, mean degree {2*len(p1)/N:.1f}')

# 4. Friend distances: should be short in the dense coast, longer in the sparse east
dist = np.linalg.norm(xy[p1] - xy[p2], axis=1)
lon_mid = (lonlat[p1, 0] + lonlat[p2, 0]) / 2
for name, mask in [('coast (lon < -117.1)', lon_mid < -117.1),
                   ('east  (lon > -116.7)', lon_mid > -116.7)]:
    if mask.sum():
        print(f'{name}: n={mask.sum():5d}, median friend distance {np.median(dist[mask]):6.2f} km')

# 5. Plot
fig, ax = plt.subplots(1, 2, figsize=(12, 5))
ax[0].scatter(*lonlat.T, s=1, alpha=0.4)
ax[0].set_title('Sampled agents')
ax[1].hist(dist, bins=np.logspace(-2, 2, 50))
ax[1].set_xscale('log'); ax[1].set_xlabel('friend distance (km)'); ax[1].set_title('Edge lengths')
plt.tight_layout(); plt.savefig('../data/sandiego_test.png', dpi=150)
print('Saved ../data/sandiego_test.png')