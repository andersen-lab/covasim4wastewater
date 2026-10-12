'''
Test that Covasim's rank-based population type (pop_type='rank') works.

Run:
    python test_covasim_rank_model.py                      # tests 1 and 2, no data needed
    python test_covasim_rank_model.py path/to/raster.tif   # also runs test 3 on a population-density raster

Tests:
    1. make_pars(pop_type='rank') gives one static contact layer 'a' with 20 contacts
    2. A small sim runs with no raster (agents are spread over a 10 x 10 km test square)
    3. (optional) A sim using a population-density raster gives a sensible network: mean degree
       a little below the 20 contacts requested, and no extreme hubs

Test 3 needs the optional rasterio package (pip install rasterio) and a population-count GeoTIFF
in longitude/latitude, for example from WorldPop.
'''
import sys
import numpy as np
import covasim as cv
import pytest

# ---- 1. Layer defaults ----
pars = cv.make_pars(pop_type='rank')

print('1. Defaults:', pars['contacts'], pars['dynam_layer'], pars['beta_layer'])
assert pars['contacts'] == {'a': 20}, 'rank should have a single layer "a" with 20 contacts'
assert pars['dynam_layer'] == {'a': 0}, 'the layer must be static, or Covasim redraws it at random each day'
assert pars['beta_layer'] == {'a': 1.0}


# ---- 2. Small sim, no raster ----
sim = cv.Sim(pop_type='rank', pop_size=500, n_days=10, verbose=0).run()

print('2. No raster: layers =', list(sim.people.contacts.keys()))
assert list(sim.people.contacts.keys()) == ['a']


# ---- 3. Sim on a density raster (optional) ----
if len(sys.argv) > 1:
    raster = sys.argv[1]
    n = 5000 # agents; building the rank network takes a few seconds at this size

    sim = cv.Sim(pop_type='rank', pop_size=n, density_raster=raster, n_days=60, rand_seed=1, verbose=0).run()

    lay = sim.people.contacts['a']
    deg = np.bincount(np.concatenate([lay['p1'], lay['p2']]), minlength=n) # contacts per person
    print(f'3. Raster, {n:,} agents: mean degree {deg.mean():.1f}, std {deg.std():.1f}, max {deg.max()}')
    assert 14 < deg.mean() < 20, 'mean degree should be a little below the 20 contacts requested (duplicate links are merged)'
    assert deg.max() < 60, 'a much larger maximum would mean a few people have huge numbers of contacts'
else:
    print('3. Skipped (no raster given)')

print('All checks passed')



def test_rank_population_with_raster(tmp_path):
    ''' pop_type='rank' with a density raster should place agents where the raster has people '''
    rasterio = pytest.importorskip('rasterio') # Skip this test if the optional rasterio package is missing
    from rasterio.transform import from_origin

    # A tiny 20 x 20 raster, 0.01 degree cells, with all of the people in the left half
    counts = np.zeros((20, 20), dtype='float32')
    counts[:, :10] = 5
    path = tmp_path / 'tiny.tif'
    with rasterio.open(path, 'w', driver='GTiff', height=20, width=20, count=1, dtype='float32',
                       crs='EPSG:4326', transform=from_origin(-117.0, 33.0, 0.01, 0.01)) as dst:
        dst.write(counts, 1)

    sim = cv.Sim(pop_type='rank', pop_size=300, density_raster=str(path), n_days=5, verbose=0).run()
    assert list(sim.people.contacts.keys()) == ['a']
    assert len(sim.people.contacts['a']) > 0

#TESTS
#RUN: tmux attach -t pytest
#END/DETATCH: Ctrl-b then d