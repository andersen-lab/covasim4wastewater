'''
CHECK: do the density raster, the boundary file and the rank population agree?

Builds a small rank-based population (5,000 agents) from the San Diego density raster and a boundary file, then checks
that every agent's REGION matches the polygon its LOCATION is in. A check for you, not part of the Covasim test suite.

=======================================================================================
THE THREE INPUTS
=======================================================================================
  Raster    A GeoTIFF grid where each ~100 m cell holds the estimated number of people living in it. It says WHERE
            people live (san_diego_county_ppp_2020.tif).
  Boundary  A GeoJSON of region polygons (example_boundaries.geojson, 2 regions) with region_code, region_name,
            population and a coordinate system (CRS). It says which REGION a place is in.
  Decision  The raster is the ground truth for where people live. The boundary file's population column is NOT used
            for the rank type (it is not accurate); the file only names the regions.

=======================================================================================
HOW THE GEOGRAPHY WORKS (geography.py: assign_density_coords_in_boundary)
=======================================================================================
  Called by make_randpop() when pop_type='rank' and a raster is given.

  1. Read the raster: people per cell, plus each cell centre's longitude and latitude (load_density_raster).
     "No data" cells become 0.
  2. Put the boundary in lon/lat (EPSG:4326), like the raster. Done once.
  3. Mask the raster (mask_raster_to_boundary): set to 0 the people in every cell whose centre is outside all the
     polygons, so agents can only be placed inside the boundary.
  4. Place the agents (sample_coords_from_density): pick a cell for each agent with probability proportional to its
     people, then a random spot inside it. Dense cells get many agents, empty cells none.
  5. Find each agent's region (assign_regions_from_points): take the region_code of the polygon containing its point.
     A point on a shared edge can match twice, so one match is kept. The region comes from the polygon holding the
     point, so region and location agree by design.
  6. Redraw strays: a random spot can land just outside the boundary even when its cell centre is inside. Those
     agents are redrawn (steps 4 and 5), up to 20 times, until none is left.
  7. Cast the region codes back to their original type.
  8. Return:
       xy_km    x, y in KILOMETRES (lonlat_to_km), so straight-line distance is real distance. The rank network
                ranks people by distance, so it needs this.
       coords   x, y in the boundary file's own coordinate system (convert_points), stored as pars['people_coords']
                for the sewershed code.
       regions  each agent's region_code, stored as popdict['region'].

  In make_randpop(), for the rank type, the rank network is built from xy_km (make_rank_based_contacts). With NO
  raster, agents are spread uniformly inside the polygons instead, with regions chosen by the file's population
  column (a testing fallback). Then make_people() builds the people and assign_sewersheds() runs.

=======================================================================================
HOW THIS EXAMPLE WORKS
=======================================================================================
  A. Set the paths from this script's location (so it runs from any folder): the raster is in
     <repo>/network_rank_model/data, the boundary in <repo>/tests. Change to <repo>/tests, where the default
     sewershed file is looked up.
  B. Build a Sim (pop_type='rank', 5,000 agents, density_raster, boundary_shapefile) and call sim.initialize(),
     which runs the whole chain above.
  C. Read the stored regions (sim.people.region) and locations (sim.pars['people_coords']). Print agents per region
     and the range of locations.
  D. Independent check: read the boundary again, make a point from each stored location and find the polygon
     containing it. Print how many agents are outside every polygon and how many stored regions match the polygon
     found (compared by value, so 1, 1.0 and "1" match). Mismatches are listed.
  E. Second check without the Sim: call assign_density_coords_in_boundary directly and look up the polygon for each
     location it returns. This separates "the function is wrong" from "something later changes the region".
  F. Run the 5-day epidemic and print the layer names and the number of friendships.

=======================================================================================
WHAT THE PRINTED LINES MEAN
=======================================================================================
  agents: 5000 | region codes: {...}    agents per region (decided by the raster).
  locations (lon, lat) from ... to ...  the box the agents occupy; it should sit inside the boundary.
  agents outside every polygon: 0       should be 0 (strays were redrawn).
  ... matches the polygon ...: N of 5000   should be 5000 of 5000, and the same for the function on its own.
  contact layers: ['a'] | friendships   one static layer 'a'; friendships x 2 / agents is the average number of
                                        friends (about 17 for 20 target contacts).

  First run: 5,000 agents; region codes {1: 1,467, 2: 3,533}; locations (-117.25, 32.7) to (-117.051, 32.899),
  inside the boundary box; 0 agents outside; layer ['a'] with 43,539 friendships (17.4 each).
'''
import os
from pathlib import Path
import numpy as np
import geopandas as gpd
import covasim as cv

CURRENT_FOLDER = Path(__file__).resolve().parent            # the folder this script is in (network_rank_model/tests)
REPO = CURRENT_FOLDER.parents[1]                            # the repo root
TIF      = str(REPO/'network_rank_model'/'data'/'san_diego_county_ppp_2020.tif')
BOUNDARY = str(REPO/'tests'/'example_boundaries.geojson')
os.chdir(REPO/'tests')                            # the default sewershed file is looked up from the repo's tests folder

sim = cv.Sim(pop_type='rank', pop_size=5000, density_raster=TIF, boundary_shapefile=BOUNDARY, n_days=5, verbose=0)
sim.initialize()
regions = np.array(sim.people.region)
coords  = sim.pars['people_coords']                      # lon/lat, in the boundary file's coordinate system
print('agents:', len(regions), '| region codes:', dict(zip(*np.unique(regions, return_counts=True))))
print('locations (lon, lat) from', coords.min(axis=0).round(3), 'to', coords.max(axis=0).round(3))

boundary = gpd.read_file(BOUNDARY)
points = gpd.GeoDataFrame(geometry=gpd.points_from_xy(coords[:, 0], coords[:, 1]), crs=boundary.crs)
joined = gpd.sjoin(points, boundary[['region_code', 'geometry']], how='left', predicate='within')
joined = joined[~joined.index.duplicated()].reindex(points.index)
found = joined['region_code'].to_numpy()
print('agents outside every polygon:', int(np.sum(joined['region_code'].isna())))


def same_code(a, b):
    '''Compare two region codes by value, so that 1, 1.0 and "1" count as the same code'''
    try:
        return float(a) == float(b)
    except (TypeError, ValueError):
        return str(a) == str(b)


ok = np.array([same_code(a, b) for a, b in zip(found, regions)])
print(f'agents whose stored region matches the polygon their location is in: {ok.sum()} of {len(ok)}')
print('types: stored region', regions.dtype, '| polygon lookup', found.dtype)
if not ok.all():
    print('first mismatches (agent, stored region, polygon found, lon/lat):')
    for i in np.flatnonzero(~ok)[:5]:
        print('  ', i, regions[i], found[i], coords[i])

# Second check, without the Sim: does the new function itself give consistent regions and locations?
from covasim.population import load_population_regions
from covasim.geography import assign_density_coords_in_boundary, assign_regions_from_points
b2 = load_population_regions(BOUNDARY)
xy_km2, coords2, regions2 = assign_density_coords_in_boundary({'pop_size': 5000, 'density_raster': TIF, 'density_bbox': None}, b2)
found2 = assign_regions_from_points(coords2, b2.to_crs('EPSG:4326'))
ok2 = np.array([same_code(a, b) for a, b in zip(found2, regions2)])
print(f'function on its own: regions consistent with locations for {ok2.sum()} of {len(ok2)} agents')

sim.run()
print('contact layers:', list(sim.people.contacts.keys()), '| friendships:', len(sim.people.contacts['a']))