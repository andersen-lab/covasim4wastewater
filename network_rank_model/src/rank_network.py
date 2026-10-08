'''
Rank-based geographic friendship network for Covasim.

Model: Liben-Nowell et al. (2005), "Geographic routing in social networks".
Person u befriends person v with probability proportional to 1/rank_u(v)^alpha
(alpha = 1 in the paper), where rank_u(v) = how many people are at least as
close to u as v is. Specifically person u links to person v with probability proportional to

    P(u -> v)  ~  1 / rank_u(v)^alpha,    alpha = 1 in the paper

where rank_u(v) = number of people w (w != u) that are at least as close to u
as v is. 

Rank (not distance) means links are short in dense areas
and long in sparse areas. 

=============================== GLOSSARY ====================================
    .tif / GeoTIFF : an image file where each pixel stores a number instead of a
                     colour, plus its map location. Here: people per pixel.
    raster         : any such grid of numbers laid over a map. Ours (WorldPop)
                     covers San Diego County in ~100 m x 100 m cells.
    density_raster : the parameter holding the file path to that .tif.
    bbox           : "bounding box", a rectangle (lon_min, lat_min, lon_max, lat_max)
                     used to read only part of a raster. None = read the whole file.
    density_bbox   : the parameter holding that rectangle.
    lon / lat      : longitude (east-west) / latitude (north-south), in degrees.
    nodata         : the value a raster uses for "no data here" (WorldPop: -99999);
                     we replace it with 0 people.
    rank           : position in a person's list of everyone else sorted by distance
                     (1 = nearest person, 2 = next nearest, ...).
    alpha          : exponent in P(friend at rank r) ~ 1/r^alpha. It sets how
                     strongly friendships favour nearby people:
                       alpha = 0  -> distance ignored, friends chosen at random
                       alpha = 1  -> the paper's value; they measured real friendships
                                     (LiveJournal) and found probability fell off
                                     about 1/rank
                       alpha > 1  -> even more local
                     It is a fitted value from data, not a derived constant;
                     you can change it to test sensitivity.
    n              : the target AVERAGE number of friends per person (also called
                     the "degree" of a person in network terms). n = 10 means a
                     typical person has about 10 friends; some will have 4,
                     some 16, because each person's count is drawn at random
                     (Poisson). The final average comes out a little below n
                     (about 9.2-9.5 for n = 10) because duplicate links are merged.
    edgelist       : how the network is stored: a list of friendships, one per
                     position, held in two equal-length arrays p1 and p2.
                     Friendship number i is between person p1[i] and person p2[i]
                     (people are numbered 0 .. N-1). Example:
                         p1 = [0, 0, 3]
                         p2 = [5, 2, 4]
                     means 0-5, 0-2 and 3-4 are friends. Friendships are mutual
                     (undirected), so each pair is listed once, not twice.
                     This is the format Covasim uses for a contact layer, which
                     is why make_random_contacts and this file can be swapped.

=============================== PIPELINE ====================================

STEP 1  load_density_raster(path, bbox=None)
    IN : path to a WorldPop .tif (a grid; each cell = number of people living there),
         optional bbox (lon_min, lat_min, lon_max, lat_max) to read only a window
    OUT: dict with
           counts : 2D array, people per cell (nodata/negatives set to 0)
           lons, lats : 1D arrays, centre coordinate of each column / row
           dlon, dlat : cell size in degrees

STEP 2  sample_coords_from_density(raster, pop_size)
    IN : the dict from step 1, number of agents
    OUT: (pop_size, 2) array of (lon, lat), one row per agent
    HOW: pick a cell with probability proportional to its people count, then
         place the agent at a random spot inside that cell.
         => dense cells get many agents, empty cells get none.

STEP 3  lonlat_to_km(lonlat)
    IN : (N, 2) array of (lon, lat) in degrees
    OUT: (N, 2) array of (x, y) in km, so Euclidean distance = real distance

STEP 4  assign_coords(pars)
    Wrapper for steps 1-3.
    IN : pars with 'pop_size', and optionally 'density_raster', 'density_bbox'
    OUT: (N, 2) array of (x, y) in km
         (no raster given -> uniform random points in a 10x10 km box, for tests)

STEP 5  make_rank_based_contacts(coords, n, alpha=1.0, mapping=None)
    IN : coords  (N, 2) km array from step 4
         n       average friends per person
         alpha   rank exponent (1.0 = paper)
         mapping optional; remaps indices 0..N-1 onto other ids
    OUT: dict(p1=array, p2=array), an edgelist: person p1[i] is friends with p2[i]
    HOW: for each person u
           a) draw k ~ Poisson(n/2) friendships
           b) for each, draw a rank r in 1..N-1 with P(r) ~ 1/r^alpha
           c) sort everyone else by distance from u; link u to the r-th nearest
         then merge duplicate / mutual pairs (u-v and v-u are one edge).
         Mean degree ends up slightly below n (about 0.9n-0.95n).

=========================== VS make_random_contacts ==========================
    Same OUTPUT format: dict(p1, p2) of int arrays (cvd.default_int), undirected.
    Different INPUT: needs coords (locations); random version only needs pop_size.
    Different RULE : random picks partners uniformly; this picks by geographic rank.
    Cost: random is ~O(N); this is O(N^2) (about 20 s at 20k agents).

=========================== NOT YET WIRED INTO COVASIM =======================
    Next: call assign_coords() and make_rank_based_contacts() from make_randpop()
    in population.py, store result as contacts['f'], add 'f' to layer parameters.

Note: agent distances depend on the number of agents simulated (rank is relative
to the sample), so friend distances in km shrink as pop_size grows.
'''

import numpy as np

import covasim.defaults as cvd
import covasim.utils as cvu


# ----------------------------------------------------------------------------
# 1. Coordinates: sample agent locations from a population-density raster
# ----------------------------------------------------------------------------

def load_density_raster(path, bbox=None):
    '''
    Read a WorldPop-style population-count GeoTIFF (people per pixel).

    Args:
        path (str): path to the .tif
        bbox (tuple): (lon_min, lat_min, lon_max, lat_max) to crop to, e.g. San Diego County

    Returns:
        dict with 'counts' (2D array), 'lons' (1D pixel-centre lons), 'lats' (1D pixel-centre lats),
        'dlon', 'dlat' (pixel size in degrees)
    '''
    import rasterio  # Optional import: pip install rasterio
    from rasterio.windows import from_bounds

    with rasterio.open(path) as src:
        if bbox is not None:
            window = from_bounds(*bbox, transform=src.transform)
            window = window.round_offsets().round_lengths()
        else:
            window = rasterio.windows.Window(0, 0, src.width, src.height)
        counts = src.read(1, window=window).astype(float)
        transform = src.window_transform(window)
        nodata = src.nodata

    if nodata is not None:
        counts[counts == nodata] = 0
    counts[~np.isfinite(counts)] = 0
    counts[counts < 0] = 0

    nrows, ncols = counts.shape
    dlon, dlat = transform.a, transform.e  # dlat is negative (north-up rasters)
    lons = transform.c + dlon*(np.arange(ncols) + 0.5)
    lats = transform.f + dlat*(np.arange(nrows) + 0.5)
    return dict(counts=counts, lons=lons, lats=lats, dlon=dlon, dlat=dlat)


def sample_coords_from_density(raster, pop_size):
    '''
    Draw pop_size (lon, lat) points: pick a pixel with probability proportional
    to its population count, then place the agent uniformly within the pixel.
    '''
    counts = raster['counts']
    probs = counts.ravel()/counts.sum()
    pix = np.random.choice(probs.size, size=pop_size, p=probs)
    rows, cols = np.unravel_index(pix, counts.shape)
    lon = raster['lons'][cols] + (np.random.random(pop_size) - 0.5)*abs(raster['dlon'])
    lat = raster['lats'][rows] + (np.random.random(pop_size) - 0.5)*abs(raster['dlat'])
    return np.column_stack((lon, lat))


def lonlat_to_km(lonlat):
    '''
    Equirectangular projection to km about the mean latitude. Fine at city/county
    scale, so Euclidean distance in this space is a good distance in km.
    '''
    lon0, lat0 = lonlat[:, 0].mean(), lonlat[:, 1].mean()
    kx = 111.320*np.cos(np.radians(lat0))
    ky = 110.574
    return np.column_stack(((lonlat[:, 0]-lon0)*kx, (lonlat[:, 1]-lat0)*ky))


def assign_coords(pars, regions=None):
    '''
    Replacement for the placeholder assign_coords(). Returns (N,2) array in km.

    Expects (add to cv.make_pars defaults):
        pars['density_raster'] : path to WorldPop .tif
        pars['density_bbox']   : optional (lon_min, lat_min, lon_max, lat_max)
    Falls back to uniform-in-a-box if no raster is given (useful for unit tests).
    '''
    pop_size = int(pars['pop_size'])
    path = pars.get('density_raster', None)
    if path is None:
        return np.random.uniform(0, 10, size=(pop_size, 2))  # 10x10 km toy box
    raster = load_density_raster(path, pars.get('density_bbox', None))
    return lonlat_to_km(sample_coords_from_density(raster, pop_size))


# ----------------------------------------------------------------------------
# 2. The rank-based network
# ----------------------------------------------------------------------------

def make_rank_based_contacts(coords, n, alpha=1.0, dispersion=None, mapping=None):
    '''
    Make a friendship layer as an edgelist, where who you befriend depends on
    how near they are relative to everyone else (rank-based attachment).

    Args:
        coords     (array) : (N,2) array of x,y locations in km, one row per person
        n          (float) : the average number of friends per person for this layer
        alpha      (float) : how strongly friendships favour nearby people; 1.0 is the
                             value from Liben-Nowell et al. (0 = ignore distance)
        dispersion (float) : if not None, use a negative binomial distribution with this
                             dispersion parameter instead of Poisson for friend counts
        mapping    (array) : optionally map the generated indices onto new indices

    Returns:
        Dictionary of two arrays defining UIDs of the edgelist (sources and targets),
        the same format as make_random_contacts()

    How it works, for each person p:
        1. Decide how many friendships p starts (Poisson, averaging n/2).
        2. For each one, pick a rank r. Small ranks are much more likely: P(r) ~ 1/r^alpha.
        3. Sort everyone else by distance from p and befriend whoever is r-th nearest.
    Duplicate pairs (A-B and B-A, or A-B twice) are then merged into one friendship.

    Speed: about O(N^2), roughly 20 s for 20,000 people. Fine for one-off population building.
    '''

    # Preprocessing
    coords   = np.asarray(coords, dtype=float)
    pop_size = len(coords) # Number of people
    p1 = [] # Initialize the "sources"
    p2 = [] # Initialize the "targets"
    if pop_size < 2: # Nobody to befriend
        return dict(p1=np.array([], dtype=cvd.default_int), p2=np.array([], dtype=cvd.default_int))

    # Precalculate the probability of each rank
    # Rank 1 = nearest person, rank pop_size-1 = farthest. Weight of rank r is 1/r^alpha.
    rank_weights = np.arange(1, pop_size, dtype=float)**(-alpha)
    rank_cdf     = np.cumsum(rank_weights) # Running total, used to turn random numbers into ranks
    rank_cdf    /= rank_cdf[-1]

    # Precalculate how many friendships each person starts (same approach as make_random_contacts)
    if dispersion is None:
        p_count = cvu.n_poisson(n, pop_size) # Draw the number of Poisson contacts for this person
    else:
        p_count = cvu.n_neg_binomial(rate=n, dispersion=dispersion, n=pop_size) # Or, from a negative binomial
    p_count = np.array((p_count/2.0).round(), dtype=cvd.default_int) # Halve: each friendship has two ends

    # Make contacts
    for p in range(pop_size):
        n_friends = p_count[p]
        if n_friends == 0:
            continue

        # Pick a rank for each friendship (1 = nearest person); rank_cdf maps random numbers to ranks
        ranks = np.searchsorted(rank_cdf, np.random.random(n_friends)) + 1
        ranks = np.minimum(ranks, pop_size-1) # Guard against rounding at the far end

        # Squared distance from this person to everyone (squared is fine for ordering)
        dist2 = ((coords - coords[p])**2).sum(axis=1)
        dist2[p] = np.inf # Put self last so it can never be chosen

        # Find who sits at each chosen rank; argpartition avoids sorting everyone
        order   = np.argpartition(dist2, np.unique(ranks-1))
        friends = order[ranks-1]

        p1.append(np.full(n_friends, p))
        p2.append(friends)

    # Tidy up: join the pieces, then merge duplicate pairs (A-B and B-A are the same friendship)
    p1 = np.concatenate(p1) if p1 else np.empty(0)
    p2 = np.concatenate(p2) if p2 else np.empty(0)
    lo, hi = np.minimum(p1, p2), np.maximum(p1, p2)
    pairs  = np.unique(np.column_stack((lo, hi)), axis=0)
    p1, p2 = pairs[:, 0], pairs[:, 1]

    if mapping is not None:
        mapping = np.asarray(mapping)
        p1, p2  = mapping[p1], mapping[p2]

    output = dict(p1=p1.astype(cvd.default_int), p2=p2.astype(cvd.default_int))
    return output