'''
Rank-based geographic friendship network for Covasim, SCALED version.

Model: Liben-Nowell et al. (2005), "Geographic routing in social networks".
Person u links to person v with probability proportional to

    P(u -> v)  ~  1 / rank_u(v)^alpha,    alpha = 1 in the paper

where rank_u(v) = number of people w (w != u) that are at least as close to u
as v is.

Rank (not distance) means links are short in dense areas
and long in sparse areas.

WHAT IS NEW IN THIS VERSION (vs rank_network.py)
    1. FAST. rank_network.py compares every person with every other person (cost ~ N^2: 20,000 agents
       = 20 s, 100,000 = 404 s, 3.4 million = about a week). Here the r-th nearest person is found
       with a KD-tree (scipy cKDTree), so the cost is roughly linear in N:
           N = 200,000 -> 16 s      N = 800,000 -> 72 s      (1 CPU core; more cores help)
       This makes simulating hundreds of thousands to millions of agents practical.
    2. n_real option. Ranks can be drawn from the REAL population (e.g. 3.4 million) and converted to
       the sample: r_sample = r_real * N / n_real. Leave n_real=None for the original behaviour.

    HOW THE KD-TREE FINDS THE r-th NEAREST AGENT
       A KD-tree returns the K nearest agents quickly (K = 128), which covers ranks 1..128 exactly.
       For larger ranks, the same trick is applied to a THINNED copy of the agents (every 16th agent,
       then every 256th, ...). The r-th nearest in a copy thinned by m is about the (r*m)-th nearest
       overall, so every rank is reachable with K neighbours, with a relative error of at most a
       few percent (about 6% at the worst point).

    WHAT TESTING SHOWED (synthetic clustered data, degree 20)
       - Fast vs brute force (N = 4,000): median friend distance 2.52 vs 2.59 km, 90th percentile
         16.8 vs 16.9 km, so the KD-tree version reproduces the original.
       - Friend distances still depend on N. In a population of P = 400,000 simulated entirely
         (the ground truth), the median friend distance is 0.75 km. Simulating only a sample of it
         (with n_real = P) gave:
               agents N      12,500   25,000   50,000   100,000   200,000   (truth at 400,000: 0.75)
               median km       1.79     1.50     1.26      1.04      0.87
         The error shrinks as N approaches the real population. n_real changed the result by only
         ~3%: a friend closer than the nearest sampled agent cannot be represented, so the benefit
         comes from using MORE AGENTS, not from rescaling ranks.
       => For realistic San Diego distances, simulate as many agents as Covasim can handle
          (aim for 500,000 to 3.4 million), then use pop_scale for any remainder.

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
    n_real         : the REAL number of people the agents stand for (e.g. 3,400,000 for
                     San Diego County). Ranks are drawn from this larger population.
    agent          : one simulated person. With N agents standing in for n_real people,
                     each agent represents n_real / N real people (Covasim's pop_scale).
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

STEP 5  make_rank_based_contacts(coords, n, alpha=1.0, n_real=None, dispersion=None, mapping=None,
                                 distinct=None, K=128, thin=16, chunk=50000, workers=-1)
    IN : coords  (N, 2) km array from step 4
         n       average friends per person
         alpha   rank exponent (1.0 = paper)
         n_real  real population the N agents represent (None = treat the N agents as everyone)
         dispersion / mapping : as in make_random_contacts
         distinct  redraw repeated friends so each person's friends are different agents
                   (default: on when n_real is given, off otherwise)
         K, thin   KD-tree settings: neighbours per query, and thinning factor between levels
         chunk     agents per KD-tree query batch (limits memory)
         workers   CPU cores for KD-tree queries (-1 = all)
    OUT: dict(p1=array, p2=array), an edgelist: person p1[i] is friends with p2[i]
    HOW: a) draw how many friendships each person starts (Poisson, averaging n/2)
         b) draw a rank for every friendship, P(r) ~ 1/r^alpha over 1..n_real-1, and convert to the
            sample: r_sample = r_real * N / n_real (so r_sample = r_real when n_real=None)
         c) look up the r_sample-th nearest agent with the KD-tree levels described above
         d) if distinct, redraw repeats (up to 5 rounds)
         then merge duplicate / mutual pairs (u-v and v-u are one edge).

=========================== VS make_random_contacts ==========================
    Same OUTPUT format: dict(p1, p2) of int arrays (cvd.default_int), undirected.
    Different INPUT: needs coords (locations); random version only needs pop_size.
    Different RULE : random picks partners uniformly; this picks by geographic rank.
    Cost: random is ~O(N) (3.4 million agents = 10 s); this is roughly linear too (800k agents = 72 s).

=========================== NOT YET WIRED INTO COVASIM =======================
    Next: call assign_coords() and make_rank_based_contacts() from make_randpop()
    in population.py, store result as contacts['f'], add 'f' to layer parameters.
'''

import numpy as np
from scipy.spatial import cKDTree # Fast nearest-neighbour search (scipy is a Covasim dependency)

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

def make_rank_based_contacts(coords, n, alpha=1.0, n_real=None, dispersion=None, mapping=None,
                             distinct=None, K=128, thin=16, chunk=50000, workers=-1, max_redraws=5):
    '''
    Make a friendship layer as an edgelist, where who you befriend depends on
    how near they are relative to everyone else (rank-based attachment).

    Args:
        coords     (array) : (N,2) array of x,y locations in km, one row per agent
        n          (float) : the average number of friends per person for this layer
        alpha      (float) : how strongly friendships favour nearby people; 1.0 is the
                             value from Liben-Nowell et al. (0 = ignore distance)
        n_real     (int)   : the real population the N agents stand for (e.g. 3.4 million).
                             Ranks are drawn from this population and converted to the sample.
                             None = the N agents are the whole population (original behaviour)
        dispersion (float) : if not None, use a negative binomial distribution with this
                             dispersion parameter instead of Poisson for friend counts
        mapping    (array) : optionally map the generated indices onto new indices
        distinct   (bool)  : redraw repeated friends so each person's friends are different agents.
                             None = on when n_real is given, off otherwise
        K          (int)   : neighbours returned per KD-tree query (ranks up to K are exact)
        thin       (int)   : thinning factor between KD-tree levels (for ranks above K)
        chunk      (int)   : agents per KD-tree query batch; lower this if memory is tight
        workers    (int)   : CPU cores for KD-tree queries (-1 = all)
        max_redraws (int)  : rounds of redrawing repeated friends

    Returns:
        Dictionary of two arrays defining UIDs of the edgelist (sources and targets),
        the same format as make_random_contacts()

    How it works, for each person p:
        1. Decide how many friendships p starts (Poisson, averaging n/2).
        2. For each one, pick a rank r. Small ranks are much more likely: P(r) ~ 1/r^alpha.
           (With n_real, r is a rank among the real population, then converted to the sample.)
        3. Find the r-th nearest agent with a KD-tree (see the file header for how large ranks are handled).
    Duplicate pairs (A-B and B-A, or A-B twice) are then merged into one friendship.

    Speed: roughly linear in N; about 16 s for 200,000 agents and 72 s for 800,000 on one core.
    '''

    # Preprocessing
    coords = np.asarray(coords, dtype=float)
    N = len(coords) # Number of agents
    if N < 2: # Nobody to befriend
        return dict(p1=np.array([], dtype=cvd.default_int), p2=np.array([], dtype=cvd.default_int))

    # Scaling: ranks are drawn from the real population, then shrunk to the sample
    scaled = n_real is not None
    n_real = N if not scaled else int(n_real)
    if n_real < N:
        raise ValueError(f'n_real ({n_real}) cannot be smaller than the number of agents ({N})')
    scale = N/n_real # How many agents per real person (1.0 when not scaling)
    if distinct is None:
        distinct = scaled

    # Precalculate the probability of each real rank (rank 1 = nearest person, weight 1/r^alpha)
    rank_cdf  = np.cumsum(np.arange(1, n_real, dtype=float)**(-alpha)) # Running total, turns random numbers into ranks
    rank_cdf /= rank_cdf[-1]

    def draw_sample_ranks(k):
        '''Draw k ranks among the real population, as (fractional) ranks among the agents.'''
        return (np.searchsorted(rank_cdf, np.random.random(k)) + 1)*scale

    # Precalculate how many friendships each person starts (same approach as make_random_contacts)
    if dispersion is None:
        p_count = cvu.n_poisson(n, N) # Draw the number of Poisson contacts for this person
    else:
        p_count = cvu.n_neg_binomial(rate=n, dispersion=dispersion, n=N) # Or, from a negative binomial
    p_count = np.array((p_count/2.0).round(), dtype=np.int64) # Halve: each friendship has two ends
    src = np.repeat(np.arange(N, dtype=np.int64), p_count) # One entry per friendship: who starts it
    if len(src) == 0:
        return dict(p1=np.array([], dtype=cvd.default_int), p2=np.array([], dtype=cvd.default_int))

    # Build the KD-tree levels. Level 0 holds every agent. Each further level holds a random 1/thin of the
    # previous one, until the coarsest level is small enough to reach the farthest ranks with K neighbours.
    K_eff = min(K, N-1)
    perm = np.random.permutation(N) # Random order, so each level is a random subset of the agents
    subsets, trees = [np.arange(N)], [cKDTree(coords)]
    size = N
    while size > K_eff+1:
        size = max(int(np.ceil(size/thin)), 2)
        subset = perm[:size]
        subsets.append(subset)
        trees.append(cKDTree(coords[subset]))
    n_levels = len(subsets)
    sizes = [len(s) for s in subsets]

    def resolve(src_d, r_s):
        '''For each friendship (starter src_d, wanted rank r_s among agents), return the agent at that rank.'''
        friend = np.empty(len(src_d), dtype=np.int64)

        # Pick the finest level whose neighbour list reaches the wanted rank: r_s / thin^level <= K
        level = np.zeros(len(src_d), dtype=np.int64)
        big = r_s > K_eff
        level[big] = np.ceil(np.log(r_s[big]/K_eff)/np.log(thin)).astype(np.int64)
        level = np.minimum(level, n_levels-1)

        for l in range(n_levels):
            sel = np.where(level == l)[0]
            if len(sel) == 0:
                continue
            kq = min(K_eff, sizes[l]-1) # Neighbours available at this level
            r_l = np.clip(np.rint(r_s[sel]/thin**l).astype(np.int64), 1, kq) # Rank within this level's (thinned) agents

            # Query each starter once, in batches, then pick the needed column for each of its friendships
            uniq, inv = np.unique(src_d[sel], return_inverse=True)
            order = np.argsort(inv, kind='stable')
            inv_sorted = inv[order]
            for a in range(0, len(uniq), chunk):
                b = min(a+chunk, len(uniq))
                rows = uniq[a:b]
                _, idx = trees[l].query(coords[rows], k=kq+1, workers=workers) # Nearest first; may include the agent itself
                is_self = subsets[l][idx[:, 0]] == rows # If the agent is in this level, it is its own nearest: drop it
                idx = np.where(is_self[:, None], idx[:, 1:], idx[:, :-1])
                lo, hi = np.searchsorted(inv_sorted, [a, b])
                d = order[lo:hi] # The friendships belonging to this batch of starters
                friend[sel[d]] = subsets[l][idx[inv[d]-a, r_l[d]-1]]
        return friend

    # Make contacts
    r_s = draw_sample_ranks(len(src))
    friend = resolve(src, r_s)

    # Keep each person's friends distinct by redrawing repeats
    if distinct:
        for _ in range(max_redraws):
            _, first = np.unique(src*N + friend, return_index=True)
            repeat = np.ones(len(src), dtype=bool)
            repeat[first] = False
            if not repeat.any():
                break
            ii = np.where(repeat)[0]
            r_s[ii] = draw_sample_ranks(len(ii))
            friend[ii] = resolve(src[ii], r_s[ii])

    # Tidy up: merge duplicate pairs (A-B and B-A are the same friendship)
    lo, hi = np.minimum(src, friend), np.maximum(src, friend)
    pairs  = np.unique(lo*N + hi)
    p1, p2 = pairs//N, pairs % N

    if mapping is not None:
        mapping = np.asarray(mapping)
        p1, p2  = mapping[p1], mapping[p2]

    output = dict(p1=p1.astype(cvd.default_int), p2=p2.astype(cvd.default_int))
    return output
