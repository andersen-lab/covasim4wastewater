'''
Rank-based geographic friendship network for Covasim.

Implements the model of Liben-Nowell et al. (2005), "Geographic routing in
social networks": person u links to person v with probability proportional to

    P(u -> v)  ~  1 / rank_u(v)^alpha,    alpha = 1 in the paper

where rank_u(v) = number of people w (w != u) that are at least as close to u
as v is. Because the probability depends on *rank* rather than distance, links
are automatically short in dense cities and long in sparse rural areas.

Paste these functions into covasim/population.py (and add them to __all__), or
keep as a separate module and import from population.py.
'''

import numpy as np
import sciris as sc
import covasim.defaults as cvd


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

def make_rank_based_contacts(coords, n, alpha=1.0, mapping=None):
    '''
    Make a friendship layer as an edgelist using rank-based attachment.

    Args:
        coords  (array): (N,2) coordinates in km
        n       (float): average number of friends per person (degree)
        alpha   (float): exponent on rank; 1.0 reproduces Liben-Nowell et al.
        mapping (array): optionally map generated indices onto new indices

    Returns:
        dict(p1=..., p2=...) like make_random_contacts()

    Method: for each person u, draw a number of links k ~ Poisson(n/2) (each
    undirected edge is "owned" by one end, so mean degree is n). For each link,
    draw a rank r in 1..N-1 with P(r) ~ r^-alpha, then connect to u's r-th
    nearest neighbour. Finding the r-th neighbour only needs np.argpartition
    (O(N)), not a full sort, and the all-pairs distance matrix is never stored.
    Total cost O(N^2) time, O(N) memory -- fine up to ~50k agents; see notes
    for scaling.
    '''
    coords = np.asarray(coords, dtype=float)
    N = len(coords)

    # CDF over ranks 1..N-1
    w = np.arange(1, N, dtype=float)**(-alpha)
    cdf = np.cumsum(w)
    cdf /= cdf[-1]

    counts = np.random.poisson(n/2.0, N)
    p1, p2 = [], []
    for u in range(N):
        k = counts[u]
        if k == 0:
            continue
        pos = np.minimum(np.searchsorted(cdf, np.random.random(k)), N-2)  # 0-based rank-1
        d = ((coords - coords[u])**2).sum(axis=1)
        d[u] = np.inf  # self sorts last, so positions 0..N-2 are the other people
        part = np.argpartition(d, np.unique(pos))
        p1.append(np.full(k, u))
        p2.append(part[pos])

    p1 = np.concatenate(p1) if p1 else np.empty(0)
    p2 = np.concatenate(p2) if p2 else np.empty(0)

    # Undirected de-duplication (u->v and v->u are the same edge)
    lo, hi = np.minimum(p1, p2), np.maximum(p1, p2)
    pairs = np.unique(np.column_stack((lo, hi)), axis=0)
    p1, p2 = pairs[:, 0], pairs[:, 1]

    if mapping is not None:
        mapping = np.asarray(mapping)
        p1, p2 = mapping[p1], mapping[p2]
    return dict(p1=p1.astype(cvd.default_int), p2=p2.astype(cvd.default_int))