# ----------------------------------------------------------------------------
# Paste these two helper functions into covasim/population.py ABOVE make_rank_based_contacts(),
# and replace make_rank_based_contacts() with the version below.
# ----------------------------------------------------------------------------
import numpy as np
import covasim.defaults as cvd
import covasim.utils as cvu

def _pick_ranks(d, ranks):
    '''
    Helper for make_rank_based_contacts(): given the squared distances d from one person to everyone, return who sits
    at each wanted rank (1 = nearest). The largest rank is found first with one partial sort; the smaller ranks are then
    searched only among the people already known to be closer, so each later step works on a smaller list.
    '''
    uniq, inverse = np.unique(ranks, return_inverse=True) # Wanted ranks, smallest first (a rank asked for twice is found once)
    idx   = np.arange(len(d)) # Who each entry of d belongs to (d gets reordered below)
    found = np.empty(len(uniq), dtype=np.int64)
    for j in range(len(uniq)-1, -1, -1): # Largest rank first
        r     = uniq[j]
        part  = np.argpartition(d, r-1) # Puts the r-th nearest at position r-1, the r nearest before it
        idx, d = idx[part], d[part]
        found[j] = idx[r-1]
        idx, d = idx[:r], d[:r] # The smaller ranks can only be among these r people
    return found[inverse]


def _friends_for_people(x, y, people, counts, ranks):
    '''
    Helper for make_rank_based_contacts(): for each person in `people`, find the person at each of their wanted ranks.
    ranks holds the wanted ranks of all these people one after another (counts[i] of them for people[i]).
    '''
    friends = np.empty(len(ranks), dtype=np.int64)
    dx = np.empty(len(x)) # Reused for every person, so no new arrays are made inside the loop
    dy = np.empty(len(x))
    start = 0
    for p, k in zip(people, counts):
        np.subtract(x, x[p], out=dx)
        np.subtract(y, y[p], out=dy)
        np.multiply(dx, dx, out=dx)
        np.multiply(dy, dy, out=dy)
        np.add(dx, dy, out=dx) # Squared distance from person p to everyone (squared is fine for ordering)
        dx[p] = np.inf # Put self last so it can never be chosen
        friends[start:start+k] = _pick_ranks(dx, ranks[start:start+k])
        start += k
    return friends


def make_rank_based_contacts_fast(coords, n, alpha=1.0, dispersion=None, mapping=None, n_workers=1):
    '''
    Make a rank-based geographic friendship layer as an edgelist.

    Model: Liben-Nowell et al. (2005), "Geographic routing in social networks".
    Person u links to person v with probability proportional to

        P(u -> v)  ~  1 / rank_u(v)^alpha,    alpha = 1 in the paper

    where rank_u(v) is the number of people w (w != u) who are at least as close to u
    as v is. Rank (not distance) means links are short in dense areas and long in
    sparse areas.

    Args:
        coords     (array) : (N,2) array of x,y locations in km, one row per person
        n          (float) : the average number of friends per person for this layer
        alpha      (float) : how strongly friendships favour nearby people (0 = ignore
                             distance, 1.0 = the value from Liben-Nowell et al., larger = more local)
        dispersion (float) : if not None, use a negative binomial distribution with this
                             dispersion parameter instead of Poisson for friend counts
        mapping    (array) : optionally map the generated indices onto new indices
        n_workers  (int)   : number of CPU cores to use (1 = no parallelism; None or -1 = all cores).
                             The result is identical whatever the number of cores.

    Returns:
        Dictionary of two arrays, p1 and p2, defining the edgelist (friend pairs), in the
        same format as make_random_contacts() so it can be used as a contact layer.

    Method, for each person p:
        1. Draw how many friends p picks: a random (Poisson) number averaging n/2. Every
           friendship is picked by one of its two people, so a person's total comes from the
           n/2 they pick plus about n/2 picked by others, about n friends in all.
        2. For each friendship, pick a rank r from 1 to N-1. Small ranks are much more
           likely than large ones (probability proportional to 1/r^alpha).
        3. Connect p to the person who is r-th nearest to them (found by partially sorting the
           distances from p to everyone, not a full sort).
        4. Finally, pairs found twice (A-B and B-A, or A-B repeated) are merged into one friendship.
           This is why the average number of friends comes out slightly below n (about 0.9n to 0.95n).

    Notes:
        - All random numbers are drawn first, in one step, so the result does not depend on n_workers.
        - Cost still grows roughly with N^2, because every person's distance to everyone is needed.
          The fast distance step and the nested partial sorts make it about 3 to 4 times faster on one core
          than the first version (measured at 20,000 people); n_workers divides the time by about the
          number of cores used. make_random_contacts() never looks at distances, so its time grows only with N.
        - Do not use n_workers > 1 when this is already running inside a parallel job (e.g. MultiSim).
    '''

    # Preprocessing
    coords   = np.asarray(coords, dtype=float)
    pop_size = len(coords) # Number of people
    empty    = dict(p1=np.array([], dtype=cvd.default_int), p2=np.array([], dtype=cvd.default_int))
    if pop_size < 2: # Nobody to befriend
        return empty

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

    # Draw every rank up front (the same random numbers, in the same order, as drawing them person by person)
    people = np.flatnonzero(p_count > 0) # Who starts any friendships
    counts = p_count[people].astype(np.int64)
    n_draws = int(counts.sum())
    if n_draws == 0:
        return empty
    ranks = np.searchsorted(rank_cdf, np.random.random(n_draws)) + 1 # Rank 1 = nearest person
    ranks = np.minimum(ranks, pop_size-1) # Guard against rounding at the far end

    # Find who sits at each chosen rank, on one core or several
    x, y = np.ascontiguousarray(coords[:, 0]), np.ascontiguousarray(coords[:, 1])
    if n_workers is None or n_workers < 0:
        import os
        n_workers = os.cpu_count() or 1
    if n_workers <= 1 or pop_size < 2000: # Too small to be worth the start-up cost
        friends = _friends_for_people(x, y, people, counts, ranks)
    else:
        from concurrent.futures import ProcessPoolExecutor
        import multiprocessing as mp
        ends   = np.cumsum(counts)
        starts = ends - counts
        groups = np.array_split(np.arange(len(people)), n_workers*4) # Several small jobs per core keeps all cores busy
        jobs   = [g for g in groups if len(g)]
        try:
            ctx = mp.get_context('fork') # Cheapest on Linux: workers share the coordinates
        except ValueError: # pragma: no cover
            ctx = mp.get_context()
        with ProcessPoolExecutor(max_workers=n_workers, mp_context=ctx) as pool:
            futures = [pool.submit(_friends_for_people, x, y, people[g], counts[g], ranks[starts[g[0]]:ends[g[-1]]]) for g in jobs]
            friends = np.concatenate([f.result() for f in futures])

    # Tidy up: merge duplicate pairs (A-B and B-A are the same friendship)
    p1 = np.repeat(people, counts)
    p2 = friends
    lo, hi = np.minimum(p1, p2), np.maximum(p1, p2)
    pairs  = np.unique(lo*pop_size + hi) # One number per pair, so unique() is much faster than comparing rows
    p1, p2 = pairs//pop_size, pairs % pop_size

    if mapping is not None:
        mapping = np.asarray(mapping)
        p1, p2  = mapping[p1], mapping[p2]

    output = dict(p1=p1.astype(cvd.default_int), p2=p2.astype(cvd.default_int))
    return output