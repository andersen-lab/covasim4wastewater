'''
Fast version of the rank-based friendship network (same result as rank_network.py, about 3 times faster on one core,
and it can use several cores: pass use_workers=True and it chooses how many for this computer (or n_workers=N to choose yourself).

The main function is called make_rank_based_contacts_fast() here, so it cannot be confused with the original
make_rank_based_contacts() while both are being tested. In covasim/population.py it takes the name make_rank_based_contacts().

This file is for testing next to rank_network.py. In Covasim itself, the same two helper functions and
make_rank_based_contacts() live in covasim/population.py.
'''
import numpy as np
import covasim.defaults as cvd
import covasim.utils as cvu


_SHARED = {} # The coordinates, shared with worker processes that are started by fork (so they are not sent again with every job)
_MIN_AGENTS_PER_WORKER = 2000 # A rough guide: fewer agents than this per worker and starting the workers costs more than it saves
_AUTO_MAX_WORKERS = 16 # use_workers=True never picks more than this: on the test server (100,000 agents) efficiency stayed above about 80% up to 16 workers


def choose_workers(pop_size, use_workers=False, n_workers=None):
    '''
    Helper for make_rank_based_contacts(): how many worker processes to use. One means no parallelism.
        n_workers given    use that many (-1 = every core this job may use), but never more than the cores available or the
                           size justifies (about 2,000 agents per worker)
        use_workers=True   choose automatically: leave one core free, never more than 16 (more cores help less and less),
                           and only count cores that are not already busy with other work
        neither            one core
    '''
    import os
    usable = len(os.sched_getaffinity(0)) if hasattr(os, 'sched_getaffinity') else (os.cpu_count() or 1) # Cores this job may really use
    if n_workers is not None:
        wanted = usable if n_workers < 0 else int(n_workers)
    elif use_workers:
        wanted = usable - 1 if usable > 2 else usable
        wanted = min(wanted, _AUTO_MAX_WORKERS)
        try:
            wanted = min(wanted, int(usable - os.getloadavg()[0])) # Cores already busy with other work are not free
        except (AttributeError, OSError): # getloadavg does not exist on every platform
            pass
    else:
        wanted = 1
    return max(1, min(wanted, usable, pop_size//_MIN_AGENTS_PER_WORKER))


def pick_ranks(d, ranks):
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


def find_friends(x, y, people, counts, ranks):
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
        friends[start:start+k] = pick_ranks(dx, ranks[start:start+k])
        start += k
    return friends


def friends_job(people, counts, ranks):
    '''
    Helper for make_rank_based_contacts(): one job for a worker process. The coordinates come from _SHARED, which the
    worker inherited when it was forked, so only the small per-job arrays have to be sent to it.
    '''
    return find_friends(_SHARED['x'], _SHARED['y'], people, counts, ranks)


def make_rank_based_contacts_fast(coords, n, alpha=1.0, dispersion=None, mapping=None, use_workers=False, n_workers=None):
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
        use_workers (bool) : False (the default) uses one core, which works everywhere. True lets the code choose how many
                             cores to use for this computer and this size (see below).
        n_workers  (int)   : use this many cores instead (-1 = all the cores this job may use). Overrides use_workers.
                             Never more than the cores available or about one per 2,000 agents.
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
        - use_workers=True picks the number of workers itself: one core left free, at most 16, only cores that are not
          already busy, and about one worker per 2,000 agents. Below about 4,000 agents it stays on one core. If worker
          processes cannot be started (for example inside MultiSim) it warns and uses one core.
        - All random numbers are drawn first, in one step, so the result does not depend on the number of workers.
        - Cost still grows roughly with N^2, because every person's distance to everyone is needed.
          The fast distance step and the nested partial sorts make it about 3 to 4 times faster on one core
          than the first version (measured at 20,000 people); n_workers divides the time by about the
          number of cores used. make_random_contacts() never looks at distances, so its time grows only with N.
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
    n_workers = choose_workers(pop_size, use_workers, n_workers) # One core unless asked otherwise

    friends = None
    if n_workers > 1:
        import multiprocessing as mp
        if 'fork' in mp.get_all_start_methods(): # Forked workers inherit the coordinates; other platforms use one core
            from concurrent.futures import ProcessPoolExecutor
            ctx    = mp.get_context('fork')
            ends   = np.cumsum(counts)
            starts = ends - counts
            groups = np.array_split(np.arange(len(people)), n_workers*4) # Several small jobs per core keeps all cores busy
            jobs   = [g for g in groups if len(g)]
            _SHARED['x'], _SHARED['y'] = x, y
            try:
                with ProcessPoolExecutor(max_workers=n_workers, mp_context=ctx) as pool:
                    futures = [pool.submit(friends_job, people[g], counts[g], ranks[starts[g[0]]:ends[g[-1]]]) for g in jobs]
                    friends = np.concatenate([f.result() for f in futures])
            except (AssertionError, OSError, RuntimeError) as E: # e.g. already inside a worker process (MultiSim), or not allowed to start processes
                import warnings
                warnings.warn(f'Could not start worker processes for the rank network ({E}); using one core instead')
                friends = None
            finally:
                _SHARED.clear()
    if friends is None: # One core: the default, and the fallback
        friends = find_friends(x, y, people, counts, ranks)

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