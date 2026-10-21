'''
Compare the fast rank network (src/rank_network_fast.py) with the original version: the edge lists must be exactly the
same, and the fast one should be quicker. The original is copied into this file, so it does not depend on any other file.

Run from network_rank_model/tests (with the covasim environment active):
    python test_fast_vs_original.py
'''
import inspect
import sys
import time
from pathlib import Path
import numpy as np
import covasim.defaults as cvd
import covasim.utils as cvu

sys.path.append(str(Path(__file__).resolve().parent.parent / 'src'))
# NOTE: seed with cvu.set_seed(), not np.random.seed(): Covasim's cvu.n_poisson runs in numba, which has its own random
# number generator that np.random.seed() does not reset. cvu.set_seed() resets both.
from rank_network_fast import make_rank_based_contacts_fast as fast   # the fast version, in src/rank_network_fast.py
assert 'n_workers' in inspect.signature(fast).parameters, 'the function imported as fast has no n_workers: the wrong file is being imported'


def original(coords, n, alpha=1.0, dispersion=None, mapping=None):
    ''' The original (slow) version of make_rank_based_contacts(), copied unchanged, to compare against '''
    coords   = np.asarray(coords, dtype=float)
    pop_size = len(coords) # Number of people
    p1 = [] # Initialize the "sources"
    p2 = [] # Initialize the "targets"
    if pop_size < 2: # Nobody to befriend
        return dict(p1=np.array([], dtype=cvd.default_int), p2=np.array([], dtype=cvd.default_int))
    rank_weights = np.arange(1, pop_size, dtype=float)**(-alpha)
    rank_cdf     = np.cumsum(rank_weights)
    rank_cdf    /= rank_cdf[-1]
    if dispersion is None:
        p_count = cvu.n_poisson(n, pop_size)
    else:
        p_count = cvu.n_neg_binomial(rate=n, dispersion=dispersion, n=pop_size)
    p_count = np.array((p_count/2.0).round(), dtype=cvd.default_int)
    for p in range(pop_size):
        n_friends = p_count[p]
        if n_friends == 0:
            continue
        ranks = np.searchsorted(rank_cdf, np.random.random(n_friends)) + 1
        ranks = np.minimum(ranks, pop_size-1)
        dist2 = ((coords - coords[p])**2).sum(axis=1)
        dist2[p] = np.inf
        order   = np.argpartition(dist2, np.unique(ranks-1))
        friends = order[ranks-1]
        p1.append(np.full(n_friends, p))
        p2.append(friends)
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

def points(N, seed):
    ''' A clustered set of locations: a dense core, a looser area and a sparse one '''
    rng = np.random.RandomState(seed)
    k = rng.choice(3, N, p=[0.6, 0.3, 0.1])
    return np.array([[0, 0], [15, 5], [40, 10]])[k] + rng.randn(N, 2)*np.array([4, 8, 15])[k][:, None]


def same(a, b):
    return np.array_equal(a['p1'], b['p1']) and np.array_equal(a['p2'], b['p2']) and a['p1'].dtype == b['p1'].dtype


all_ok = True
cases = [(300, 10, {}), (2500, 20, {}), (2500, 20, dict(alpha=1.5)), (2500, 20, dict(dispersion=1.0)),
         (2500, 12, dict(mapping=np.arange(2500)[::-1]))]
for N, n, kw in cases:
    c = points(N, N)
    cvu.set_seed(5); a = original(c, n, **kw)
    cvu.set_seed(5); b = fast(c, n, **kw)                        # one core
    cvu.set_seed(5); p = fast(c, n, n_workers=2, **kw)           # two cores
    ok = same(a, b) and same(a, p)
    all_ok &= ok
    print(f'N={N:5d} n={n:3d} {str(list(kw)):16s} edges {len(a["p1"]):6d}   fast == original: {same(a, b)}   two cores == original: {same(a, p)}')
print('\nALL IDENTICAL' if all_ok else '\nMISMATCH: do not use the fast version until this is fixed')

for N in (5000, 10000):
    c = points(N, 1)
    cvu.set_seed(7); t = time.time(); a = original(c, 20); t_old = time.time() - t
    cvu.set_seed(7); t = time.time(); b = fast(c, 20); t_new = time.time() - t
    print(f'N={N:6,d}, 20 friends, one core: original {t_old:5.1f} s, fast {t_new:5.1f} s ({t_old/t_new:.1f}x faster), identical: {same(a, b)}')