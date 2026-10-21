'''
CHECK: is the rank-based network really rank-based?

The model: person u picks person v as a friend with probability proportional to 1 / rank^alpha, where rank is v's
position in u's list of everyone sorted by distance (1 = nearest). This script checks the code against that definition.
It uses made-up clustered points (a dense core, a looser area, a sparse area), so it needs no data files.
It is a check for you, not part of the Covasim test suite.

  1. Right person at each rank. For 300 picks, sort everyone by distance from the picker (brute force) and check that
     the friend returned is the person at the chosen rank. Expect 0 wrong.
  2. Ranks follow the 1/rank law (alpha = 1). The share of picks with rank <= r should match the law,
     sum(1/k, k <= r) / sum(1/k, k <= N-1). The z column is the gap in standard errors; expect |z| below 4.
  3. Distance depends on crowding, rank does not. People are grouped by local crowding (distance to their 10th nearest
     neighbour). In the rank network the median friend distance grows from dense to sparse areas, but the median
     rank of a pick stays about the same. That constant rank is the model's signature. Random friends are far everywhere.
  4. alpha works. The median rank of a pick should match the law's median (alpha 0: about N/2, alpha 1: about 50,
     alpha 2: 1).

Run (one core, so the picks can be recorded):   python check_rank_model.py
'''
import numpy as np
import pandas as pd
import covasim.utils as cvu
from covasim import population as cvpop

N, DEG = 5000, 20

def make_points(n, seed=0):
    ''' A dense core, a looser area and a sparse area '''
    rng = np.random.RandomState(seed)
    k = rng.choice(3, n, p=[0.6, 0.3, 0.1])
    return np.array([[0, 0], [15, 5], [40, 10]])[k] + rng.randn(n, 2)*np.array([4, 8, 15])[k][:, None]

coords = make_points(N)

# Record every pick (who picked, which rank was drawn, who was returned) by wrapping find_friends
picks = {}
real_find_friends = cvpop.find_friends
def recording_find_friends(x, y, people, counts, ranks):
    friends = real_find_friends(x, y, people, counts, ranks)
    picks.update(picker=np.repeat(people, counts), ranks=ranks, friends=friends)
    return friends
cvpop.find_friends = recording_find_friends

def dist_to_all(p):
    d = (coords[:, 0] - coords[p, 0])**2 + (coords[:, 1] - coords[p, 1])**2
    d[p] = np.inf
    return d

def build(alpha=1.0, seed=1):
    cvu.set_seed(seed)
    edges = cvpop.make_rank_based_contacts(coords, DEG, alpha=alpha)
    return edges, dict(picks)

# ---- 1. right person at each rank ----
edges, p = build()
rng = np.random.RandomState(0)
sample = rng.choice(len(p['ranks']), 300, replace=False)
wrong = sum(np.argsort(dist_to_all(p['picker'][i]))[p['ranks'][i] - 1] != p['friends'][i] for i in sample)
print(f'1. Right person at the chosen rank: {wrong} wrong out of {len(sample)} picks checked (expect 0)')

# ---- 2. ranks follow the 1/rank law ----
alpha = 1.0
cdf = np.cumsum(np.arange(1, N)**-alpha); cdf /= cdf[-1]
ranks = p['ranks']; n_picks = len(ranks)
print(f'\n2. Share of picks with rank <= r, from {n_picks:,} picks (alpha = 1)')
print(f'   {"r":>6} {"expected":>9} {"observed":>9} {"z":>6}')
worst = 0
for r in (1, 3, 10, 30, 100, 300, 1000):
    expected, observed = cdf[r - 1], np.mean(ranks <= r)
    z = (observed - expected)/np.sqrt(expected*(1 - expected)/n_picks)
    worst = max(worst, abs(z))
    print(f'   {r:>6} {expected:>8.1%} {observed:>8.1%} {z:>6.1f}')
print('   ' + ('matches the 1/rank law' if worst < 4 else 'DOES NOT match the 1/rank law'))

# ---- 3. short links in dense areas, long links in sparse areas ----
def median_friend_distance(e):
    d = np.hypot(coords[e['p1'], 0] - coords[e['p2'], 0], coords[e['p1'], 1] - coords[e['p2'], 1])
    person = np.concatenate([e['p1'], e['p2']])
    return pd.Series(np.concatenate([d, d])).groupby(person).median()

people = rng.choice(N, 1500, replace=False)
density = pd.Series([np.sort(dist_to_all(q))[9] for q in people], index=people)   # distance to the 10th nearest person
random_edges = cvpop.make_random_contacts(N, DEG)
table = pd.DataFrame({'local_spacing_km': density,
                      'rank_friend_km': median_friend_distance(edges).reindex(people),
                      'random_friend_km': median_friend_distance(random_edges).reindex(people)})
table['rank_of_pick'] = pd.Series(p['ranks']).groupby(p['picker']).median().reindex(people)
table['area'] = pd.qcut(table['local_spacing_km'], 4, labels=['densest', 'dense', 'sparse', 'sparsest'])
print('\n3. By how crowded the area is: median friend distance (km) and median rank of a pick')
print(table.groupby('area', observed=True).median().round(2).to_string())

# ---- 4. alpha works ----
print('\n4. Median rank of a pick, by alpha')
print(f'   {"alpha":>6} {"observed":>10} {"expected":>10}')
for a in (0.0, 1.0, 2.0):
    law = np.cumsum(np.arange(1, N)**-a); law /= law[-1]
    _, pa = build(alpha=a)
    print(f'   {a:>6} {np.median(pa["ranks"]):>10,.0f} {np.searchsorted(law, 0.5) + 1:>10,}')