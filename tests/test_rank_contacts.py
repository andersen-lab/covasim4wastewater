'''
Unit tests for the rank-based friendship network (covasim/population.py):
    - the format of the edge list
    - the new fast version gives exactly the same network as the original (kept as make_rank_based_contacts_basic)
    - use_workers / n_workers change how fast the network is built, not the network
    - how many workers are chosen for a given computer and size
No raster or optional packages are needed.
'''
import os
import numpy as np
import covasim.defaults as cvd
import covasim.utils as cvu
from covasim import population as cvpop


def make_points(n, seed=0):
    ''' A clustered set of locations: a dense core, a looser area and a sparse one '''
    rng = np.random.RandomState(seed)
    k = rng.choice(3, n, p=[0.6, 0.3, 0.1])
    return np.array([[0, 0], [15, 5], [40, 10]])[k] + rng.randn(n, 2)*np.array([4, 8, 15])[k][:, None]


def same(a, b):
    return np.array_equal(a['p1'], b['p1']) and np.array_equal(a['p2'], b['p2'])


def test_rank_contacts_format():
    ''' An edge list in the same format as make_random_contacts: two integer arrays, each friendship once, nobody their own friend '''
    n = 500
    e = cvpop.make_rank_based_contacts(make_points(n), 10)
    assert set(e.keys()) == {'p1', 'p2'}
    assert e['p1'].dtype == cvd.default_int and e['p2'].dtype == cvd.default_int
    assert len(e['p1']) == len(e['p2']) > 0
    assert np.all(e['p1'] != e['p2']) # Nobody befriends themselves
    pairs = e['p1'].astype(np.int64)*n + e['p2']
    assert len(np.unique(pairs)) == len(pairs) # Each friendship is listed once


def test_rank_contacts_match_original():
    ''' The fast version gives exactly the same network as the original for the same seed '''
    coords = make_points(2000)
    for kwargs in (dict(), dict(alpha=1.5), dict(dispersion=1.0)):
        cvu.set_seed(5); original = cvpop.make_rank_based_contacts_basic(coords, 15, **kwargs)
        cvu.set_seed(5); fast     = cvpop.make_rank_based_contacts(coords, 15, **kwargs)
        assert same(original, fast), f'the networks differ for {kwargs}'


def test_rank_contacts_workers_same_network(monkeypatch):
    ''' use_workers and n_workers change the speed, not the network. Pretend four idle cores so more than one worker really starts. '''
    monkeypatch.setattr(os, 'sched_getaffinity', lambda pid: {0, 1, 2, 3}, raising=False)
    monkeypatch.setattr(os, 'getloadavg', lambda: (0.0, 0.0, 0.0), raising=False)
    coords = make_points(6000)
    cvu.set_seed(1); one_core = cvpop.make_rank_based_contacts(coords, 20)
    for kwargs in (dict(use_workers=True), dict(n_workers=2)):
        cvu.set_seed(1); other = cvpop.make_rank_based_contacts(coords, 20, **kwargs)
        assert same(one_core, other), f'the network changed with {kwargs}'


def test_choose_workers(monkeypatch):
    ''' One core by default; use_workers=True picks a number that depends on the computer and the size '''
    def machine(cores, load):
        monkeypatch.setattr(os, 'sched_getaffinity', lambda pid: set(range(cores)), raising=False)
        monkeypatch.setattr(os, 'getloadavg', lambda: (load, load, load), raising=False)

    machine(128, 10)
    assert cvpop.choose_workers(100000) == 1                                    # the default is one core
    assert cvpop.choose_workers(100000, use_workers=True) == 16                 # never more than 16
    assert cvpop.choose_workers(5000, use_workers=True) == 2                    # about 2,000 agents per worker
    assert cvpop.choose_workers(1500, use_workers=True) == 1                    # too small to split
    assert cvpop.choose_workers(100000, n_workers=64) == 50                     # asked for 64, but the size allows 50
    assert cvpop.choose_workers(100000, use_workers=True, n_workers=1) == 1     # n_workers overrides use_workers
    machine(8, 0.5)
    assert cvpop.choose_workers(100000, use_workers=True) == 7                  # one core left free
    machine(8, 7.5)
    assert cvpop.choose_workers(100000, use_workers=True) == 1                  # a busy machine gets one worker
    machine(1, 0.0)
    assert cvpop.choose_workers(100000, use_workers=True) == 1                  # a single core stays serial
