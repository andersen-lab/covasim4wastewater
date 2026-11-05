'''
CHECK: does dispersion really switch make_rank_based_contacts from Poisson to the negative binomial?

It wraps the two draw functions (cvu.n_poisson and cvu.n_neg_binomial) so every call is recorded, then builds a small
rank network with and without dispersion and prints which function ran, and with which arguments.
Part 3 repeats it through a Sim, to see whether sim.initialize(dispersion=...) reaches the function.
Part 2 checks the negative binomial draws themselves: mean n, variance n + n^2/dispersion.

Run:  python check_dispersion_called.py
'''
import os
from pathlib import Path
import numpy as np
import covasim.utils as cvu
from covasim import population as cvpop

RUN_SIM_PIPELINE = True        # needs the real Covasim and the example boundary file in the repo's tests folder
REPO = Path(__file__).resolve().parents[2]

calls = []
real_poisson, real_negbin = cvu.n_poisson, cvu.n_neg_binomial
def poisson_spy(*args, **kwargs):
    calls.append('Poisson')
    return real_poisson(*args, **kwargs)
def negbin_spy(*args, **kwargs):
    calls.append('negative binomial ' + ', '.join(f'{k}={v}' for k, v in kwargs.items()))
    return real_negbin(*args, **kwargs)
cvu.n_poisson, cvu.n_neg_binomial = poisson_spy, negbin_spy

# ---- 1. the function, directly ----
print('1. make_rank_based_contacts(coords, 20, dispersion=...) on 500 random points')
coords = np.random.RandomState(0).uniform(0, 10, (500, 2))
for d in (None, 1.0):
    calls.clear()
    cvpop.make_rank_based_contacts(coords, 20, dispersion=d)
    print(f'   dispersion={d!s:5} -> {calls}')

# ---- 2. the draws themselves ----
cvu.n_poisson, cvu.n_neg_binomial = real_poisson, real_negbin
print('\n2. The draws: 200,000 values, rate 20')
for label, x in (('Poisson', real_poisson(20, 200000)), ('negative binomial, dispersion 1', real_negbin(rate=20, dispersion=1.0, n=200000)),
                 ('negative binomial, dispersion 10', real_negbin(rate=20, dispersion=10.0, n=200000))):
    print(f'   {label:34s} mean {x.mean():5.1f}   variance {x.var():6.1f}')
print('   expected: Poisson 20 / 20; dispersion 1: 20 / 420 (= 20 + 20^2/1); dispersion 10: 20 / 60 (= 20 + 20^2/10)')

# ---- 3. through a Sim ----
if RUN_SIM_PIPELINE:
    import covasim as cv
    cvu.n_poisson, cvu.n_neg_binomial = poisson_spy, negbin_spy
    os.chdir(REPO / 'tests')                      # the default boundary and sewershed files are looked up from here
    print('\n3. Through cv.Sim(pop_type="rank") and sim.initialize(...)')
    for d in (None, 1.0):
        calls.clear()
        sim = cv.Sim(pop_type='rank', pop_size=2000, n_days=2, verbose=0)
        sim.initialize(**({} if d is None else dict(dispersion=d)))
        print(f'   sim.initialize({"" if d is None else f"dispersion={d}"}) -> {calls}')