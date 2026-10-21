'''Throwaway check: does sim.initialize(use_workers=True) reach the rank function? Not part of the repo.'''
import covasim as cv
from covasim import population as cvpop

original = cvpop.choose_workers
def spy(pop_size, use_workers=False, n_workers=None):
    n = original(pop_size, use_workers, n_workers)
    print(f'choose_workers called: pop_size={pop_size}, use_workers={use_workers}, n_workers={n_workers} -> {n} workers')
    return n
cvpop.choose_workers = spy

sim = cv.Sim(pop_type='rank', pop_size=5000, n_days=5, verbose=0)
_ = sim.initialize(use_workers=True)
sim.run()
print(list(sim.people.contacts.keys()))      # expect ['a']