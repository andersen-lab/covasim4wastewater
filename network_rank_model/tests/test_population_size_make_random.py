import time, resource
from covasim.population import make_random_contacts
for N in [100_000, 1_000_000, 3_400_000]:
    t = time.time(); e = make_random_contacts(N, 20)
    print(N, f'{time.time()-t:.0f} s', f'{len(e["p1"]):,} edges',
          f'peak mem {resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1e6:.1f} GB')