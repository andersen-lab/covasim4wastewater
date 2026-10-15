'''
How many CPU cores should the fast rank network use? Times the same network build with different numbers of workers,
so you can see where adding more cores stops helping.

Run from network_rank_model/tests:   python test_worker_sweep.py
(about 3 minutes at the default size, mostly the one-worker run; change N below for a bigger or smaller test)

The fast rank network never uses more than one worker per 2,000 agents, so at N = 100,000 the most it can use is 50.
Worker counts above that give the same result, so the table skips them.
'''
import os
import sys
import time
from pathlib import Path
import numpy as np
import covasim.utils as cvu

sys.path.append(str(Path(__file__).resolve().parent.parent / 'src'))
from rank_network_fast import make_rank_based_contacts_fast as fast

N        = 100000         # agents
DEG      = 20             # average friends per person
WORKERS  = [1, 2, 4, 8, 16, 32, 64, 128]

# ---- what this machine offers ----
allowed = len(os.sched_getaffinity(0)) if hasattr(os, 'sched_getaffinity') else None
print(f'cores reported by the machine: {os.cpu_count()};  cores this job is allowed to use: {allowed}')
if hasattr(os, 'getloadavg'):
    print(f'load average over the last 1, 5 and 15 minutes: {os.getloadavg()[0]:.1f}, {os.getloadavg()[1]:.1f}, {os.getloadavg()[2]:.1f}  (if this is already high, other jobs are using the machine)')

# ---- a clustered set of locations: a dense core, a looser area and a sparse one ----
rng = np.random.RandomState(0)
k = rng.choice(3, N, p=[0.6, 0.3, 0.1])
coords = np.array([[0, 0], [15, 5], [40, 10]])[k] + rng.randn(N, 2)*np.array([4, 8, 15])[k][:, None]

# ---- time each worker count ----
limit = allowed or os.cpu_count()
rows, reference, seen = [], None, set()
print(f'\n{N:,} agents, {DEG} friends per person\n')
print(f'{"asked for":>10} {"workers used":>13} {"seconds":>9} {"speedup":>9} {"efficiency":>11}')
for w in WORKERS:
    used = max(1, min(w, limit, N//2000))      # the same caps the function applies
    if used in seen:
        continue                               # asking for more would not change anything
    seen.add(used)
    cvu.set_seed(1)
    t = time.time(); out = fast(coords, DEG, n_workers=w); dt = time.time() - t
    key = (out['p1'].tobytes(), out['p2'].tobytes())
    reference = reference or key
    speedup = rows[0][1]/dt if rows else 1.0
    rows.append((used, dt))
    print(f'{w:>10} {used:>13} {dt:>9.1f} {speedup:>8.1f}x {speedup/used:>10.0%}' + ('' if key == reference else '   <-- DIFFERENT NETWORK (should not happen)'), flush=True)

best = min(rows, key=lambda r: r[1])
print(f'\nFastest: {best[0]} workers ({best[1]:.1f} s). Efficiency is the speedup divided by the number of workers: 100% would be perfect.')