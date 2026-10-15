'''
Timing comparison plot: the original rank network against the fast version (default and use_workers=True) and random.
All numbers are copied from the timing runs; nothing is re-run here.

Run:  python make_timing_comparison_plot.py      ->  timing_comparison_plot.png
'''
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

# ---- seconds, from the timing runs ----
# Original rank network (earlier timing run); not timed beyond 100,000 agents
orig_n = [1000, 5000, 10000, 25000, 50000, 100000]
orig_s = [0.096, 1.533, 5.831, 31.575, 214.622, 631.312]
# Fast rank network (latest timing run, 128-core server)
fast_n          = [1000, 5000, 10000, 25000, 50000, 100000, 200000]
fast_default_s  = [0.08, 0.62, 1.7, 7.9, 27.2, 100.5, 374]            # default: one core
fast_workers_s  = [0.08, 0.37, 0.44, 0.86, 2.1, 7.5, 28.4]            # use_workers=True (up to 16 workers)
random_s        = [0.004, 0.017, 0.033, 0.079, 0.153, 0.317, 0.685]   # Covasim make_random_contacts

series = [  # label, agents, seconds, colour, label the points at these sizes
    ('random',                                          fast_n, random_s,       'tab:gray',   []),
    ('rank, original',                                  orig_n, orig_s,         'tab:blue',   [25000, 50000, 100000]),
    ('rank, fast (default: one core)',                  fast_n, fast_default_s, 'tab:orange', [100000, 200000]),
    ('rank, fast, use_workers=True (up to 16 workers)', fast_n, fast_workers_s, 'tab:green',  [100000, 200000]),
]

fig, ax = plt.subplots(figsize=(9, 6))
for label, n, s, color, labelled in series:
    minutes = np.array(s)/60
    ax.plot(n, minutes, 'o-', color=color, label=label)
    for x, m in zip(n, minutes):
        if x in labelled:
            ax.annotate(f'{m:.1f} min', (x, m), textcoords='offset points', xytext=(-8, -16) if color == 'tab:orange' else (-8, 8),
                        ha='right', fontsize=9, color=color)
ax.set_xlim(0, 210000); ax.set_ylim(0, 12)
ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f'{int(v):,}'))
ax.set_xlabel('number of agents'); ax.set_ylabel('time to build network (minutes)')
ax.set_title('Network build time vs number of agents (20 friends per person)')
ax.grid(True, alpha=0.3); ax.legend(fontsize=9, loc='upper right')
plt.tight_layout(); plt.savefig('timing_comparison_plot.png', dpi=150)
print('saved timing_comparison_plot.png')