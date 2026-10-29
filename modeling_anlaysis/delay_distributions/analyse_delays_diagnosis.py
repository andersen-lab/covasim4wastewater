# %% Imports and settings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import covasim as cv

POP_TYPE   = 'rank'          # your fork's pop type
POP_SIZE   = 50_000
N_DAYS     = 120
SYMP_PROBS = [0.03, 0.10, 0.20, 0.35]
SEEDS      = [1, 2, 3]
CUTOFF     = 60              # only use people exposed on/before this day

# %% Run simulations (slow, run once)
sims = {}   # (symp_prob, seed) -> finished sim

for sp in SYMP_PROBS:
    for seed in SEEDS:
        test = cv.test_prob(symp_prob=sp, asymp_prob=0.0, test_delay=0, start_day=0)
        sim = cv.Sim(pop_type=POP_TYPE, pop_size=POP_SIZE, n_days=N_DAYS,
                     rand_seed=seed, verbose=0, interventions=test)
        sim.run()
        sims[(sp, seed)] = sim
        print(f'done: symp_prob={sp}, seed={seed}')

# %% Sanity check: raw per-person dates from one sim
sim = sims[(0.10, 1)]
p = sim.people
df = pd.DataFrame({
    'exposed':     p.date_exposed,
    'symptomatic': p.date_symptomatic,
    'diagnosed':   p.date_diagnosed,
})
df = df[np.isfinite(df.exposed)]
df['inf_to_diag'] = df.diagnosed - df.exposed
df.head(20)

# %% Extract delays
def extract_delays(sim, cutoff=CUTOFF):
    p = sim.people
    exposed, diagnosed, symptom = p.date_exposed, p.date_diagnosed, p.date_symptomatic
    early    = np.isfinite(exposed) & (exposed <= cutoff)
    got_diag = early & np.isfinite(diagnosed)
    has_symp = got_diag & np.isfinite(symptom)
    return dict(
        inf_to_diag  = (diagnosed - exposed)[got_diag],
        symp_to_diag = (diagnosed - symptom)[has_symp],
        n_early_inf  = int(early.sum()),
        n_diag       = int(got_diag.sum()),
    )

results = {}
for sp in SYMP_PROBS:
    outs = [extract_delays(sims[(sp, s)]) for s in SEEDS]
    results[sp] = dict(
        inf_to_diag  = np.concatenate([o['inf_to_diag'] for o in outs]),
        symp_to_diag = np.concatenate([o['symp_to_diag'] for o in outs]),
        frac_diag    = sum(o['n_diag'] for o in outs) / max(sum(o['n_early_inf'] for o in outs), 1),
    )

# %% Summary table
summary = pd.DataFrame({
    sp: {
        'n_diagnosed':    len(r['inf_to_diag']),
        'frac_diagnosed': round(r['frac_diag'], 3),
        'mean':           r['inf_to_diag'].mean(),
        'sd':             r['inf_to_diag'].std(),
        'median':         np.median(r['inf_to_diag']),
        'p90':            np.percentile(r['inf_to_diag'], 90),
    } for sp, r in results.items()
}).T.round(2)
summary.index.name = 'symp_prob'
print(summary)

# %% Plots (re-run freely to tweak)
fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
colors = plt.cm.viridis(np.linspace(0, 0.9, len(SYMP_PROBS)))
bins = np.arange(0, 61, 1)

for c, (sp, r) in zip(colors, results.items()):
    d = r['inf_to_diag']
    axes[0].hist(d, bins=bins, density=True, histtype='step', lw=2, color=c, label=f'{sp:.2f}')
    xs = np.sort(d)
    axes[1].plot(xs, np.arange(1, len(xs)+1)/len(xs), color=c, lw=2, label=f'{sp:.2f}')
    axes[2].hist(r['symp_to_diag'], bins=bins, density=True, histtype='step', lw=2, color=c, label=f'{sp:.2f}')

axes[0].set(title='Infection → diagnosis delay', xlabel='days', ylabel='density')
axes[1].set(title='Same, cumulative (ECDF)', xlabel='days', ylabel='fraction diagnosed')
axes[2].set(title='Symptom onset → diagnosis', xlabel='days')
for ax in axes:
    ax.legend(title='symp_prob'); ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig('delay_vs_symp_prob.png', dpi=150)
plt.show()
# %%

#### Check dispersion in the rank model
sim = cv.Sim(pop_type='rank', pop_size=5000, n_days=5, verbose=0)
sim.initialize(dispersion=1.0)
sim.run()
