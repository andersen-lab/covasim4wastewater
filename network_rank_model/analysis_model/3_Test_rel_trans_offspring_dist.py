'''
Aim: (1) keep track of the values of every term in this line of cvu.compute_trans_sus:

        rel_trans = rel_trans * inf * f_quar * f_asymp * f_iso * beta_layer * viral_load

     for every infectious person on every day, and (2) plot the offspring distribution
     (number of secondary infections caused per person).

How (1) works: compute_trans_sus is compiled by numba, so we cannot print inside it. Instead an
analyzer runs at the end of every sim.step() and recomputes the same terms from the same people
arrays. (Infecting someone does not change who is infectious, so the values match what step() used.)

    rel_trans   : people.rel_trans, the person's own draw from beta_dist (neg binomial, par2 = k)
    f_asymp     : symp + ~symp * asymp_factor           (1 if symptomatic, asymp_factor otherwise)
    f_iso       : ~iso  + iso  * iso_factor[layer]      (1 if not isolated, iso_factor if isolated)
    f_quar      : ~quar + quar * quar_factor[layer]     (1 if not quarantined, quar_factor if quarantined)
    beta_layer  : sim['beta_layer'][layer]
    viral_load  : people.viral_load, from the viral load curve over the infection
    final       : the product of all of these (what compute_trans_sus returns for an infectious person)

Note: this ignores the optional haplotype fitness multiplier (evo_pars), which is off by default.

Run:  python track_rel_trans_and_offspring.py
'''
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import covasim as cv

# ---------------------------------------------------------------------------
# Settings
# ---------------------------------------------------------------------------
N          = 20000
N_DAYS     = 90
SEED       = 1
N_INITIAL  = 20
BETA       = 0.016
K_INFECT   = 0.45        # dispersion of infectiousness (Covasim default). Change this for the second file.
POP_TYPE   = 'hybrid'    # 'hybrid' has layers h, s, w, c. Use 'random' for a single layer 'a'.
OUT        = f'rel_trans_offspring_k{K_INFECT:g}'


# ---------------------------------------------------------------------------
# Analyzer: records every term of the rel_trans line for each infectious person, each day, each layer
# ---------------------------------------------------------------------------
class RelTransTracker(cv.Analyzer):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.chunks = []
        self.beta_layer = {}

    def initialize(self, sim):
        super().initialize(sim)
        self.beta_layer = {k: float(v) for k, v in sim['beta_layer'].items()}

    def apply(self, sim):
        p = sim.people
        idx = np.flatnonzero(p.infectious)
        if len(idx) == 0:
            return
        symp = p.symptomatic[idx]
        iso  = p.isolated[idx]
        quar = p.quarantined[idx]
        base = p.rel_trans[idx]
        vl   = p.viral_load[idx]
        f_asymp = symp + ~symp * sim['asymp_factor']
        for lkey in sim.layer_keys():
            f_iso  = ~iso  + iso  * sim['iso_factor'][lkey]
            f_quar = ~quar + quar * sim['quar_factor'][lkey]
            bl = sim['beta_layer'][lkey]
            final = base * f_quar * f_asymp * f_iso * bl * vl
            self.chunks.append(pd.DataFrame(dict(
                day=sim.t, person=idx, layer=lkey, rel_trans=base, f_asymp=f_asymp, f_iso=f_iso,
                f_quar=f_quar, beta_layer=bl, viral_load=vl, final=final)))

    def table(self):
        return pd.concat(self.chunks, ignore_index=True)


# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------
tracker = RelTransTracker(label='tracker')
sim = cv.Sim(pop_type=POP_TYPE, pop_size=N, pop_infected=N_INITIAL, n_imports=0, beta=BETA, n_days=N_DAYS,
             rand_seed=SEED, verbose=0, analyzers=tracker,
             beta_dist=dict(dist='neg_binomial', par1=1.0, par2=K_INFECT, step=0.01))
sim.run()
tracker = sim.get_analyzer('tracker')     # sim deep-copies the analyzer, so fetch the one that actually ran
df = tracker.table()

layer0 = sim.layer_keys()[0]
first = df[df.layer == layer0]            # one row per infectious person-day (layer-independent terms are the same in every layer)

print(f'\nbeta_layer: {tracker.beta_layer}')
print(f'{len(df):,} rows (infectious person x day x layer)')
print(f'\nSummary of each term over infectious person-days (layer "{layer0}"):')
print(first[['rel_trans', 'f_asymp', 'f_iso', 'f_quar', 'beta_layer', 'viral_load', 'final']].describe().round(3).to_string())

# ---------------------------------------------------------------------------
# Offspring distribution: secondary infections per person, from the infection log
# ---------------------------------------------------------------------------
log = sim.people.infection_log
sources = np.array([d['source'] for d in log if d['source'] is not None], dtype=int)
offspring_all = np.bincount(sources, minlength=N)

ppl = sim.people
infected = np.isfinite(np.array(ppl.date_exposed))
finished = np.isfinite(np.array(ppl.date_recovered)) | np.isfinite(np.array(ppl.date_dead))   # infection over, so the offspring count is complete
offspring = offspring_all[infected & finished]
print(f'\nOffspring distribution: {len(offspring):,} people with a finished infection, mean = {offspring.mean():.2f}, max = {offspring.max()}')

# ---------------------------------------------------------------------------
# Figures (all y axes on a normal, linear scale)
# ---------------------------------------------------------------------------
# Figure 1: distribution of each term of the rel_trans line
fig, ax = plt.subplots(2, 4, figsize=(20, 8))
ax = ax.ravel()
ax[0].hist(first.rel_trans, bins=60, color='tab:blue');    ax[0].set_title(f'rel_trans (beta_dist draw, k = {K_INFECT:g})')
ax[1].hist(first.viral_load, bins=60, color='tab:orange'); ax[1].set_title('viral_load')
for a, col, c in [(ax[2], 'f_asymp', 'tab:green'), (ax[3], 'f_iso', 'tab:red'), (ax[4], 'f_quar', 'tab:purple'), (ax[5], 'beta_layer', 'tab:brown')]:
    vals = df[col].round(3).value_counts().sort_index()
    a.bar([str(x) for x in vals.index], vals.values, color=c); a.set_title(col + (' (all layers)' if col == 'beta_layer' else ''))
ax[6].hist(first.final, bins=60, color='black');            ax[6].set_title(f'final rel_trans (layer {layer0})')
ax[7].axis('off')
for a in ax[:7]:
    a.set_ylabel('infectious person-days')
plt.tight_layout(); plt.savefig(f'{OUT}_terms.png', dpi=150)

# Figure 2: mean of each term of the line over time, including the beta_dist draw (rel_trans) and beta_layer
cols = ['rel_trans', 'f_asymp', 'f_iso', 'f_quar', 'beta_layer', 'viral_load', 'final']
labels = {'rel_trans': 'rel_trans (beta_dist draw)', 'final': 'final (product of all)'}
daily = first.groupby('day')[cols].mean()
fig, ax = plt.subplots(figsize=(10, 6))
for col in cols:
    ax.plot(daily.index, daily[col], label=labels.get(col, col), lw=3 if col == 'final' else 1.8,
            ls='--' if col.startswith('f_') else '-')
ax.set_xlabel('day'); ax.set_ylabel('mean over infectious people')
ax.set_title(f'Terms of the rel_trans line over time (layer {layer0})'); ax.legend()
plt.tight_layout(); plt.savefig(f'{OUT}_over_time.png', dpi=150)

# Figure 3: the offspring distribution, number of people on a normal y axis
fig, ax = plt.subplots(figsize=(9, 5))
counts = np.bincount(offspring)
ax.bar(np.arange(len(counts)), counts, color='tab:blue')
ax.set_xlabel('secondary infections per person'); ax.set_ylabel('number of people')
ax.set_title(f'Offspring distribution (beta_dist k = {K_INFECT:g})')
plt.tight_layout(); plt.savefig(f'{OUT}_offspring.png', dpi=150)
print(f'\nSaved {OUT}_terms.png, {OUT}_over_time.png, {OUT}_offspring.png')

df.to_csv(f'{OUT}_terms.csv', index=False)    # the raw tracked values, in case you want to dig further