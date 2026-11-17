'''
Aim: (1) keep track of the values of every term in this line of cvu.compute_trans_sus:

        rel_trans = rel_trans * inf * f_quar * f_asymp * f_iso * beta_layer * viral_load

     for every infectious person on every day, (2) keep track of the per-contact transmission probability

        inf_probs = beta * betas * rel_trans[p1] * rel_sus[p2]

     and (3) plot the offspring distribution (number of secondary infections caused per person).

How it works: compute_trans_sus and compute_infections are compiled by numba, so we cannot print inside them.
Instead an analyzer runs at the end of every sim.step() and recomputes the same quantities from the same arrays.

    rel_trans   : people.rel_trans, the person's own draw from beta_dist (neg binomial, par2 = k)
    f_asymp     : symp + ~symp * asymp_factor           (1 if symptomatic, asymp_factor otherwise)
    f_iso       : ~iso  + iso  * iso_factor[layer]      (1 if not isolated, iso_factor if isolated)
    f_quar      : ~quar + quar * quar_factor[layer]     (1 if not quarantined, quar_factor if quarantined)
    beta_layer  : sim['beta_layer'][layer]
    viral_load  : people.viral_load, from the viral load curve over the infection
    final       : the product of all of these (what compute_trans_sus returns for an infectious person)

    inf_probs   : for every contact where the source is infectious and the target is susceptible, in both directions,
                  beta * betas * final[source] * rel_sus[target], with rel_sus = rel_sus * sus * f_quar * (1 - sus_imm).
                  Contacts with probability 0 are dropped (nothing can be transmitted there).

Notes
 - The analyzer does NOT draw the random coin flips itself, since that would use up random numbers and change the sim.
   The coin-flip outcomes are the real transmissions, which come from sim.people.infection_log.
 - People infected on a given day are no longer 'susceptible' by the time the analyzer runs, so they are counted as
   susceptible for that day (date_exposed == today).
 - The optional haplotype fitness multiplier (evo_pars) is ignored (off by default).
 - Assumes compute_infections uses both directions of each contact (p1->p2 and p2->p1) and that there is one variant.
   Panel (c) of the inf_probs figure checks this: expected and actual new infections should be about equal.

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
# Analyzer
# ---------------------------------------------------------------------------
class RelTransTracker(cv.Analyzer):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.chunks = []          # terms of the rel_trans line, one row per infectious person x day x layer
        self.prob_days = []       # the day of each entry of self.prob_arrays
        self.prob_arrays = []     # inf_probs of every contact with a nonzero probability, one array per day
        self.beta_layer = {}

    def initialize(self, sim):
        super().initialize(sim)
        self.beta_layer = {k: float(v) for k, v in sim['beta_layer'].items()}

    def apply(self, sim):
        p = sim.people
        inf = p.infectious
        idx = np.flatnonzero(inf)
        if len(idx) == 0:
            return
        symp, iso, quar = p.symptomatic, p.isolated, p.quarantined
        f_asymp_all = symp + ~symp * sim['asymp_factor']
        sus_now = p.susceptible | (p.date_exposed == sim.t)                    # susceptible when today's coins were flipped
        variant_label = sim['variant_map'][0]
        beta = sim['beta'] * sim['rel_beta'] * sim['variant_pars'][variant_label]['rel_beta']
        day_probs = []

        for lkey in sim.layer_keys():
            f_iso_all  = ~iso  + iso  * sim['iso_factor'][lkey]
            f_quar_all = ~quar + quar * sim['quar_factor'][lkey]
            bl = sim['beta_layer'][lkey]

            # (1) the terms of the rel_trans line, for the infectious people
            final_all = p.rel_trans * inf * f_quar_all * f_asymp_all * f_iso_all * bl * p.viral_load
            self.chunks.append(pd.DataFrame(dict(
                day=sim.t, person=idx, layer=lkey, rel_trans=p.rel_trans[idx], f_asymp=f_asymp_all[idx],
                f_iso=f_iso_all[idx], f_quar=f_quar_all[idx], beta_layer=bl, viral_load=p.viral_load[idx],
                final=final_all[idx])))

            # (2) inf_probs for every contact, both directions
            rel_sus_all = p.rel_sus * sus_now * f_quar_all * (1 - p.sus_imm[0, :])
            layer = p.contacts[lkey]
            p1, p2, betas = layer['p1'], layer['p2'], layer['beta']
            for src, tgt in ((p1, p2), (p2, p1)):
                probs = beta * betas * final_all[src] * rel_sus_all[tgt]
                day_probs.append(probs[probs > 0].astype(np.float32))

        self.prob_days.append(sim.t)
        self.prob_arrays.append(np.concatenate(day_probs) if day_probs else np.array([], dtype=np.float32))

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

# Figure 2: each term over time. Solid line = median, shaded = 95% range (2.5th to 97.5th percentile) across infectious
# people, dashed = mean. The median is shown because rel_trans is heavy-tailed, so the mean is pulled up by a few people.
panels = [('rel_trans', 'rel_trans (beta_dist draw)'), ('viral_load', 'viral_load'), ('f_asymp', 'f_asymp'), ('f_iso', 'f_iso'),
          ('f_quar', 'f_quar'), ('beta_layer', 'beta_layer'), ('final', 'final (product of all)')]
fig, ax = plt.subplots(2, 4, figsize=(20, 8), sharex=True)
ax = ax.ravel()
for a, (col, title) in zip(ax, panels):
    g = first.groupby('day')[col]
    a.fill_between(g.quantile(0.5).index, g.quantile(0.025), g.quantile(0.975), color='tab:blue', alpha=0.25, label='95% range')
    a.plot(g.median(), color='tab:blue', lw=2.5, label='median')
    a.plot(g.mean(), color='black', lw=1.5, ls='--', label='mean')
    a.set_title(title); a.set_xlabel('day')
ax[0].legend()
ax[7].axis('off')
fig.suptitle(f'Terms of the rel_trans line over time among infectious people (layer {layer0})')
plt.tight_layout(); plt.savefig(f'{OUT}_over_time.png', dpi=150)

# Figure 3: inf_probs = beta * betas * rel_trans[p1] * rel_sus[p2], the probability each contact transmits that day
all_probs = np.concatenate(tracker.prob_arrays)
days = np.array(tracker.prob_days)
lo  = np.array([np.percentile(a, 2.5) if len(a) else np.nan for a in tracker.prob_arrays])
med = np.array([np.median(a) if len(a) else np.nan for a in tracker.prob_arrays])
hi  = np.array([np.percentile(a, 97.5) if len(a) else np.nan for a in tracker.prob_arrays])
expected = np.array([a.sum() for a in tracker.prob_arrays])                       # expected new infections = sum of the probabilities
actual = np.bincount([d['date'] for d in log if d['source'] is not None], minlength=N_DAYS + 1)[days]

print(f'\ninf_probs: {len(all_probs):,} contact-directions with a nonzero probability, median = {np.median(all_probs):.4f}, '
      f'95% range = {np.percentile(all_probs, 2.5):.4f} to {np.percentile(all_probs, 97.5):.4f}, max = {all_probs.max():.3f}')
print(f'expected infections (sum of inf_probs) = {expected.sum():.0f}, actual infections in the log = {actual.sum()}')

fig, ax = plt.subplots(1, 3, figsize=(20, 5.5))
ax[0].hist(all_probs, bins=80, color='tab:blue')
ax[0].set_xlabel('inf_probs (probability a contact transmits on that day)'); ax[0].set_ylabel('number of contact-days'); ax[0].set_title('Distribution of inf_probs')
ax[1].fill_between(days, lo, hi, color='tab:blue', alpha=0.25, label='95% range')
ax[1].plot(days, med, color='tab:blue', lw=2.5, label='median')
ax[1].set_xlabel('day'); ax[1].set_ylabel('inf_probs'); ax[1].set_title('inf_probs over time'); ax[1].legend()
ax[2].plot(days, expected, color='tab:red', lw=2.5, label='expected (sum of inf_probs)')
ax[2].plot(days, actual, color='black', lw=2, ls='--', label='actual (infection log)')
ax[2].set_xlabel('day'); ax[2].set_ylabel('new infections'); ax[2].set_title('Expected vs actual new infections'); ax[2].legend()
plt.tight_layout(); plt.savefig(f'{OUT}_inf_probs.png', dpi=150)

# Figure 4: the offspring distribution, number of people on a normal y axis
fig, ax = plt.subplots(figsize=(9, 5))
counts = np.bincount(offspring)
ax.bar(np.arange(len(counts)), counts, color='tab:blue')
ax.set_xlabel('secondary infections per person'); ax.set_ylabel('number of people')
ax.set_title(f'Offspring distribution (beta_dist k = {K_INFECT:g})')
plt.tight_layout(); plt.savefig(f'{OUT}_offspring.png', dpi=150)
print(f'\nSaved {OUT}_terms.png, {OUT}_over_time.png, {OUT}_inf_probs.png, {OUT}_offspring.png')

df.to_csv(f'{OUT}_terms.csv', index=False)    # the raw tracked values, in case you want to dig further