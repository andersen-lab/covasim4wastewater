'''
Aim: (1) keep track of the values of every term in this line of cvu.compute_trans_sus:

        rel_trans = rel_trans * inf * f_quar * f_asymp * f_iso * beta_layer * viral_load

     for every infectious person on every day, (2) keep track of the per-contact transmission probability, which in
     cvu.compute_infections is the local variable called `betas`:

        betas = beta * layer_betas * rel_trans[source] * rel_sus[target]

     (3) plot the offspring distribution (number of secondary infections caused per person), and (4) plot the reported
     cases next to the true infections.

How it works: compute_trans_sus and compute_infections are compiled by numba, so we cannot print inside them.
Instead an analyzer runs at the end of every sim.step() and recomputes the same quantities from the same arrays.

    prel_trans  : people.rel_trans (called prel_trans in step()), the person's own draw from beta_dist (neg binomial, par2 = k)
    f_asymp     : symp + ~symp * asymp_factor           (1 if symptomatic, asymp_factor otherwise)
    f_iso       : ~iso  + iso  * iso_factor[layer]      (1 if not isolated, iso_factor if isolated)
    f_quar      : ~quar + quar * quar_factor[layer]     (1 if not quarantined, quar_factor if quarantined)
    beta_layer  : sim['beta_layer'][layer]
    viral_load  : people.viral_load, a piecewise curve over the infectious period
    rel_trans   : the product of all of these = the rel_trans that compute_trans_sus returns and step() passes to compute_infections
                  (0 for people who are not infectious). On the right-hand side of the line above, rel_trans is the input, prel_trans.

    betas       : for every contact where the source is infectious and the target is susceptible, in both directions,
                  beta * layer_betas * rel_trans[source] * rel_sus[target], with rel_sus = rel_sus * sus * f_quar * (1 - sus_imm).
                  Contacts with probability 0 are dropped (nothing can be transmitted there).

Notes
 - The analyzer does NOT draw the random coin flips itself, since that would use up random numbers and change the sim.
   The coin-flip outcomes are the real transmissions, which come from sim.people.infection_log.
 - People infected on a given day are no longer 'susceptible' by the time the analyzer runs, so they are counted as
   susceptible for that day (date_exposed == today).
 - The optional haplotype fitness multiplier (evo_pars) is ignored (off by default).
 - Matches cvu.compute_infections (checked against its code): both directions of each contact, probability = beta * layer_betas * rel_trans[source] * rel_sus[target].
   This script assumes one variant (beta is taken from variant 0).
   The right-hand plot of the betas figure checks this: expected and actual new infections should be about equal.

Terms that differ by layer (beta_layer, f_iso, f_quar, rel_trans) are shown pooled over ALL layers, with each infectious person counted once per
contact they have in that layer (transmission happens per contact, so a layer with more contacts matters more). The other terms (prel_trans,
viral_load, f_asymp) belong to the person, not the layer, so they are shown once per infectious person.

The simulation is the one from 5_Build_epinow2_episewer_input_final.ipynb: the San Diego 'rank' population built from the density raster,
zip code boundaries and sewersheds, 50,000 agents, 180 days, beta = 0.012, 10 seeds, k = 0.45 for infectiousness, and symptomatic testing at 0.1 per day.
Reported cases are sim.results['new_diagnoses'], as in the notebook.

Run:  python track_rel_trans_and_offspring.py
'''
import logging
import warnings
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import covasim as cv

warnings.filterwarnings('ignore', message='.*no "population" column.*')   # harmless: with a raster the regions come from the raster
logging.getLogger('rasterio._env').setLevel(logging.ERROR)                # the repeated PROJ version message

# ---------------------------------------------------------------------------
# Settings
# ---------------------------------------------------------------------------
N          = 50000       # number of simulated people (N_AGENTS in the notebook)
N_DAYS     = 180         # length of the epidemic, in days
SEED       = 1
N_INITIAL  = 10          # agents infected on day 0 (the seeds)
BETA       = 0.012       # Covasim's beta: chance of transmission per infectious contact (the Covasim default is 0.016)
K_INFECT   = 0.45        # dispersion of infectiousness (beta_dist), the Covasim default
FRIEND_COUNT_DISPERSION = None    # k1: None = Poisson, so it is not passed to Covasim
DAILY_TEST_PROB_IF_SYMPTOMATIC = 0.1
POP_TYPE   = 'rank'      # San Diego population built from the three map files below
OUT        = f'rel_trans_offspring_k{K_INFECT:g}'

# The three map files (the same ones as the notebook and run_covasim.py). This assumes the script sits in network_rank_model/<folder>;
# otherwise set DATA_FOLDER to the folder that holds them.
DATA_FOLDER = None
if DATA_FOLDER is None:
    HERE = Path(__file__).resolve().parent if '__file__' in globals() else Path.cwd()
    DATA_FOLDER = HERE.parents[1]/'docs'/'tutorials'/'san-diego-data' if len(HERE.parents) > 1 else HERE
POPULATION_DENSITY_RASTER = Path(DATA_FOLDER)/'san_diego_county_ppp_2020.tif'
ZIP_CODE_BOUNDARIES       = Path(DATA_FOLDER)/'sandiego_zip_boundaries.geojson'
SEWERSHED_BOUNDARIES      = Path(DATA_FOLDER)/'sandiego_sewersheds.geojson'
for f in (POPULATION_DENSITY_RASTER, ZIP_CODE_BOUNDARIES, SEWERSHED_BOUNDARIES):
    if not f.exists():
        raise FileNotFoundError(f'{f} not found. Set DATA_FOLDER at the top of this script to the folder with the three map files.')


# ---------------------------------------------------------------------------
# Analyzer
# ---------------------------------------------------------------------------
class RelTransTracker(cv.Analyzer):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.chunks = []          # terms of the rel_trans line, one row per infectious person x day x layer
        self.betas_days = []      # the day of each entry of self.betas_arrays
        self.betas_arrays = []    # betas of every contact with a nonzero probability, one array per day
        self.beta_layer = {}
        self.expected = None      # per person: sum of betas over all their contacts and days as the source = expected offspring

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
        day_betas = []

        days_since = sim.t - p.date_infectious[idx]                             # days since each person became infectious (0 on their first infectious day)

        for lkey in sim.layer_keys():
            f_iso_all  = ~iso  + iso  * sim['iso_factor'][lkey]
            f_quar_all = ~quar + quar * sim['quar_factor'][lkey]
            bl = sim['beta_layer'][lkey]
            layer = p.contacts[lkey]
            p1, p2, layer_betas = layer['p1'], layer['p2'], layer['beta']
            deg = np.bincount(np.concatenate([p1, p2]), minlength=len(p))     # number of contacts of each person in this layer

            # (1) the terms of the rel_trans line, for the infectious people
            rel_trans_all = p.rel_trans * inf * f_quar_all * f_asymp_all * f_iso_all * bl * p.viral_load     # the rel_trans that compute_trans_sus returns
            self.chunks.append(pd.DataFrame(dict(
                day=sim.t, person=idx, layer=lkey, prel_trans=p.rel_trans[idx], f_asymp=f_asymp_all[idx],
                f_iso=f_iso_all[idx], f_quar=f_quar_all[idx], beta_layer=bl, viral_load=p.viral_load[idx],
                rel_trans=rel_trans_all[idx], n_contacts=deg[idx],
                days_since_infectious=days_since)))

            # (2) betas for every contact, both directions
            rel_sus_all = p.rel_sus * sus_now * f_quar_all * (1 - p.sus_imm[0, :])
            for src, tgt in ((p1, p2), (p2, p1)):
                betas = beta * layer_betas * rel_trans_all[src] * rel_sus_all[tgt]
                if self.expected is None:
                    self.expected = np.zeros(len(p))
                self.expected += np.bincount(src, weights=betas, minlength=len(p))
                day_betas.append(betas[betas > 0].astype(np.float32))

        self.betas_days.append(sim.t)
        self.betas_arrays.append(np.concatenate(day_betas) if day_betas else np.array([], dtype=np.float32))

    def table(self):
        return pd.concat(self.chunks, ignore_index=True)


# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------
def make_sim(seed, **over):
    ''' The simulation from the notebook (rank network, symptomatic testing). Anything in over replaces a Covasim parameter. '''
    testing = cv.test_prob(symp_prob=DAILY_TEST_PROB_IF_SYMPTOMATIC, asymp_prob=0.0, start_day=0)
    pars = dict(pop_type=POP_TYPE, pop_size=N, pop_infected=N_INITIAL, n_imports=0, beta=BETA, n_days=N_DAYS, rand_seed=seed, verbose=0,
                beta_dist=dict(dist='neg_binomial', par1=1.0, par2=K_INFECT, step=0.01),
                interventions=testing, density_raster=str(POPULATION_DENSITY_RASTER),
                boundary_shapefile=str(ZIP_CODE_BOUNDARIES), sewershed_file=str(SEWERSHED_BOUNDARIES))
    pars.update(over)
    return cv.Sim(**pars)


def initialize_and_run(sim):
    ''' Initialize (the workers only speed up building the network), then run '''
    extra = dict(use_workers=True)
    if FRIEND_COUNT_DISPERSION is not None:
        extra['dispersion'] = FRIEND_COUNT_DISPERSION
    sim.initialize(**extra)
    sim.run()
    return sim


tracker = RelTransTracker(label='tracker')
sim = make_sim(SEED, analyzers=tracker)
initialize_and_run(sim)
tracker = sim.get_analyzer('tracker')     # sim deep-copies the analyzer, so fetch the one that actually ran
df = tracker.table()

layer0 = sim.layer_keys()[0]
first = df[df.layer == layer0]            # one row per infectious person-day (layer-independent terms are the same in every layer)

# --- footer start
def _fmt(d, names=None):
    return ', '.join(f"{k}{' (' + names[k] + ')' if names and k in names else ''} = {v:g}" for k, v in d.items())


_vd = sim['viral_dist']
_norm = _vd['frac_time']*_vd['load_ratio'] + (1 - _vd['frac_time'])      # normalises the mean viral load to 1
_bd = sim['beta_dist']
PARAM_LINES = [
    'WHAT EACH TERM IS',
    f"beta = {BETA:g}: the global transmission scale per contact; it multiplies everything below.",
    f"prel_trans (people.rel_trans, from beta_dist): each person's own infectiousness, drawn once when they are infected from a negative binomial with mean {_bd['par1']:g} and dispersion k = {_bd['par2']:g}. Small k = a few super-spreaders and many people near 0.",
    f"viral_load (viral_dist): the first {_vd['frac_time']:.0%} of the infectious period is at {_vd['load_ratio']:g}x the load of the rest, scaled so the mean is 1 ({_vd['load_ratio']/_norm:.2f} early, {1/_norm:.2f} later).",
    f"f_asymp: 1 if symptomatic, asymp_factor = {sim['asymp_factor']:g} if asymptomatic (1 means asymptomatic people are just as infectious).",
    f"f_iso: 1 unless isolated, then iso_factor ({_fmt(sim['iso_factor'])}).   f_quar: 1 unless quarantined, then quar_factor ({_fmt(sim['quar_factor'])}).",
    f"beta_layer: a fixed multiplier for each contact layer ({_fmt(sim['beta_layer'], dict(h='household', s='school', w='work', c='community', a='all contacts'))}).",
    'rel_trans (what compute_trans_sus returns) = prel_trans x infectious (0 or 1) x f_asymp x f_iso x f_quar x beta_layer x viral_load.   Probability a contact transmits on a day = betas = beta x layer_betas x rel_trans[source] x rel_sus[target].',
]


def add_param_footer(fig, lines=PARAM_LINES, fontsize=11, top=1.0):
    ''' lay out the figure and write the description of each term underneath the plots '''
    footer = (0.26*len(lines) + 0.3)/fig.get_figheight()                  # height of the text in figure fractions
    fig.tight_layout(rect=(0, footer, 1, top))
    fig.text(0.01, 0.006, '\n'.join(lines), ha='left', va='bottom', fontsize=fontsize, linespacing=1.4)
# --- footer end

print(f'\nbeta_layer: {tracker.beta_layer}')
print(f'{len(df):,} rows (infectious person x day x layer)')
# --- helpers start
LAYER_TERMS = ['beta_layer', 'f_iso', 'f_quar', 'rel_trans']      # terms that differ by layer


def wq(vals, weights, q):
    ''' weighted quantile '''
    o = np.argsort(vals)
    v, w = vals[o], weights[o]
    c = np.cumsum(w)
    if c[-1] <= 0:
        return np.nan
    return np.interp(q, (c - 0.5*w)/c[-1], v)


def daily_summary(frame, col, weighted):
    ''' per day: 2.5th percentile, median, 97.5th percentile and mean, optionally weighted by each row's number of contacts '''
    rows = []
    for day, g in frame.groupby('day'):
        v = g[col].to_numpy(float)
        w = g['n_contacts'].to_numpy(float) if weighted else np.ones(len(v))
        rows.append((day, wq(v, w, 0.025), wq(v, w, 0.5), wq(v, w, 0.975), np.average(v, weights=w) if w.sum() > 0 else np.nan))
    return pd.DataFrame(rows, columns=['day', 'lo', 'med', 'hi', 'mean']).set_index('day')


def plot_viral_load_by_day(a, frame, max_days=25, min_n=20):
    ''' viral_load on each day since a person became infectious (day 0 = their first infectious day). Each bar is the share of the
    infectious person-days on that day that are at the high level (the rest are at the low level). '''
    high, low = _vd['load_ratio']/_norm, 1/_norm
    cut = (high + low)/2
    d = frame.days_since_infectious.round().astype(int)
    n = frame.groupby(d).size()
    share_high = (frame.viral_load > cut).groupby(d).mean()*100
    keep = n.index[(n >= min_n) & (n.index >= 0) & (n.index <= max_days)]
    a.bar(keep, share_high[keep], color='tab:orange')
    a.set_xticks(keep[::2] if len(keep) > 14 else keep)
    a.set_xlabel('days since the person became infectious (0 = first infectious day)')
    a.set_ylabel(f'% of infectious person-days at the high level ({high:.2f})')
    a.set_title(f'viral_load: share at the high level on each day\nof the infectious period (the low level is {low:.2f})')
    a.set_ylim(0, 105)
# --- helpers end

print('\nSummary of the person-level terms over infectious person-days:')
print(first[['prel_trans', 'f_asymp', 'viral_load']].describe().round(3).to_string())
fv, fw = df.rel_trans.to_numpy(float), df.n_contacts.to_numpy(float)
print(f'\nrel_trans (the product) over all layers, weighted by number of contacts: median {wq(fv, fw, 0.5):.2f}, '
      f'95% range {wq(fv, fw, 0.025):.2f} to {wq(fv, fw, 0.975):.2f}, mean {np.average(fv, weights=fw):.2f}')

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
fig, ax = plt.subplots(2, 4, figsize=(20, 11))
ax = ax.ravel()
ax[0].hist(first.prel_trans, bins=60, color='tab:blue');    ax[0].set_title(f'prel_trans (people.rel_trans, beta_dist draw, k = {K_INFECT:g})')
ax[1].hist(first.viral_load, bins=60, color='tab:orange'); ax[1].set_title('viral_load')
for a, col, c in [(ax[2], 'f_asymp', 'tab:green'), (ax[3], 'f_iso', 'tab:red'), (ax[4], 'f_quar', 'tab:purple'), (ax[5], 'beta_layer', 'tab:brown')]:
    vals = df[col].round(3).value_counts().sort_index()
    a.bar([str(x) for x in vals.index], vals.values, color=c); a.set_title(col + (' (all layers)' if col == 'beta_layer' else ''))
ax[6].hist(df.rel_trans, bins=60, weights=df.n_contacts, color='black'); ax[6].set_title('rel_trans (product of all terms; all layers, weighted by contacts)')
ax[7].axis('off')
for a in ax[:7]:
    a.set_ylabel('infectious person-days')
ax[6].set_ylabel('infectious person-days x contacts')
add_param_footer(fig); plt.savefig(f'{OUT}_terms.png', dpi=150)


# Figure 2: each term over time among infectious people (solid = median, shaded = 95% range (2.5th to 97.5th percentile), dashed = mean;
# the median is shown because rel_trans is heavy-tailed, so the mean is pulled up by a few people). viral_load is shown over the
# infectious period instead of over days, since that is what its piecewise shape depends on.
def varies(col):
    return df[col].max() - df[col].min() > 1e-9
asymp_varies, iso_varies, quar_varies = varies('f_asymp'), varies('f_iso'), varies('f_quar')
multi_layer = len(sim.layer_keys()) > 1
print(f'\nTerms that vary in this run: f_asymp {asymp_varies}, f_iso {iso_varies}, f_quar {quar_varies}; layers: {sim.layer_keys()}')

base_bl = dict(sim['beta_layer'])
_bl_names = dict(h='household', s='school', w='work', c='community', a='all contacts')
fixed_text = {                                       # terms that are a fixed value are written out instead of plotted
    'f_asymp': f"f_asymp\n1 if symptomatic,\n{sim['asymp_factor']:g} if asymptomatic\n(same for everyone)",
    'beta_layer': "beta_layer\nfixed for each layer:\n" + '\n'.join(f"{k} ({_bl_names.get(k, k)}) = {v:g}" for k, v in base_bl.items()),
}
cols = [('prel_trans', 'prel_trans (people.rel_trans,\nbeta_dist draw)'), ('viral_load', 'viral_load'), ('f_asymp', 'f_asymp' if asymp_varies else None)]
if iso_varies:
    cols.append(('f_iso', 'f_iso'))
if quar_varies:
    cols.append(('f_quar', 'f_quar'))
if multi_layer:
    cols.append(('beta_layer', None))
cols.append(('rel_trans', 'rel_trans (product of all terms)'))

fig, ax = plt.subplots(1, len(cols), figsize=(5.4*len(cols), 7), squeeze=False)
for j, (col, title) in enumerate(cols):
    a = ax[0, j]
    if col == 'viral_load':
        plot_viral_load_by_day(a, first)
    elif title is None:
        a.axis('off')
        a.text(0.5, 0.5, fixed_text[col], ha='center', va='center', fontsize=13, transform=a.transAxes)
    else:
        by_layer = col in LAYER_TERMS
        sm = daily_summary(df if by_layer else first, col, weighted=by_layer)     # layer terms: all layers, weighted by contacts
        a.fill_between(sm.index, sm.lo, sm.hi, color='tab:blue', alpha=0.25, label='95% range')
        a.plot(sm.med, color='tab:blue', lw=2.5, label='median')
        a.plot(sm['mean'], color='black', lw=1.5, ls='--', label='mean')
        a.set_title(title + ('\n(all layers, weighted by contacts)' if by_layer else '\n(per infectious person)')); a.set_xlabel('day')
ax[0, 0].legend()
fig.suptitle(f'Terms of the rel_trans line among infectious people (k = {K_INFECT:g})', fontsize=15)
add_param_footer(fig, top=0.93); plt.savefig(f'{OUT}_over_time.png', dpi=150)

# Figure 2b: one infected person -- how their viral_load (and so their rel_trans) changes day by day
person_days = first.groupby('person').size()                                      # number of infectious days of each person
pid = (person_days - person_days.median()).abs().sort_values().index[0]           # a person whose infectious period is of typical length
one = first[first.person == pid].sort_values('day')
d1 = one.days_since_infectious.round().astype(int)
high_level, low_level = _vd['load_ratio']/_norm, 1/_norm
isolated_days = (one.f_iso < 1).to_numpy()
print(f'\nOne person (person {pid}): infectious for {len(one)} days, from day {one.day.min()} to {one.day.max()}, prel_trans = {one.prel_trans.iloc[0]:.2f}')
fig, ax = plt.subplots(1, 2, figsize=(15, 8))
ax[0].plot(d1, one.viral_load, 'o-', color='tab:orange', lw=2, ms=9)
ax[0].axhline(high_level, color='gray', ls=':'); ax[0].axhline(low_level, color='gray', ls=':')
ax[0].set_ylim(0, high_level*1.25); ax[0].set_xticks(d1)
ax[0].set_xlabel('days since this person became infectious (0 = first infectious day)'); ax[0].set_ylabel('viral_load')
ax[0].set_title('viral_load on each day')
ax[1].plot(d1, one.rel_trans, 'o-', color='black', lw=2, ms=9, label='rel_trans')
if isolated_days.any():
    ax[1].plot(d1[isolated_days], one.rel_trans[isolated_days], 'o', color='tab:red', ms=15, mfc='none', mew=2, label='isolated that day (f_iso < 1)')
ax[1].set_ylim(bottom=0); ax[1].set_xticks(d1)
ax[1].set_xlabel('days since this person became infectious (0 = first infectious day)'); ax[1].set_ylabel('rel_trans')
ax[1].set_title('rel_trans on each day'); ax[1].legend()
fig.suptitle(f'One infected person (person {pid}): infectious for {len(one)} days, prel_trans = {one.prel_trans.iloc[0]:.2f}', fontsize=15)
ONE_LINES = [
    'WHAT IS PLOTTED',
    'Left: this person\'s viral_load on each day of their infectious period.',
    'Right: their rel_trans on the same days (the number that goes into betas); a red circle marks a day they were isolated.',
    'rel_trans = prel_trans x f_asymp x f_iso x f_quar x beta_layer x viral_load.',
    'For one person prel_trans, f_asymp and beta_layer stay the same, so rel_trans changes only through viral_load and f_iso.',
]
add_param_footer(fig, lines=ONE_LINES, fontsize=12, top=0.93); plt.savefig(f'{OUT}_one_person.png', dpi=150)

# Figure 2c: the same person, every term of the rel_trans line, day by day
terms = [('prel_trans', "prel_trans (this person's beta_dist draw)"), ('f_asymp', 'f_asymp'), ('f_iso', 'f_iso'),
         ('f_quar', 'f_quar'), ('beta_layer', 'beta_layer'), ('viral_load', 'viral_load')]
term_colors = ['tab:blue', 'tab:green', 'tab:red', 'tab:purple', 'tab:brown', 'tab:orange']
fig = plt.figure(figsize=(16, 15))
gs = fig.add_gridspec(3, 3, height_ratios=[1, 1, 1.15])
for k, ((col, title), c) in enumerate(zip(terms, term_colors)):
    a = fig.add_subplot(gs[k // 3, k % 3])
    a.plot(d1, one[col], 'o-', color=c, lw=2, ms=8)
    if col == 'prel_trans':
        a.axhline(1.0, color='gray', ls=':'); a.text(d1.max(), 1.0, ' population mean = 1', ha='right', va='bottom', color='gray', fontsize=10)
        a.set_ylim(0, max(1.4, one[col].max()*1.4))
    elif col == 'viral_load':
        a.set_ylim(0, high_level*1.25)
    elif col == 'beta_layer':
        a.set_ylim(0, max(1.0, one[col].max()*1.3))
    else:
        a.set_ylim(0, 1.15)
    a.set_title(title); a.set_xticks(d1); a.set_xlabel('days since this person became infectious')
a = fig.add_subplot(gs[2, :])
a.axis('off')
col_labels = ['day', 'prel_trans', 'inf', 'f_asymp', 'f_iso', 'f_quar', 'beta_layer', 'viral_load', '= rel_trans']
rows = [[int(dd), f'{r.prel_trans:.4f}', '1', f'{r.f_asymp:.2f}', f'{r.f_iso:.2f}', f'{r.f_quar:.2f}', f'{r.beta_layer:.2f}', f'{r.viral_load:.3f}', f'{r.rel_trans:.4f}']
        for dd, r in zip(d1, one.itertuples())]
tbl = a.table(cellText=rows, colLabels=col_labels, loc='center', cellLoc='center')
tbl.auto_set_font_size(False); tbl.set_fontsize(13); tbl.scale(1, 1.8)
for i, v in enumerate(one.f_iso.to_numpy()):
    if v < 1:
        tbl[(i + 1, 4)].set_facecolor('#f8c9c9')                             # days this person was isolated
for i in range(len(rows) + 1):
    tbl[(i, 8)].set_text_props(fontweight='bold')
product_check = (one.prel_trans*one.f_asymp*one.f_iso*one.f_quar*one.beta_layer*one.viral_load - one.rel_trans).abs().max()
print(f'Check for person {pid}: the product of the terms differs from the recorded rel_trans by at most {product_check:.2e}')
fig.suptitle(f'Every term of the rel_trans line for person {pid}, day by day', fontsize=16)
TERM_LINES = [
    'THE TERMS FOR THIS PERSON',
    'rel_trans = prel_trans x inf x f_asymp x f_iso x f_quar x beta_layer x viral_load      (the line in cvu.compute_trans_sus; multiply across a row of the table)',
    "prel_trans: this person's own beta_dist draw, the same every day.     inf: 1 on every day shown, since only days they are infectious are listed.",
    'f_asymp: 1 if symptomatic, asymp_factor otherwise.     f_iso: 1, or iso_factor on days they are isolated (shaded red).     f_quar: 1, or quar_factor on days quarantined.',
    'beta_layer: the fixed multiplier for the contact layer.     viral_load: the two-level curve over the infectious period.',
]
add_param_footer(fig, lines=TERM_LINES, fontsize=12, top=0.95); plt.savefig(f'{OUT}_one_person_terms.png', dpi=150)

# Reported cases and true infections (used in the betas figure and in its own figure)
true_inf = np.array(sim.results['new_infections'].values)                 # everyone newly infected that day
reported = np.array(sim.results['new_diagnoses'].values)                  # people diagnosed that day: the reported cases (as in the notebook)
cum_true, cum_rep = np.cumsum(true_inf), np.cumsum(reported)
ascertainment = np.divide(cum_rep, cum_true, out=np.full(len(cum_true), np.nan), where=cum_true > 0)
day_axis = np.arange(len(true_inf))
peak_true, peak_rep = int(day_axis[true_inf.argmax()]), int(day_axis[reported.argmax()])
print(f'\nReported cases: {cum_rep[-1]:,.0f} of {cum_true[-1]:,.0f} infections ({ascertainment[-1]:.0%}). '
      f'Infections peak on day {peak_true}, reported cases on day {peak_rep}.')

# Figure 3: betas = beta * layer_betas * rel_trans[source] * rel_sus[target], the probability each contact transmits that day
all_betas = np.concatenate(tracker.betas_arrays)
days = np.array(tracker.betas_days)
lo  = np.array([np.percentile(a, 2.5) if len(a) else np.nan for a in tracker.betas_arrays])
med = np.array([np.median(a) if len(a) else np.nan for a in tracker.betas_arrays])
hi  = np.array([np.percentile(a, 97.5) if len(a) else np.nan for a in tracker.betas_arrays])
expected = np.array([a.sum() for a in tracker.betas_arrays])                       # expected new infections = sum of the probabilities
actual = np.bincount([d['date'] for d in log if d['source'] is not None], minlength=N_DAYS + 1)[days]

print(f'\nbetas: {len(all_betas):,} contact-directions with a nonzero probability, median = {np.median(all_betas):.4f}, '
      f'95% range = {np.percentile(all_betas, 2.5):.4f} to {np.percentile(all_betas, 97.5):.4f}, max = {all_betas.max():.3f}')
print(f'expected infections (sum of betas) = {expected.sum():.0f}, actual infections in the log = {actual.sum()}')

results_new = np.array(sim.results['new_infections'].values)[days]           # the sim's own count of new infections each day
print(f'{(all_betas > 1).sum():,} of {len(all_betas):,} contact-directions have betas above 1 (these transmit for certain). '
      f'sim.results new_infections total = {results_new.sum():.0f}')


def add_betas_footer(fig):
    ''' write a clearly laid out explanation of betas under the plots '''
    blocks = [   # (text, font size, weight, family)
        ('Where betas comes from  (the local variable `betas` in cvu.compute_infections)', 15, 'bold', None),
        ('Every day, for every contact between an infectious person (source) and a susceptible person (target), in both directions:', 12, 'normal', None),
        ('betas  =  beta  ×  layer_betas  ×  rel_trans[source]  ×  rel_sus[target]', 16, 'bold', 'monospace'),
        ('Then one coin flip per contact:  the contact transmits if  random() < betas', 12, 'normal', None),
        ('', 6, 'normal', None),
        ('where', 12, 'bold', None),
        (f'   •  beta = {BETA:g}:  the global transmission scale', 12, 'normal', None),
        ("   •  layer_betas:  the contact's own weight within its layer", 12, 'normal', None),
        ('   •  rel_trans[source]:  how infectious the source is  =  prel_trans × f_asymp × f_iso × f_quar × beta_layer × viral_load   (what compute_trans_sus returns)', 12, 'normal', None),
        ('   •  rel_sus[target]:  how susceptible the target is  =  rel_sus × susceptible (0 or 1) × f_quar × (1 − immunity)', 12, 'normal', None),
        ('   •  a betas value above 1 just means that contact transmits for certain', 12, 'normal', None),
        ('', 6, 'normal', None),
        ('How to read the middle plot', 12, 'bold', None),
        ('   •  red = expected new infections per day  =  the sum of betas over all contacts', 12, 'normal', None),
        ('   •  black and green = the new infections the sim actually recorded that day (what the coin flips produced)', 12, 'normal', None),
        ('', 6, 'normal', None),
        ('How to read the right-hand plot', 12, 'bold', None),
        ("   •  black = true infections (sim.results['new_infections']);  red = reported cases (sim.results['new_diagnoses'], people diagnosed that day)", 12, 'normal', None),
    ]
    line_in = lambda size: size*1.55/72                                   # height of one line, in inches
    total_in = sum(line_in(size) for _, size, _, _ in blocks) + 0.3
    footer = total_in/fig.get_figheight()                                  # height of the text, as a fraction of the figure
    fig.tight_layout(rect=(0, footer, 1, 1))
    y = footer - 0.1/fig.get_figheight()
    for text, size, weight, family in blocks:
        if text:
            fig.text(0.012, y, text, ha='left', va='top', fontsize=size, fontweight=weight, family=family or 'DejaVu Sans')
        y -= line_in(size)/fig.get_figheight()


fig, ax = plt.subplots(1, 3, figsize=(20, 11))
ax[0].fill_between(days, lo, hi, color='tab:blue', alpha=0.25, label='95% range')
ax[0].plot(days, med, color='tab:blue', lw=2.5, label='median')
ax[0].set_xlabel('day'); ax[0].set_ylabel('betas (probability a contact transmits on that day)'); ax[0].set_title('betas over time'); ax[0].legend()
ax[1].plot(days, expected, color='tab:red', lw=2.5, label='expected (sum of betas)')
ax[1].plot(days, actual, color='black', lw=2, ls='--', label='actual (infection log)')
ax[1].plot(days, results_new, color='tab:green', lw=1.5, ls=':', label='actual (sim.results new_infections)')
ax[1].set_xlabel('day'); ax[1].set_ylabel('new infections per day'); ax[1].set_title('Expected vs actual new infections'); ax[1].legend()
ax[2].plot(day_axis, true_inf, color='black', lw=2.5, label='true infections')
ax[2].plot(day_axis, reported, color='tab:red', lw=2.5, label='reported cases')
ax[2].axvline(peak_true, color='black', ls=':', lw=1); ax[2].axvline(peak_rep, color='tab:red', ls=':', lw=1)
ax[2].set_xlabel('day'); ax[2].set_ylabel('per day'); ax[2].set_title(f'True infections and reported cases\n(peaks: day {peak_true} infections, day {peak_rep} reported)'); ax[2].legend()
add_betas_footer(fig); plt.savefig(f'{OUT}_betas.png', dpi=150)

# Figure 4: how prel_trans (each person's beta_dist draw) feeds into the epidemic -- all from this one run
done = infected & finished                                          # people whose infection is over, so their offspring count is complete
prel = np.array(ppl.rel_trans)                                      # each person's beta_dist draw
exp_off = tracker.expected                                          # expected offspring from the betas
grp_df = pd.DataFrame(dict(prel=prel[done], actual=offspring_all[done], expected=exp_off[done]))
grp_df['decile'] = pd.qcut(grp_df.prel, 10, labels=False, duplicates='drop')
by_dec = grp_df.groupby('decile').mean()

order = np.argsort(-grp_df.prel.to_numpy())                         # highest prel_trans first
cum_share = np.cumsum(grp_df.actual.to_numpy()[order])/grp_df.actual.sum()
cum_people = np.arange(1, len(order) + 1)/len(order)
top10 = cum_share[int(0.1*len(order)) - 1]
top20 = cum_share[int(0.2*len(order)) - 1]

log_src = np.array([-1 if d['source'] is None else d['source'] for d in log], dtype=int)
q50, q90 = np.quantile(prel[infected], [0.5, 0.9])
src_prel = prel[np.where(log_src >= 0, log_src, 0)]
group_names = ['lowest 50%\nof prel_trans', '50-90%\nof prel_trans', 'top 10%\nof prel_trans']
infector_masks = [(log_src >= 0) & (src_prel <= q50), (log_src >= 0) & (src_prel > q50) & (src_prel <= q90), (log_src >= 0) & (src_prel > q90)]
infections_by_group = np.array([m.sum() for m in infector_masks])                    # infections caused by each group
prel_inf = prel[infected]
people_by_group = np.array([(prel_inf <= q50).sum(), ((prel_inf > q50) & (prel_inf <= q90)).sum(), (prel_inf > q90).sum()])   # infected people in each group
share_people = people_by_group/people_by_group.sum()
share_infections = infections_by_group/infections_by_group.sum()
print(f'\nHow prel_trans feeds into the epidemic: the top 10% of people by prel_trans caused {top10:.0%} of all secondary infections, the top 20% caused {top20:.0%}.')
for name, sp, si in zip(group_names, share_people, share_infections):
    print(f'   {name.replace(chr(10), " ")}: {sp:.0%} of infected people, {si:.0%} of the infections they caused')

fig, ax = plt.subplots(1, 3, figsize=(20, 9))
x = np.arange(len(by_dec))
ax[0].bar(x, by_dec.actual, color='tab:blue', label='actual (infection log)')
ax[0].plot(x, by_dec.expected, 'o', color='tab:red', ms=8, label='expected (sum of betas)')
ax[0].set_xticks(x); ax[0].set_xticklabels([f'{v:.2f}' for v in by_dec.prel], rotation=45)
ax[0].set_xlabel('Infected people grouped into tenths by prel_trans, lowest to highest\n(the number under each bar is that group\'s average prel_trans)')
ax[0].set_ylabel('secondary infections per person')
ax[0].set_title('More infectious people infect more people'); ax[0].legend()
ax[1].plot(cum_people, cum_share, color='tab:blue', lw=2.5, label='people sorted from highest to lowest prel_trans')
ax[1].plot([0, 1], [0, 1], color='gray', ls='--', label='if everyone infected equally')
ax[1].set_xlabel('share of people (highest prel_trans first)'); ax[1].set_ylabel('share of all infections they caused')
ax[1].set_title(f'The top 10% caused {top10:.0%}, the top 20% caused {top20:.0%}'); ax[1].legend()
xg = np.arange(3)
b1 = ax[2].bar(xg - 0.2, 100*share_people, width=0.4, color='tab:gray', label='share of infected people')
b2 = ax[2].bar(xg + 0.2, 100*share_infections, width=0.4, color='tab:red', label='share of all infections they caused')
for bars in (b1, b2):
    for rect in bars:
        ax[2].annotate(f'{rect.get_height():.0f}%', (rect.get_x() + rect.get_width()/2, rect.get_height()), ha='center', va='bottom', fontsize=12)
ax[2].set_xticks(xg); ax[2].set_xticklabels(group_names)
ax[2].set_ylabel('percent'); ax[2].set_title('Who caused the infections?'); ax[2].legend(loc='upper left')
ax[2].set_ylim(0, max(100*share_people.max(), 100*share_infections.max())*1.15)
HOW_LINES = [
    'HOW prel_trans FEEDS INTO THE EPIDEMIC',
    'prel_trans (people.rel_trans, from beta_dist)  →  × f_asymp × f_iso × f_quar × beta_layer × viral_load  =  rel_trans  (per person, per day)',
    '→  betas = beta × layer_betas × rel_trans[source] × rel_sus[target]  (per contact, per day)  →  coin flip  →  new infections',
    'Left: people with a higher prel_trans infect more people (blue = actual, red = expected from betas).   Middle: how concentrated transmission is among the highest-prel_trans people.',
    'Right: infected people split into three groups by prel_trans: each group\'s share of the infected people (grey) next to its share of all the infections that were caused (red).',
]
add_param_footer(fig, lines=HOW_LINES, fontsize=12); plt.savefig(f'{OUT}_rel_trans_to_epidemic.png', dpi=150)

# Figure 5: reported cases and true infections
CASES_LINES = [
    'WHAT IS PLOTTED',
    "true infections = sim.results['new_infections']: everyone who was newly infected that day, whether or not anyone knows.",
    f"reported cases = sim.results['new_diagnoses']: people diagnosed that day. Only symptomatic people are tested ({DAILY_TEST_PROB_IF_SYMPTOMATIC:g} per day), so most infections are never reported, and reports come after infection.",
]
fig, ax = plt.subplots(1, 2, figsize=(18, 8))
ax[0].plot(day_axis, true_inf, color='black', lw=2.5, label='true infections')
ax[0].plot(day_axis, reported, color='tab:red', lw=2.5, label='reported cases')
ax[0].axvline(peak_true, color='black', ls=':', lw=1); ax[0].axvline(peak_rep, color='tab:red', ls=':', lw=1)
ax[0].set_xlabel('day'); ax[0].set_ylabel('per day'); ax[0].set_title(f'Per day (peaks: day {peak_true} infections, day {peak_rep} reported)'); ax[0].legend()
ax[1].plot(day_axis, cum_true, color='black', lw=2.5, label='true infections')
ax[1].plot(day_axis, cum_rep, color='tab:red', lw=2.5, label='reported cases')
ax[1].set_xlabel('day'); ax[1].set_ylabel('cumulative'); ax[1].set_title(f'Cumulative: {cum_rep[-1]:,.0f} reported of {cum_true[-1]:,.0f} infections ({ascertainment[-1]:.0%})'); ax[1].legend()
add_param_footer(fig, lines=CASES_LINES, fontsize=12); plt.savefig(f'{OUT}_cases_vs_infections.png', dpi=150)

# Figure 6: the offspring distribution, number of people on a normal y axis
fig, ax = plt.subplots(figsize=(9, 5))
counts = np.bincount(offspring)
ax.bar(np.arange(len(counts)), counts, color='tab:blue')
ax.set_xlabel('secondary infections per person'); ax.set_ylabel('number of people')
ax.set_title(f'Offspring distribution (beta_dist k = {K_INFECT:g})')
plt.tight_layout(); plt.savefig(f'{OUT}_offspring.png', dpi=150)
print(f'\nSaved {OUT}_terms.png, {OUT}_over_time.png, {OUT}_one_person.png, {OUT}_one_person_terms.png, {OUT}_betas.png, {OUT}_rel_trans_to_epidemic.png, {OUT}_cases_vs_infections.png, {OUT}_offspring.png')

df.to_csv(f'{OUT}_terms.csv', index=False)    # the raw tracked values, in case you want to dig further