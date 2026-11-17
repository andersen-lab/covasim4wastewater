'''
Aim: how do prel_trans, rel_trans and betas change over an epidemic when the dispersion of infectiousness (k2) is changed?

k2 is the dispersion of beta_dist, a negative binomial with mean 1 that gives every infected person their own infectiousness.
    small k2 (0.1)  : a few people are very infectious and most are barely infectious (super-spreaders)
    k2 = 0.45       : the Covasim default
    k2 = 10         : most people are close to the average
    Poisson         : k2 so large (1e6) that everyone is almost exactly 1

The simulation is the one from 5_Build_epinow2_episewer_input_final.ipynb (San Diego 'rank' population, symptomatic testing),
the same one as track_rel_trans_and_offspring.py. Only k2 changes between columns. Every run is tracked with an analyzer that recomputes,
for every infectious person on every day, the terms Covasim multiplies together:

    prel_trans : people.rel_trans, the person's own beta_dist draw
    rel_trans  : prel_trans * inf * f_asymp * f_iso * f_quar * beta_layer * viral_load   (what cvu.compute_trans_sus returns)
    betas      : beta * layer_betas * rel_trans[source] * rel_sus[target]                (what cvu.compute_infections uses for each contact)

Saves two figures:
    k_comparison_grid.png    : one column per k2; rows = new infections per day, prel_trans, rel_trans, betas over the epidemic (with the 95% range)
    k_comparison_grid_mean_median.png : the same grid with only the median and the mean lines (no shading), so the y axis is not stretched by the tail
    k_comparison_overlay.png : the median, mean and 97.5th percentile of each quantity over the epidemic, all k2 on the same axes

Each k2 is run N_RUNS times with the same seeds as the other k2 values. A run that infects under TAKE_OFF_PCT % of the agents counts as
died out; it is shown in the epidemic curves but left out of the rows below, which average the runs that took off.

Run:  python k_comparison_rel_trans_betas.py        (try N_RUNS = 1 first)
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
N          = 20000       # agents (the notebook uses 50000; smaller is faster)
N_DAYS     = 180
SEED       = 1           # first seed (run r uses SEED + r)
N_RUNS     = 10          # runs per k2
N_INITIAL  = 10          # agents infected on day 0
BETA       = 0.012
INFECT_KS  = [0.1, 0.2, 0.45, 0.75, 1.0, 5.0,10.0, None]     # k2 values (columns). None = Poisson
POISSON_K  = 1e6         # a negative binomial with this huge k is a Poisson, so None is run as this
K_COLORS   = {None: 'tab:blue', 0.1: 'tab:red', 0.45: 'tab:orange', 1.0: 'tab:green', 10.0: 'tab:gray'}   # same colours as the k1/k2 grid
FRIEND_COUNT_DISPERSION = None    # k1: None = Poisson, so it is not passed to Covasim
DAILY_TEST_PROB_IF_SYMPTOMATIC = 0.1
POP_TYPE   = 'rank'
TAKE_OFF_PCT = 1.0       # a run that infects less than this % of agents counts as 'died out'
OUT        = f'k_comparison_{N//1000}k_beta{BETA:g}_{N_RUNS}runs_seed{SEED}'     # the settings are in the file names, so runs with other settings do not overwrite each other

# The three map files (the same ones as the notebook). This assumes the script sits in network_rank_model/<folder>;
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


def k_name(k):
    return 'Poisson' if k is None else f'{k:g}'


# ---------------------------------------------------------------------------
# Analyzer: recomputes the terms of the rel_trans line and betas every day
# ---------------------------------------------------------------------------
class Tracker(cv.Analyzer):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.chunks = []          # one row per infectious person x day x layer
        self.betas_days = []      # the day of each entry of self.betas_arrays
        self.betas_arrays = []    # betas of every contact with a nonzero probability, one array per day

    def apply(self, sim):
        p = sim.people
        inf = p.infectious
        idx = np.flatnonzero(inf)
        if len(idx) == 0:
            return
        symp, iso, quar = p.symptomatic, p.isolated, p.quarantined
        f_asymp_all = symp + ~symp * sim['asymp_factor']
        sus_now = p.susceptible | (p.date_exposed == sim.t)                    # susceptible when today's coins were flipped
        beta = sim['beta'] * sim['rel_beta'] * sim['variant_pars'][sim['variant_map'][0]]['rel_beta']
        day_betas = []
        for lkey in sim.layer_keys():
            f_iso_all  = ~iso  + iso  * sim['iso_factor'][lkey]
            f_quar_all = ~quar + quar * sim['quar_factor'][lkey]
            bl = sim['beta_layer'][lkey]
            layer = p.contacts[lkey]
            p1, p2, layer_betas = layer['p1'], layer['p2'], layer['beta']
            deg = np.bincount(np.concatenate([p1, p2]), minlength=len(p))     # number of contacts of each person in this layer
            rel_trans_all = p.rel_trans * inf * f_quar_all * f_asymp_all * f_iso_all * bl * p.viral_load     # the rel_trans that compute_trans_sus returns
            self.chunks.append(pd.DataFrame(dict(day=sim.t, layer=lkey, prel_trans=p.rel_trans[idx], rel_trans=rel_trans_all[idx], n_contacts=deg[idx])))
            rel_sus_all = p.rel_sus * sus_now * f_quar_all * (1 - p.sus_imm[0, :])
            for src, tgt in ((p1, p2), (p2, p1)):
                betas = beta * layer_betas * rel_trans_all[src] * rel_sus_all[tgt]
                day_betas.append(betas[betas > 0].astype(np.float32))
        self.betas_days.append(sim.t)
        self.betas_arrays.append(np.concatenate(day_betas) if day_betas else np.array([], dtype=np.float32))

    def table(self):
        return pd.concat(self.chunks, ignore_index=True)


# --- functions start
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


def betas_daily_stats(betas_arrays, betas_days):
    ''' per day: 2.5th percentile, median, 97.5th percentile and mean of betas over all contacts with a nonzero probability '''
    rows = []
    for day, a in zip(betas_days, betas_arrays):
        if len(a):
            q = np.percentile(a, [2.5, 50, 97.5])
            rows.append((day, q[0], q[1], q[2], float(a.mean())))
    return pd.DataFrame(rows, columns=['day', 'lo', 'med', 'hi', 'mean']).set_index('day')


def overall_stats(df, betas_arrays):
    ''' median, mean and 97.5th percentile of each quantity over the whole epidemic '''
    first = df[df.layer == df.layer.iloc[0]]                                    # prel_trans belongs to the person: count each person-day once
    out = {}
    v = first.prel_trans.to_numpy(float); w = np.ones(len(v))
    out['prel_trans'] = (wq(v, w, 0.5), np.average(v, weights=w), wq(v, w, 0.975))
    v = df.rel_trans.to_numpy(float); w = df.n_contacts.to_numpy(float)         # rel_trans: each person counted once per contact
    out['rel_trans'] = (wq(v, w, 0.5), np.average(v, weights=w), wq(v, w, 0.975))
    allb = np.concatenate(betas_arrays)
    out['betas'] = (float(np.median(allb)), float(allb.mean()), float(np.percentile(allb, 97.5)))
    return out


def average_over_runs(stats_list):
    ''' average each daily statistic over the runs that took off (days with no data in a run are skipped) '''
    cols = ['lo', 'med', 'hi', 'mean']
    if not stats_list:
        return pd.DataFrame(np.nan, index=range(N_DAYS), columns=cols)
    arr = np.stack([s.reindex(range(N_DAYS))[cols].to_numpy(float) for s in stats_list])    # runs x days x statistics
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', category=RuntimeWarning)
        m = np.nanmean(arr, axis=0)
    return pd.DataFrame(m, index=range(N_DAYS), columns=cols)


def add_footer(fig, lines, fontsize=11, top=1.0):
    ''' lay out the figure and write an explanation underneath the plots '''
    footer = (0.26*len(lines) + 0.3)/fig.get_figheight()
    fig.tight_layout(rect=(0, footer, 1, top))
    fig.text(0.01, 0.006, '\n'.join(lines), ha='left', va='bottom', fontsize=fontsize, linespacing=1.4)


QUANTITIES = [('prel_trans', 'prel_trans\n(each person\'s beta_dist draw)'),
              ('rel_trans', 'rel_trans\n(product of all terms)'),
              ('betas', 'betas\n(probability a contact transmits)')]


def plot_grid(results, show_range=True):
    ''' columns = k2; rows = new infections per day, then prel_trans, rel_trans and betas over the epidemic (each row shares its y axis).
    show_range=True shades the 95% range; False shows only the median and the mean lines (no shading), so the y axis is not stretched by the tail. '''
    fig, ax = plt.subplots(1 + len(QUANTITIES), len(INFECT_KS), figsize=(5.4*len(INFECT_KS), 17), sharex=True, sharey='row', squeeze=False)
    for j, k in enumerate(INFECT_KS):
        res, c = results[k], K_COLORS.get(k, 'tab:blue')
        a = ax[0, j]
        for curve in res['curves']:
            a.plot(curve, color=c, alpha=0.35, lw=1)
        a.plot(np.mean(res['curves'], axis=0), color=c, lw=3.5)
        n_took = int(np.sum(res['took']))
        a.set_title(f'k2 = {k_name(k)}\n({n_took}/{N_RUNS} runs took off)', fontsize=15, fontweight='bold')
        for i, (key, label) in enumerate(QUANTITIES, start=1):
            a = ax[i, j]
            sm = average_over_runs(res[key])
            if show_range:
                a.fill_between(sm.index, sm.lo, sm.hi, color=c, alpha=0.25, label='95% range')
            a.plot(sm.med, color=c, lw=2.5, label='median')
            a.plot(sm['mean'], color='black', lw=2 if not show_range else 1.5, ls='--', label='mean')
            if n_took == 0:
                a.text(0.5, 0.5, 'no run took off', ha='center', va='center', transform=a.transAxes, fontsize=12, color='gray')
    ax[0, 0].set_ylabel('new infections per day\n(thin = one run, thick = mean)')
    for i, (key, label) in enumerate(QUANTITIES, start=1):
        ax[i, 0].set_ylabel(label)
    ax[1, 0].legend(loc='upper left')
    for a in ax[-1]:
        a.set_xlabel('day')
    what = 'change' if show_range else 'compare (median and mean only)'
    fig.suptitle(f'How prel_trans, rel_trans and betas {what} over the epidemic for different k2 (dispersion of infectiousness)', fontsize=16)
    lines = [
        'WHAT IS PLOTTED',
        'k2 (columns) is the dispersion of beta_dist, a negative binomial with mean 1. Small k2: a few people are very infectious and most are barely infectious. Poisson: everyone is about 1.',
        "prel_trans: each infectious person's own beta_dist draw.     rel_trans = prel_trans x f_asymp x f_iso x f_quar x beta_layer x viral_load (what compute_infections uses for the source).",
        'betas = beta x layer_betas x rel_trans[source] x rel_sus[target]: the chance that one contact transmits on one day (only contacts with an infectious source and a susceptible target).',
        ('Solid line = median, dashed = mean, shaded = 95% range' if show_range else 'Solid line = median, dashed = mean')
        + f' among infectious people (rel_trans counts each person once per contact), averaged over the runs that took off ({N_RUNS} runs per k2, same seeds in every column).',
        f'A run that infects under {TAKE_OFF_PCT:g}% of the agents counts as died out: it is in the epidemic curves but not in the rows below. Each row shares its y axis so the columns can be compared.',
    ]
    add_footer(fig, lines, top=0.96)
    return fig


def plot_overlay(results):
    ''' rows = prel_trans, rel_trans, betas; columns = median, mean, 97.5th percentile; one line per k2 '''
    stats = [('med', 'median'), ('mean', 'mean'), ('hi', '97.5th percentile')]
    fig, ax = plt.subplots(len(QUANTITIES), len(stats), figsize=(18, 14), sharex=True, squeeze=False)
    for i, (key, label) in enumerate(QUANTITIES):
        for j, (stat, stat_label) in enumerate(stats):
            a = ax[i, j]
            for k in INFECT_KS:
                sm = average_over_runs(results[k][key])
                a.plot(sm.index, sm[stat], color=K_COLORS.get(k, 'tab:blue'), lw=2.5, label=f'k2 = {k_name(k)}')
            a.set_title(f'{label.splitlines()[0]}: {stat_label}')
            if j == 0:
                a.set_ylabel(label.splitlines()[0])
            if i == len(QUANTITIES) - 1:
                a.set_xlabel('day')
    ax[0, 0].legend()
    fig.suptitle('Median, mean and upper tail of prel_trans, rel_trans and betas over the epidemic, for each k2', fontsize=16)
    lines = [
        'HOW TO READ THIS',
        'Each line is one k2, averaged over the runs that took off. The mean of prel_trans is fixed at 1 by beta_dist, so what k2 changes is the shape: a lower median and a longer upper tail when k2 is small.',
        'rel_trans and betas follow prel_trans: the 97.5th percentile shows how extreme the most infectious people (and the riskiest contacts) get.',
    ]
    add_footer(fig, lines, top=0.96)
    return fig
# --- functions end


# ---------------------------------------------------------------------------
# Run the simulation for every k2
# ---------------------------------------------------------------------------
def make_sim(seed, k, **over):
    ''' The simulation from the notebook (rank network, symptomatic testing) with infectiousness dispersion k (None = Poisson) '''
    testing = cv.test_prob(symp_prob=DAILY_TEST_PROB_IF_SYMPTOMATIC, asymp_prob=0.0, start_day=0)
    pars = dict(pop_type=POP_TYPE, pop_size=N, pop_infected=N_INITIAL, n_imports=0, beta=BETA, n_days=N_DAYS, rand_seed=seed, verbose=0,
                beta_dist=dict(dist='neg_binomial', par1=1.0, par2=POISSON_K if k is None else k, step=0.01),
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


results = {}
for k in INFECT_KS:
    res = dict(curves=[], took=[], final=[], prel_trans=[], rel_trans=[], betas=[], overall=[])
    for r in range(N_RUNS):
        tracker = Tracker(label='tracker')
        sim = make_sim(SEED + r, k, analyzers=tracker)
        initialize_and_run(sim)
        tracker = sim.get_analyzer('tracker')                     # sim deep-copies the analyzer, so fetch the one that actually ran
        final = 100*sim.results['cum_infections'].values[-1]/N    # % of agents infected by the end
        took = bool(final >= TAKE_OFF_PCT) and len(tracker.chunks) > 0
        res['curves'].append(np.array(sim.results['new_infections'].values))
        res['final'].append(final); res['took'].append(took)
        if took:
            df = tracker.table()
            first = df[df.layer == df.layer.iloc[0]]
            res['prel_trans'].append(daily_summary(first, 'prel_trans', weighted=False))
            res['rel_trans'].append(daily_summary(df, 'rel_trans', weighted=True))
            res['betas'].append(betas_daily_stats(tracker.betas_arrays, tracker.betas_days))
            res['overall'].append(overall_stats(df, tracker.betas_arrays))
        print(f'k2 = {k_name(k):8s} run {r + 1}/{N_RUNS}: {final:5.1f}% infected ({"took off" if took else "died out"})', flush=True)
        del tracker, sim
    results[k] = res

# ---------------------------------------------------------------------------
# Summary table and figures
# ---------------------------------------------------------------------------
print(f'\n{N:,} agents, {N_DAYS} days, beta {BETA}, {N_RUNS} runs per k2. Median / mean / 97.5th percentile over the whole epidemic, averaged over runs that took off:')
print(f'{"k2":>8} {"took off":>9} {"% infected":>11}   {"prel_trans":>22}   {"rel_trans":>22}   {"betas":>24}')
for k in INFECT_KS:
    res = results[k]
    n_took = int(np.sum(res['took']))
    pct = np.mean([f for f, t in zip(res['final'], res['took']) if t]) if n_took else np.nan
    cells = []
    for key in ('prel_trans', 'rel_trans', 'betas'):
        if n_took:
            m = np.mean([o[key] for o in res['overall']], axis=0)
            cells.append(f'{m[0]:7.3f} /{m[1]:7.3f} /{m[2]:7.3f}')
        else:
            cells.append(f'{"-":>23}')
    print(f'{k_name(k):>8} {n_took:>6}/{N_RUNS} {pct:>11.1f}   {cells[0]}   {cells[1]}   {cells[2]}')

plot_grid(results).savefig(f'{OUT}_grid.png', dpi=150)
plot_grid(results, show_range=False).savefig(f'{OUT}_grid_mean_median.png', dpi=150)
plot_overlay(results).savefig(f'{OUT}_overlay.png', dpi=150)
print(f'\nSaved {OUT}_grid.png, {OUT}_grid_mean_median.png and {OUT}_overlay.png')