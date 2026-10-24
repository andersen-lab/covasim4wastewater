"""
Wastewater vs cases: sensitivity analysis (clean version).

Part 1: Cases vs Cases & Wastewater, as the symptomatic testing probability varies.
Part 2: Wastewater vs Cases & Wastewater, as the wastewater sampling interval varies.

Layout: imports -> all functions -> config -> Part 1 -> Part 2.
Cells are marked with '# %%' so the file also opens as a notebook in VS Code / Spyder / Jupytext.
"""

# %% [markdown]
# # Wastewater vs cases: sensitivity analysis (clean version)
#
# Two analyses, each answering "how well does X predict true infections (`burden`)?" using adjusted R² from OLS fits:
#
# 1. **Cases vs Cases & Wastewater**, as the **symptomatic testing probability varies**.
# 2. **Wastewater vs Cases & Wastewater**, as the **wastewater sampling interval varies** (daily, every 3, 5, 7 days; two ways of filling the gaps).
#
# Layout: **imports → all functions → config → Part 1 → Part 2**. Nothing below the functions section defines a function.
#
# The simulation is the slow step. Run it once, it is saved to CSV, and both parts then work from that file (set `LOAD_SAVED = True` to skip re-simulating).

# %% [markdown]
# ## Imports

# %%
import time
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

import statsmodels.formula.api as smf
from statsmodels.stats.anova import anova_lm

import covasim as cv

# %% [markdown]
# ## Functions
#
# All functions only compute and return things. They don't read config globals, and only the plotting functions write a file (when given `save_path`).

# %% [markdown]
# ### Simulation and data extraction

# %%
def extract_wastewater_regression_data(sim) -> pd.DataFrame:
    """
    Extract daily (burden, cases, wastewater) from a Covasim sim that has been run.

    burden     -> new_infections   (true, unobserved disease burden)
    cases      -> new_diagnoses    (observed via testing/surveillance)
    wastewater -> sum of viral_shedding_hist across agents, per day

    `new_diagnoses` is all zeros unless the sim has a testing intervention.
    """
    results = sim.results

    burden = np.asarray(results['new_infections'].values, dtype=float)

    if 'new_diagnoses' not in results:
        raise KeyError("sim.results has no 'new_diagnoses' - did the sim run?")
    cases = np.asarray(results['new_diagnoses'].values, dtype=float)
    if cases.sum() == 0:
        print("Warning: 'cases' (new_diagnoses) is all zero. "
              "Add a testing intervention, e.g. cv.test_prob(...), to the sim.")

    viral_shedding_hist = np.asarray(results['viral_shedding_hist'])  # (n_people, n_days)
    wastewater = viral_shedding_hist.sum(axis=0)

    n = len(burden)
    return pd.DataFrame({
        'day': np.arange(n),
        'date': [sim.date(d) for d in range(n)],
        'burden': burden,
        'cases': cases,
        'wastewater': wastewater[:n],
    })


def build_sim(pop_size, pop_infected, n_days, test_kwargs, iso_factor, rand_seed):
    """Build one Covasim sim with a test_prob intervention."""
    return cv.Sim(
        pop_size=pop_size,
        pop_type='hybrid',
        pop_infected=pop_infected,
        location='zambia',
        n_days=n_days,
        n_imports=0,
        interventions=[cv.test_prob(**test_kwargs)],
        iso_factor=iso_factor,
        rand_seed=rand_seed,
        verbose=0,
    )


def simulate_setting(param_name, param_value, baseline, n_runs, chunk_size,
                     pop_size, pop_infected, n_days, iso_factor, rand_seed):
    """Run n_runs stochastic sims at one parameter setting (others at baseline).
    Returns a long DataFrame of daily time series: param, value, run, day, burden, cases, wastewater.
    Sims run in chunks and are discarded after extraction to keep memory bounded."""
    test_kwargs = dict(baseline)
    test_kwargs[param_name] = param_value

    frames = []
    for start in range(0, n_runs, chunk_size):
        k = min(chunk_size, n_runs - start)
        base_sim = build_sim(pop_size, pop_infected, n_days, test_kwargs, iso_factor,
                             rand_seed=rand_seed + start)
        msim = cv.MultiSim(base_sim, n_runs=k)
        msim.run()
        for i, sim in enumerate(msim.sims):
            df = extract_wastewater_regression_data(sim)[['day', 'burden', 'cases', 'wastewater']]
            df.insert(0, 'run', start + i)
            frames.append(df)
        del msim
    out = pd.concat(frames, ignore_index=True)
    out.insert(0, 'value', param_value)
    out.insert(0, 'param', param_name)
    return out


def simulate_sweeps(sweeps, **kwargs):
    """Run simulate_setting for every (param, value) in sweeps; return one concatenated DataFrame."""
    frames = []
    for param_name, values in sweeps.items():
        for value in values:
            print(f"Simulating {param_name} = {value} ...")
            frames.append(simulate_setting(param_name, value, **kwargs))
    return pd.concat(frames, ignore_index=True)

# %% [markdown]
# ### Model fitting and summaries

# %%
def fit_models(timeseries, model_type="linear", fizzle_threshold=0.0):
    """Fit cases / wastewater / both models on each run's time series.
    model_type 'linear': OLS on the raw values.
    model_type 'rank'  : OLS on the ranks of burden, cases and wastewater (Spearman-style).
    Returns one row per run."""
    if model_type not in ("linear", "rank"):
        raise ValueError("model_type must be 'linear' or 'rank'")

    rows = []
    for (param, value, run), df in timeseries.groupby(['param', 'value', 'run'], sort=False):
        fizzled = df['burden'].sum() < fizzle_threshold
        row = dict(param=param, value=value, run=run, fizzled=fizzled)
        if not fizzled:
            d = df[['burden', 'cases', 'wastewater']]
            if model_type == "rank":
                d = d.rank()
            m_cases = smf.ols('burden ~ cases', data=d).fit()
            m_ww    = smf.ols('burden ~ wastewater', data=d).fit()
            m_both  = smf.ols('burden ~ cases + wastewater', data=d).fit()
            row.update(
                adjr2_cases=m_cases.rsquared_adj,
                adjr2_wastewater=m_ww.rsquared_adj,
                adjr2_both=m_both.rsquared_adj,
                p_add_ww=anova_lm(m_cases, m_both)['Pr(>F)'].iloc[1],     # wastewater adds value beyond cases?
                p_add_cases=anova_lm(m_ww, m_both)['Pr(>F)'].iloc[1],     # cases adds value beyond wastewater?
            )
        rows.append(row)
    return pd.DataFrame(rows)


def summarize(per_run, n_boot=2000, seed=0):
    """Collapse per-run fits to one row per (param, value): median, IQR, bootstrap 95% CI of the median,
    plus mean and SD of adjusted R² (used by the error-bar plots)."""
    rng = np.random.default_rng(seed)
    out = []
    for (param, value), g in per_run.groupby(['param', 'value'], sort=False):
        used = g[~g['fizzled']]
        row = dict(param=param, value=value, n_used=len(used), n_fizzled=int(g['fizzled'].sum()))
        for name in ("cases", "wastewater", "both"):
            x = used[f"adjr2_{name}"].to_numpy(dtype=float)
            if len(x):
                boot = np.median(rng.choice(x, size=(n_boot, len(x)), replace=True), axis=1)
                row[f"median_AdjR2_{name}"] = np.median(x)
                row[f"q25_{name}"], row[f"q75_{name}"] = np.percentile(x, [25, 75])
                row[f"ci_low_{name}"], row[f"ci_high_{name}"] = np.percentile(boot, [2.5, 97.5])
                row[f"mean_{name}"] = x.mean()
                row[f"sd_{name}"] = x.std(ddof=1) if len(x) > 1 else np.nan
            else:
                row.update({f"median_AdjR2_{name}": np.nan, f"q25_{name}": np.nan, f"q75_{name}": np.nan,
                            f"ci_low_{name}": np.nan, f"ci_high_{name}": np.nan,
                            f"mean_{name}": np.nan, f"sd_{name}": np.nan})
        row["pct_ww_helps"]    = (used["p_add_ww"] < 0.05).mean() * 100 if len(used) else np.nan
        row["pct_cases_helps"] = (used["p_add_cases"] < 0.05).mean() * 100 if len(used) else np.nan
        out.append(row)
    return pd.DataFrame(out)

# %% [markdown]
# ### Wastewater sampling (Part 2)

# %%
def thin_wastewater(daily_run, days_between_samples, n_samples_averaged,
                    first_sample_day=0, analysis_start_day=0):
    """
    Turn ONE run's daily wastewater series into what a sparse sampling programme would report.

    1. A sample is taken only every `days_between_samples` days, starting on `first_sample_day`.
       On all other days there is no measurement.
    2. Every day is then given the average of the samples inside a BACKWARD-looking window that ends on that
       day and is (n_samples_averaged * days_between_samples) days long, i.e. the last `n_samples_averaged`
       samples. Nothing after the day is used, so there is no leakage from the future.
       With n_samples_averaged = 1 this is "carry the latest sample forward" (persistence).
       Early in a run fewer samples exist, so it averages whatever is available.
    3. Days before `analysis_start_day` are dropped, so every sampling scheme is compared on the same days.

    days_between_samples = 1 means daily wastewater: the series is untouched (apart from step 3).

    Args:
        daily_run:            DataFrame for one run, one row per day, sorted by day, with a `wastewater` column.
        days_between_samples: gap between samples in days (1 = daily).
        n_samples_averaged:   how many of the latest samples are averaged (1 = last sample carried forward).
        first_sample_day:     day of the first sample (randomise between runs so no schedule is favoured).
        analysis_start_day:   first day kept for the regressions.

    Returns:
        A copy of daily_run with `wastewater` replaced by the sparsely sampled, averaged series.
    """
    result = daily_run.copy()

    if days_between_samples > 1:
        # row numbers (= day numbers) on which a sample is taken
        sample_positions = np.arange(first_sample_day, len(result), days_between_samples)

        # true wastewater on sample days, NaN (= no measurement) on every other day
        measured = pd.Series(np.nan, index=result.index)
        measured.iloc[sample_positions] = result["wastewater"].to_numpy(float)[sample_positions]

        # window length in days: always spans exactly n_samples_averaged samples
        window_days = n_samples_averaged * days_between_samples
        result["wastewater"] = measured.rolling(window_days, min_periods=1).mean()

    return result[result["day"] >= analysis_start_day]


def fit_sampling_schemes(timeseries, symp_probs, sampling_every, windows,
                         model_type="linear", fizzle_threshold=0.0, param="symp_prob", seed=0):
    """Fit the cases / wastewater / both models for every (window, sampling interval) combination.

    Each run gets a random first sample day, so no sampling schedule is favoured. All fits use days
    >= max(sampling_every) - 1, so every run and every scheme is compared on the same rows and every day
    has a past sample to use. Daily data (every = 1) is unaffected by the window, so it is fitted once.
    Returns one row per run per scheme, with columns `every` and `window` added."""
    first_day = max(sampling_every) - 1

    selected = timeseries[(timeseries["param"] == param) &
                          np.isclose(timeseries["value"].astype(float).to_numpy()[:, None], symp_probs).any(axis=1)]
    runs = [d.sort_values("day") for _, d in selected.groupby(["param", "value", "run"], sort=False)]

    rng = np.random.default_rng(seed)
    fits = []
    for win in windows:
        for every in sampling_every:
            if every == 1 and win != windows[0]:
                continue
            t0 = time.time()
            parts = [thin_wastewater(d, every, win,
                                     first_sample_day=int(rng.integers(0, every)),
                                     analysis_start_day=first_day) for d in runs]
            f = fit_models(pd.concat(parts, ignore_index=True), model_type=model_type,
                           fizzle_threshold=fizzle_threshold)
            f["every"], f["window"] = every, win
            fits.append(f)
            print(f"window {win}, every {every} day(s): {time.time() - t0:.0f} s", flush=True)
    return pd.concat(fits, ignore_index=True)


def summarize_sampling(per_run_sampling):
    """Mean and SD of adjusted R² (cases, wastewater, both) for each (testing prob, window, interval)."""
    used = per_run_sampling[~per_run_sampling["fizzled"].astype(bool)]
    rows = []
    for (v, win, every), g in used.groupby(["value", "window", "every"]):
        rows.append(dict(value=v, window=win, every=every, n=len(g),
                         mean_cases=g["adjr2_cases"].mean(), sd_cases=g["adjr2_cases"].std(ddof=1),
                         mean_ww=g["adjr2_wastewater"].mean(), sd_ww=g["adjr2_wastewater"].std(ddof=1),
                         mean_both=g["adjr2_both"].mean(), sd_both=g["adjr2_both"].std(ddof=1)))
    return pd.DataFrame(rows)

# %% [markdown]
# ### Plotting

# %%
BLUE = "#224e95"      # colour of the "Cases & Wastewater" line in both plots
WW_BLUE = "cornflowerblue"   # colour of the "Wastewater only" line


def _clean_axes(ax):
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.grid(alpha=0.3)


def plot_cases_vs_both(results, param="symp_prob", offset=0.004,
                       title="Reported Cases: Varying the Symptomatic Testing Probability",
                       save_path=None):
    """Part 1 plot: mean adjusted R² (± SD across runs) of Cases only vs Cases & Wastewater,
    as the symptomatic testing probability varies. Returns the figure."""
    styles = [  # model, legend label, colour, line style, marker
        ("cases", "Reported Cases only",           "black", "-",  "o"),
        ("both",  "Reported Cases & Wastewater",   BLUE,    "--", "^"),
    ]
    sub = results[results["param"] == param].sort_values("value")
    x = sub["value"].astype(float).to_numpy()

    fig, ax = plt.subplots(figsize=(9.2, 6.9))
    for k, (name, label, color, ls, mk) in enumerate(styles):
        dx = (k - (len(styles) - 1) / 2) * offset      # small horizontal nudge so the bars don't overlap
        ax.errorbar(x + dx, sub[f"mean_{name}"], yerr=sub[f"sd_{name}"], color=color, ls=ls, marker=mk,
                    lw=3, ms=10, capsize=4, elinewidth=1.8, label=label, clip_on=False)
    ax.set_xlim(left=0)
    _clean_axes(ax)

    handles, _ = ax.get_legend_handles_labels()
    handles.append(Line2D([0], [0], color="gray", marker="_", ls="none", markersize=14,
                          label="Error bars: \u00b1 SD across runs"))
    leg = ax.legend(handles=handles, fontsize=16, loc="lower right", frameon=False)
    leg.get_texts()[-1].set_fontsize(10)               # only the error-bar entry is small

    ax.set_xlabel("Symptomatic testing probability (per day)", fontsize=17)
    ax.set_ylabel("Mean adjusted R\u00b2 across simulated outbreaks", fontsize=16)
    ax.set_title(title, fontsize=16)
    ax.tick_params(labelsize=14)
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=200, bbox_inches="tight")
    return fig


def plot_wastewater_vs_both(sampling, window, symp_probs, plot_every, window_titles, save_path=None):
    """Part 2 plot: mean adjusted R² (± SD across runs) of Wastewater only vs Cases & Wastewater,
    by wastewater sampling interval, for one averaging window. One panel per testing probability.
    The y range is shared across windows (computed from all of `sampling`), so plots compare directly.
    Returns the figure."""
    shown = sampling[sampling["every"].isin(plot_every)]        # only the intervals to display
    lo = min((shown["mean_ww"] - shown["sd_ww"]).min(), (shown["mean_both"] - shown["sd_both"]).min())
    hi = max((shown["mean_ww"] + shown["sd_ww"]).max(), (shown["mean_both"] + shown["sd_both"]).max())
    pad = 0.03 * (hi - lo)

    fig, axes = plt.subplots(1, len(symp_probs), figsize=(9.2 * len(symp_probs), 6.9), sharey=True)
    axes = np.atleast_1d(axes)
    for ax, sp in zip(axes, symp_probs):
        sub = shown[np.isclose(shown["value"].astype(float), sp)]
        daily = sub[sub["every"] == 1].iloc[0]                  # daily data is the same for every window
        s = pd.concat([sub[(sub["window"] == window) & (sub["every"] > 1)],
                       pd.DataFrame([daily])]).sort_values("every")

        # points sit at their true interval (no sideways nudge); "both" is drawn on top
        ax.errorbar(s["every"], s["mean_ww"], yerr=s["sd_ww"], color=WW_BLUE, marker="s", lw=3, ms=12,
                    capsize=4, elinewidth=1.6, label="Wastewater only", clip_on=False, zorder=2)
        ax.errorbar(s["every"], s["mean_both"], yerr=s["sd_both"], color=BLUE, marker="^", ls="--", lw=3, ms=9,
                    capsize=4, elinewidth=1.6, label="Cases & Wastewater", clip_on=False, zorder=3)

        x_pad = 0.05
        ax.set_xticks(sorted(plot_every))
        ax.set_xlim(min(plot_every) - x_pad, max(plot_every) + x_pad)
        ax.set_ylim(lo - pad, hi + pad)
        _clean_axes(ax)
        ax.set_title(f"Wastewater Sampling: {window_titles.get(window, f'window {window}')}", fontsize=14)
        ax.set_xlabel("Wastewater sampling interval (days between samples)", fontsize=16)
        ax.set_ylabel("Mean adjusted R\u00b2 across simulated outbreaks", fontsize=15)
        ax.tick_params(labelsize=14)

    handles, _ = axes[0].get_legend_handles_labels()
    handles.append(Line2D([0], [0], color="gray", marker="_", ls="none", markersize=14,
                          label="Error bars: \u00b1 SD across runs"))
    leg = fig.legend(handles=handles, loc="lower center", ncol=1, fontsize=13, frameon=False)
    leg.get_texts()[-1].set_fontsize(11)
    fig.tight_layout(rect=[0, 0.14, 1, 1])
    if save_path:
        fig.savefig(save_path, dpi=200, bbox_inches="tight")
    return fig

# %% [markdown]
# ## Config
# Edit this cell, then run everything below it.

# %%
# ---- Simulation settings (changing these means re-simulating) -------------
POP_SIZE      = 10000
POP_INFECTED  = int(POP_SIZE * 0.0016)   # ~0.16% of the population (16 per 10,000)
N_DAYS        = 120
N_RUNS        = 100                      # random seeds per parameter setting
CHUNK_SIZE    = 10                       # sims held in memory at once
RAND_SEED     = 1

TEST_DELAY    = 0                        # fixed testing delay (days)
BASELINE      = dict(symp_prob=0.03, asymp_prob=0.002, test_delay=TEST_DELAY)
ISO_FACTOR    = dict(h=0.4, s=0.1, w=0.1, c=0.1)   # constant in every simulation

# Part 1: the swept parameter (the swept value overrides BASELINE)
SWEEPS = {
    "symp_prob": [0.01, 0.03, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35],
}

# ---- Analysis settings (changing these does NOT need re-simulation) --------
MODEL_TYPE       = "linear"              # "linear" or "rank"
FIZZLE_THRESHOLD = 0.01 * POP_SIZE       # skip runs where <1% of the population was ever infected

# Part 2: wastewater sampling
SAMPLING_EVERY       = [1, 3, 5, 7, 14]  # days between wastewater samples (1 = daily, untouched)
WINDOWS              = [1, 3]            # average of the last N samples (1 = last sample carried forward)
SAMPLING_SYMP_PROBS  = [0.10]            # testing level(s) for Part 2, one panel each (must be in SWEEPS)
PLOT_EVERY           = [1, 3, 5, 7]      # intervals shown in the plots (14 stays in the saved table)
WINDOW_TITLES = {1: "Persistence model - last sample carried forward",
                 3: "Moving average of last 3 samples - Right-aligned window"}

# ---- Run mode and output locations ------------------------------------------
LOAD_SAVED = False                       # True = skip simulating and load the saved time series instead
OUT_DIR  = Path("sensitivity_results")
SIM_TAG  = f"pop{POP_SIZE}_delay{TEST_DELAY}"      # names the simulation file
FIT_TAG  = f"{SIM_TAG}_{MODEL_TYPE}"               # names the fit/plot files
OUT_DIR.mkdir(exist_ok=True)
print("SIM_TAG:", SIM_TAG, "| FIT_TAG:", FIT_TAG)

# %% [markdown]
# ## Simulate (slow) or load
# Only re-simulate when `POP_SIZE`, `TEST_DELAY`, `N_RUNS` or `SWEEPS` change.

# %%
ts_file = OUT_DIR / f"timeseries_{SIM_TAG}.csv"

if LOAD_SAVED:
    timeseries = pd.read_csv(ts_file)
    print("Loaded", ts_file)
else:
    timeseries = simulate_sweeps(
        SWEEPS,
        baseline=BASELINE, n_runs=N_RUNS, chunk_size=CHUNK_SIZE,
        pop_size=POP_SIZE, pop_infected=POP_INFECTED, n_days=N_DAYS,
        iso_factor=ISO_FACTOR, rand_seed=RAND_SEED,
    )
    timeseries.to_csv(ts_file, index=False)
    print("Saved", ts_file)

print(timeseries.head())

# %% [markdown]
# ## Part 1: Cases vs Cases & Wastewater (testing probability varies)

# %%
per_run = fit_models(timeseries, model_type=MODEL_TYPE, fizzle_threshold=FIZZLE_THRESHOLD)
results = summarize(per_run)

per_run.to_csv(OUT_DIR / f"per_run_{FIT_TAG}.csv", index=False)
results.to_csv(OUT_DIR / f"summary_{FIT_TAG}.csv", index=False)
print(results.round(3).to_string(index=False))

# %%
fig1 = plot_cases_vs_both(results, save_path=OUT_DIR / f"adjR2_cases_vs_both_symp_prob_{FIT_TAG}.png")
plt.show()

# %% [markdown]
# ## Part 2: Wastewater vs Cases & Wastewater (wastewater sampling varies)
# Each run gets a random first sample day. Fits use days ≥ `max(SAMPLING_EVERY) - 1` so all schemes use the same days.

# %%
per_run_sampling = fit_sampling_schemes(
    timeseries, SAMPLING_SYMP_PROBS, SAMPLING_EVERY, WINDOWS,
    model_type=MODEL_TYPE, fizzle_threshold=FIZZLE_THRESHOLD,
)
sampling = summarize_sampling(per_run_sampling)

sampling.to_csv(OUT_DIR / f"sampling_trailing_randomstart_{MODEL_TYPE}.csv", index=False)
print(sampling.round(3).to_string(index=False))

# %%
# window of 1 sample: each day uses the last sample carried forward
fig2a = plot_wastewater_vs_both(
    sampling, window=1, symp_probs=SAMPLING_SYMP_PROBS, plot_every=PLOT_EVERY, window_titles=WINDOW_TITLES,
    save_path=OUT_DIR / f"adjR2_wastewater_vs_both_window1_{MODEL_TYPE}.png",
)
plt.show()

# %%
# window of 3 samples: each day uses the average of the last 3 samples
fig2b = plot_wastewater_vs_both(
    sampling, window=3, symp_probs=SAMPLING_SYMP_PROBS, plot_every=PLOT_EVERY, window_titles=WINDOW_TITLES,
    save_path=OUT_DIR / f"adjR2_wastewater_vs_both_window3_{MODEL_TYPE}.png",
)
plt.show()