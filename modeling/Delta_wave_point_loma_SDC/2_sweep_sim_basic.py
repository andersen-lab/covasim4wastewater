import matplotlib
matplotlib.use("Agg")                      # no display on the server; must come before importing pyplot
import textwrap
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import StrMethodFormatter
import covasim as cv

# Make "modeling" importable: script -> Delta_wave_point_loma_SDC -> modeling -> repo root
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from modeling.modeling_helper_functions import extract_wastewater_regression_data, make_vaccination, make_variants

# ---------------------------------------------------------------
# Paths
# ---------------------------------------------------------------
SCRIPT_DIR  = Path(__file__).resolve().parent
OUT_DIR     = SCRIPT_DIR / "sim_results_basic"           # outputs next to the script, whatever folder you run from
OUT_DIR.mkdir(exist_ok=True)
DATA_FOLDER = Path.home() / "GitHub" / "covasim_fit_data"

# ---------------------------------------------------------------
# DATA / PARAMETERS
# ---------------------------------------------------------------
# Simulation size
POP_SIZE      = 50000
CATCHMENT_POP = 2116170                      # Point Loma catchment, from cases.csv
POP_SCALE     = CATCHMENT_POP / POP_SIZE     # real people per agent

# Window (Timestamps for filtering the data, strings for Covasim)
START_DAY = pd.Timestamp("2021-06-28")       # sim start
END_DAY   = pd.Timestamp("2021-11-30")       # last day of the window
START_STR, END_STR = str(START_DAY.date()), str(END_DAY.date())

# County figures (source: county updates 30 Jun and 1 Sep 2021)
POP_COUNTY       = 3347827
POP_COUNTY_12P   = 2802581
POP_PL           = 2116170

# Variant share at the start of the window (Point Loma SEARCH sequences, week of 27 Jun 2021)
VARIANT_SHARE_PL = {"delta": 0.78,
                    "alpha": 0.12,
                    "gamma": 0.06,
                    "wild_other": 0.04}

# County head counts, residents 12+
county_n_one_dose_0630 = 2190000
county_n_full_0630     = 1870000
county_n_one_dose_0901 = 2407528
county_n_full_0901     = 2112936

# Share of ALL residents (Covasim needs everyone, not just 12+).
# ASSUMPTION: Point Loma coverage = county coverage (no zip-level data used yet)
pl_share_one_dose_0630 = county_n_one_dose_0630 / POP_COUNTY
pl_share_full_0630     = county_n_full_0630     / POP_COUNTY
pl_share_one_dose_0901 = county_n_one_dose_0901 / POP_COUNTY
pl_share_full_0901     = county_n_full_0901     / POP_COUNTY

# Uptake among the unvaccinated: fraction of people with no dose on 30 Jun who had a first dose by 1 Sep
#   numerator   = rise in the share of ALL residents with >=1 dose (0.719 - 0.654 = 0.065)
#   denominator = share still unvaccinated on 30 Jun (1 - 0.654 = 0.346)
pl_uptake_unvacc = (pl_share_one_dose_0901 - pl_share_one_dose_0630) / (1 - pl_share_one_dose_0630)


# ---------------------------------------------------------------
# MODEL
# ---------------------------------------------------------------
def run_basic_no_immune(beta=0.016, n_seed=100, vaccinate=False, symp_prob=0.1,
              asymp_prob=0.005, test_delay=2, vacc_start_days=120, pfizer_share=0.60, seed=0):
    """One run. Each argument is one 'layer' you can switch or change."""
    variants, n_wild = make_variants(n_seed)
    interventions = [cv.test_prob(symp_prob=symp_prob, asymp_prob=asymp_prob, test_delay=test_delay)]
    if vaccinate:
        interventions += make_vaccination(vacc_start_days, pfizer_share, pl_share_one_dose_0630, pl_uptake_unvacc)

    sim = cv.Sim(pop_size=POP_SIZE, pop_scale=POP_SCALE, pop_type="hybrid", location="usa",
                 beta=beta, pop_infected=n_wild, n_imports=0,
                 start_day=START_STR, end_day=END_STR,
                 variants=variants, use_waning=True,
                 interventions=interventions, rand_seed=seed, verbose=0)
    sim.run()
    # remember the settings so the plot and the description can report them
    sim.layer_params = dict(beta=beta, n_seed=n_seed, vaccinate=vaccinate, symp_prob=symp_prob,
                            asymp_prob=asymp_prob, test_delay=test_delay,
                            vacc_start_days=vacc_start_days, pfizer_share=pfizer_share, seed=seed)
    return sim


# ---------------------------------------------------------------
# MEASURES (use the globals wk, real, df_pl_ww, which are loaded in main)
# ---------------------------------------------------------------
def weekly(sim, key):
    """Sum a daily sim result over the real reporting weeks."""
    s = pd.Series(sim.results[key].values, index=pd.to_datetime(sim.datevec))
    return np.array([s.loc[a:b].sum() for a, b in zip(wk["week_start"], wk["week_end"])])


def growth(y, lo=2, hi=7):
    """Growth per day over weeks 3-7 (skips the start-up weeks)."""
    t = np.arange(len(y)) * 7.0
    return np.polyfit(t[lo:hi], np.log(np.asarray(y[lo:hi], float) + 1), 1)[0]

def check_vaccination(vacc_start_days=120, pfizer_share=0.60):
    """One-day sim with only the vaccination: does day 0 have the right coverage?"""
    chk = cv.Sim(pop_size=POP_SIZE, pop_scale=POP_SCALE, pop_type="hybrid", location="usa",
                 start_day=START_STR, end_day=str((START_DAY + pd.Timedelta(days=1)).date()),
                 use_waning=True, verbose=0,
                 interventions=make_vaccination(vacc_start_days, pfizer_share,
                                                pl_share_one_dose_0630, pl_uptake_unvacc))
    chk.run()
    d = chk.people.doses
    print(f"vaccination check:  1+ dose {np.mean(d >= 1):.3f} (target {pl_share_one_dose_0630:.3f})   "
          f"2+ doses {np.mean(d >= 2):.3f} (target {pl_share_full_0630:.3f})")
    
def summarise(sim, label):
    inf, dia = weekly(sim, "new_infections"), weekly(sim, "new_diagnoses")
    return dict(layer=label,
                growth_per_day=round(growth(inf), 3),
                peak_week=int(inf.argmax()) + 1,
                peak_infections_per_week=int(inf.max()),
                peak_diagnoses_per_week=int(dia.max()),
                share_infected=round(sim.results["cum_infections"].values[-1] / CATCHMENT_POP, 2))


# ---------------------------------------------------------------
# DESCRIPTION AND PLOT
# ---------------------------------------------------------------
def describe_sim(sim):
    """Print a few plain sentences about what this run represents."""
    p = sim.layer_params
    inf, dia = weekly(sim, "new_infections"), weekly(sim, "new_diagnoses")
    cum_inf, cum_dia = sim.results["cum_infections"].values[-1], sim.results["cum_diagnoses"].values[-1]
    S = VARIANT_SHARE_PL

    para = [
        f"This simulation represents the Delta wave in the Point Loma catchment ({CATCHMENT_POP:,} people) from "
        f"{START_STR} to {END_STR}, using {POP_SIZE:,} simulated agents that each stand for about {POP_SCALE:.0f} real people.",

        f"It starts with {p['n_seed']} infected agents split by the Point Loma sequencing data: "
        f"{S['delta']:.0%} Delta, {S['alpha']:.0%} Alpha, {S['gamma']:.0%} Gamma and {S['wild_other']:.0%} wild/other. "
        f"Transmission uses beta = {p['beta']} (Covasim's default is 0.016) with Covasim's built-in variant properties and cross-immunity.",

        (f"Vaccination is ON: {pl_share_one_dose_0630:.0%} of residents have a first dose by 30 June, using county figures "
         f"(assumed to match Point Loma). Nothing else gives anyone immunity, so earlier infections are not represented."
         if p["vaccinate"] else
         "Vaccination is OFF and nobody has immunity from earlier infections, so everyone starts susceptible."),

        f"There is no change in behaviour over the summer. Symptomatic people are tested with probability {p['symp_prob']} per day, "
        f"asymptomatic people with {p['asymp_prob']}, and results take {p['test_delay']} days. "
        f"Testing only changes how many infections are reported as cases, never the infections themselves.",

        f"In this run, {cum_inf / CATCHMENT_POP:.0%} of the catchment was infected, with a peak of {int(inf.max()):,} infections "
        f"in week {int(inf.argmax()) + 1}. About {cum_dia / max(cum_inf, 1):.0%} of infections were diagnosed, giving a peak of "
        f"{int(dia.max()):,} diagnoses per week against {int(real.max()):,} real reported cases (real peak in week {int(real.argmax()) + 1}).",
    ]
    print("\nWHAT THIS SIMULATION REPRESENTS")
    for s in para:
        print(textwrap.fill(s, width=100), "\n")


def plot_basic(sim, title="", save_to=None):
    """Three linear panels: true infections, cases (sim vs real), wastewater (sim vs real)."""
    p = sim.layer_params
    inf, dia = weekly(sim, "new_infections"), weekly(sim, "new_diagnoses")
    d = extract_wastewater_regression_data(sim)
    d["date"] = pd.to_datetime(d["date"])
    ww_sim = d.set_index("date")["wastewater"]
    ww_real = (df_pl_ww[(df_pl_ww["Sample_Date"] >= START_DAY) & (df_pl_ww["Sample_Date"] <= END_DAY)]
               .set_index("Sample_Date")["ww_copies_per_L"].sort_index())
    scale = ww_real.mean() / ww_sim.reindex(ww_real.index, method="nearest").mean()   # one constant: units are arbitrary

    fig, ax = plt.subplots(3, 1, figsize=(11, 12), sharex=True)

    ax[0].plot(wk["week_mid"], inf, "o-", color="tab:gray", label="sim TRUE infections")
    ax[0].set_ylabel("infections per week")
    ax[0].set_title("Simulated true infections (not observable in real life)")

    ax[1].plot(wk["week_mid"], dia, "o-", color="tab:green", label="sim diagnoses")
    ax[1].plot(wk["week_mid"], real, "o-", color="#D95F02", lw=2.5, label="REAL reported cases")
    ax[1].set_ylabel("cases per week")
    ax[1].set_title("Reported cases: simulated diagnoses vs real")

    ax[2].plot(ww_real.index, ww_real.values, ".", ms=8, color="tab:blue", label="REAL wastewater")
    ax[2].plot(ww_sim.index, ww_sim * scale, color="tab:green", label="sim wastewater (one scale constant)")
    ax[2].set_ylabel("copies / L")
    ax[2].set_title("Wastewater")
    ax[2].set_xlim(START_DAY, END_DAY)

    for a in ax:
        a.yaxis.set_major_formatter(StrMethodFormatter("{x:,.0f}"))
        a.legend(fontsize=9)

    # Parameters, written on the figure so the plot says what it represents
    text = (f"beta={p['beta']}   seeded agents={p['n_seed']}   vaccination={'ON' if p['vaccinate'] else 'OFF'}"
            + (f" (start {p['vacc_start_days']} d before, {p['pfizer_share']:.0%} Pfizer)" if p["vaccinate"] else "")
             + f"\nprior-infection immunity={p['immune_frac']:.0%} of agents"
            + f"\nsymptomatic test prob={p['symp_prob']}   asymptomatic={p['asymp_prob']}   test delay={p['test_delay']} d"
            + f"   seed={p['seed']}\n{POP_SIZE:,} agents = {CATCHMENT_POP:,} people, {START_STR} to {END_STR}")
    fig.suptitle(title, fontsize=14, y=0.995)
    fig.text(0.01, 0.965, text, va="top", fontsize=8.5, family="monospace",
             bbox=dict(boxstyle="round", facecolor="whitesmoke", edgecolor="grey"))
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    if save_to:
        fig.savefig(save_to, dpi=150)
    plt.close(fig)

def sweep(results, label, **kwargs):
    """Run one sim with the given settings and add a row to the results table."""
    sim = run_basic(**kwargs)
    row = summarise(sim, label)
    results.append(row)
    print(f"{label:<28} growth {row['growth_per_day']:+.3f}/day  peak infections {row['peak_infections_per_week']:>9,}  "
          f"share infected {row['share_infected']:.2f}", flush=True)
    return sim

def run_basicx(beta=0.016, n_seed=100, vaccinate=False, symp_prob=0.1, asymp_prob=0.005,
              test_delay=2, vacc_start_days=120, pfizer_share=0.60, immune_frac=0.0, seed=0):
    ...
    sim = cv.Sim(pop_size=POP_SIZE, pop_scale=POP_SCALE, pop_type="hybrid", location="usa",
                 beta=beta, pop_infected=n_wild, n_imports=0,
                 start_day=START_STR, end_day=END_STR,
                 variants=variants, use_waning=True,
                 interventions=interventions, rand_seed=seed, verbose=0)
    sim.initialize()
    if immune_frac > 0:
        rng = np.random.default_rng(seed + 1000)                 # separate random stream from the sim's own
        immune = rng.random(len(sim.people)) < immune_frac
        sim.people.rel_sus[immune] = 0.0                         # these agents cannot be infected
        print(f"immune agents: {immune.mean():.2f} of {len(immune):,}")
    sim.run()
    sim.layer_params = dict(beta=beta, n_seed=n_seed, vaccinate=vaccinate, symp_prob=symp_prob,
                            asymp_prob=asymp_prob, test_delay=test_delay, vacc_start_days=vacc_start_days,
                            pfizer_share=pfizer_share, immune_frac=immune_frac, seed=seed)
    return sim

def sweep(results, label, **kwargs):
    """Run one sim with the given settings, add a row to the table, and return the sim."""
    sim = run_basic(**kwargs)
    row = summarise(sim, label)
    results.append(row)
    print(f"{label:<30} growth {row['growth_per_day']:+.3f}/day   peak infections {row['peak_infections_per_week']:>9,}   "
          f"peak diagnoses {row['peak_diagnoses_per_week']:>8,}   share infected {row['share_infected']:.2f}", flush=True)
    return sim

def run_basic(beta=0.016, n_seed=100, vaccinate=False, symp_prob=0.1, asymp_prob=0.005,
              test_delay=2, vacc_start_days=120, pfizer_share=0.60, immune_frac=0.0, seed=0):
    """One run. Each argument is one 'layer' you can switch or change."""
    variants, n_wild = make_variants(n_seed)                       # n_wild is defined here
    interventions = [cv.test_prob(symp_prob=symp_prob, asymp_prob=asymp_prob, test_delay=test_delay)]
    if vaccinate:
        interventions += make_vaccination(vacc_start_days, pfizer_share, pl_share_one_dose_0630, pl_uptake_unvacc)

    sim = cv.Sim(pop_size=POP_SIZE, pop_scale=POP_SCALE, pop_type="hybrid", location="usa",
                 beta=beta, pop_infected=n_wild, n_imports=0,
                 start_day=START_STR, end_day=END_STR,
                 variants=variants, use_waning=True,
                 interventions=interventions, rand_seed=seed, verbose=0)
    sim.initialize()
    if immune_frac > 0:
        rng = np.random.default_rng(seed + 1000)                   # separate random stream from the sim's own
        immune = rng.random(len(sim.people)) < immune_frac
        sim.people.rel_sus[immune] = 0.0                           # these agents cannot be infected
        print(f"immune agents: {immune.mean():.2f} of {len(immune):,}")
    sim.run()

    sim.layer_params = dict(beta=beta, n_seed=n_seed, vaccinate=vaccinate, symp_prob=symp_prob,
                            asymp_prob=asymp_prob, test_delay=test_delay, vacc_start_days=vacc_start_days,
                            pfizer_share=pfizer_share, immune_frac=immune_frac, seed=seed)
    return sim

# ---------------------------------------------------------------
# RUN
# ---------------------------------------------------------------
if __name__ == "__main__":
    # Parameter check
    print(f"Share with one dose 06/30: {pl_share_one_dose_0630:.3f}")
    print(f"Share fully vaccinated 06/30: {pl_share_full_0630:.3f}")
    print(f"Share with one dose 09/01: {pl_share_one_dose_0901:.3f}")
    print(f"Share fully vaccinated 09/01: {pl_share_full_0901:.3f}")
    print(f"Uptake among the unvaccinated, 06/30 to 09/01: {pl_uptake_unvacc:.3f}   (about 0.188)")

    # Real weekly cases
    df_pl_weekly = pd.read_pickle(DATA_FOLDER / "df_cases_weekly_pl.pkl")
    wk = df_pl_weekly[(df_pl_weekly["week_start"] >= START_DAY) &
                      (df_pl_weekly["week_end"] <= END_DAY)].reset_index(drop=True)
    real = wk["weekly_total"].to_numpy(float)
    print(len(wk), "weeks;", real[:3], "sum", real.sum())        # expect 22 weeks, [792. 1394. 2157.]

    # Real wastewater
    df_pl_ww = pd.read_pickle(DATA_FOLDER / "pl_wastewater.pkl")
    df_pl_ww = df_pl_ww.rename(columns={"Mean viral gene copies/L": "ww_copies_per_L"})
    df_pl_ww["Sample_Date"] = pd.to_datetime(df_pl_ww["Sample_Date"], format="%m/%d/%y")
    print(df_pl_ww.columns.tolist())                              # need Sample_Date and ww_copies_per_L
    print(df_pl_ww.shape, df_pl_ww["Sample_Date"].min().date(), df_pl_ww["Sample_Date"].max().date())


    # Results table starts with the real data
    results = [dict(layer="REAL DATA", growth_per_day=round(growth(real), 3), peak_week=int(real.argmax()) + 1,
                    peak_infections_per_week=None, peak_diagnoses_per_week=int(real.max()), share_infected=None)]

    # Default beta (0.016), known vaccination ON, increasing prior-infection immunity
    sims = {}
    for f in (0.0, 0.1, 0.2, 0.3, 0.4):
        sims[f] = sweep(results, f"vacc + prior immunity {f:.0%}", vaccinate=True, immune_frac=f)

    # Table first, so a plotting error cannot lose the results
    table = pd.DataFrame(results)
    print("\n" + table.to_string(index=False))
    table.to_csv(OUT_DIR / "layer_results_immunity.csv", index=False)
    print("Saved", OUT_DIR / "layer_results_immunity.csv")

    # What each run represents, and a figure for each
    for f, sim in sims.items():
        describe_sim(sim)
        plot_basic(sim, f"Vaccination ON + prior immunity {f:.0%} (default beta)",
                   save_to=OUT_DIR / f"immunity_{int(f * 100)}.png")
        print("Saved", OUT_DIR / f"immunity_{int(f * 100)}.png")

    ''' 
    # One run: vaccination ON, beta 0.008
    sim_b = run_basic(vaccinate=True, beta=0.008)
    describe_sim(sim_b)

    inf, dia = weekly(sim_b, "new_infections"), weekly(sim_b, "new_diagnoses")
    print(f"implied ascertainment: {dia.sum() / inf.sum():.2f} (diagnoses / infections over the window)")
    print(f"infections peak in week {inf.argmax() + 1} of {len(inf)}; real cases peak in week {real.argmax() + 1}")

    results.append(summarise(sim_b, "vacc + beta 0.008"))
    table = pd.DataFrame(results)
    print(table.to_string(index=False))
    table.to_csv(OUT_DIR / "layer_results.csv", index=False)

    plot_basic(sim_b, "Vaccination ON, beta 0.008", save_to=OUT_DIR / "vacc_beta0008.png")
    print("Saved", OUT_DIR / "vacc_beta0008.png")

    
    sim0 = sweep(results, "0: baseline")
    sim1 = sweep(results, "1: + vaccination", vaccinate=True)
    for b in (0.012, 0.010, 0.008):
        sweep(results, f"2: vacc + beta {b}", vaccinate=True, beta=b)
    for n in (30, 300):
        sweep(results, f"3: vacc + seed {n}", vaccinate=True, n_seed=n)

    table = pd.DataFrame(results)
    print(table.to_string(index=False))
    table.to_csv(OUT_DIR / "layer_results.csv", index=False)
    plot_basic(sim1, "Layer 1: + vaccination", save_to=OUT_DIR / "layer1B.png")


    # Layer 0: basic, defaults
    sim0 = run_basic()
    describe_sim(sim0)
    results.append(summarise(sim0, "0: basic, defaults"))
    plot_basic(sim0, "Layer 0: basic, defaults", save_to=OUT_DIR / "layer0.png")

    # Layer 1: + known vaccination (county coverage: 65% first dose by 30 Jun)
    check_vaccination()                                   # does the sim start with the right coverage?
    sim1 = run_basic(vaccinate=True)
    describe_sim(sim1)
    results.append(summarise(sim1, "1: + vaccination"))
    plot_basic(sim1, "Layer 1: + vaccination", save_to=OUT_DIR / "layer1.png")

    table = pd.DataFrame(results)
    print(table.to_string(index=False))
    table.to_csv(OUT_DIR / "layer_results.csv", index=False)
    print("Saved", OUT_DIR / "layer_results.csv")
    '''
    # Layer 0: basic, defaults
    #sim0 = run_basic()
    #describe_sim(sim0)
    #results.append(summarise(sim0, "0: basic, defaults"))

    # Table first, so a plotting error cannot lose the results
    #table = pd.DataFrame(results)
    #print(table.to_string(index=False))
    #table.to_csv(OUT_DIR / "layer_results.csv", index=False)
    #print("Saved", OUT_DIR / "layer_results.csv")

    #plot_basic(sim0, "Layer 0: basic, defaults", save_to=OUT_DIR / "layer0.png")
    #print("Saved", OUT_DIR / "layer0.png")