import matplotlib
matplotlib.use("Agg")                      # must come before importing pyplot
import numpy as np, pandas as pd, matplotlib.pyplot as plt, covasim as cv


#from modeling.modeling_helper_functions import extract_wastewater_regression_data, make_vaccination, make_variants

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))   # repo root: script -> Delta_wave... -> modeling -> repo

from modeling.modeling_helper_functions import extract_wastewater_regression_data, make_vaccination, make_variants

#SAVE
from pathlib import Path
OUT_DIR = Path("sim_results_basic"); OUT_DIR.mkdir(exist_ok=True)

def plot_basic(sim, title="", save_to=None):
    inf, dia = weekly(sim, "new_infections"), weekly(sim, "new_diagnoses")
    d = extract_wastewater_regression_data(sim)
    d["date"] = pd.to_datetime(d["date"])
    ww_sim = d.set_index("date")["wastewater"]
    ww_real = (df_pl_ww[(df_pl_ww["Sample_Date"] >= START_DAY) & (df_pl_ww["Sample_Date"] <= END_DAY)]
               .set_index("Sample_Date")["ww_copies_per_L"].sort_index())
    scale = ww_real.mean() / ww_sim.reindex(ww_real.index, method="nearest").mean()   # one constant: units are arbitrary

    fig, ax = plt.subplots(2, 1, figsize=(11, 8), sharex=True)
    ax[0].plot(wk["week_mid"], inf, "o-", color="tab:gray", label="sim TRUE infections")
    ax[0].plot(wk["week_mid"], dia, "o-", color="tab:green", label="sim diagnoses")
    ax[0].plot(wk["week_mid"], real, "o-", color="#D95F02", lw=2.5, label="REAL reported cases")
    ax[0].set_yscale("log"); ax[0].set_ylabel("per week (log)"); ax[0].legend(); ax[0].set_title(title)
    ax[1].plot(ww_real.index, ww_real.values, ".", ms=8, color="tab:blue", label="REAL wastewater")
    ax[1].plot(ww_sim.index, ww_sim * scale, color="tab:green", label="sim wastewater (one scale constant)")
    ax[1].set_ylabel("copies / L"); ax[1].legend()
    plt.tight_layout(); plt.show()
    if save_to:
        plt.savefig(save_to, dpi=150)
    plt.close()

def run_basic(beta=0.016, n_seed=100, vaccinate=False, symp_prob=0.1, seed=0):
    """One run. Each argument is one 'layer' you can switch or change."""
    variants, n_wild = make_variants(n_seed)
    interventions = [cv.test_prob(symp_prob=symp_prob, asymp_prob=0.005, test_delay=2)]
    if vaccinate:
        interventions += make_vaccination(120, 0.60, pl_share_one_dose_0630, pl_uptake_unvacc)

    START_DAY = pd.Timestamp("2021-06-28")
    END_DAY   = pd.Timestamp("2021-11-30")
    START_STR, END_STR = str(START_DAY.date()), str(END_DAY.date()) 

    sim = cv.Sim(pop_size=POP_SIZE, pop_scale=POP_SCALE, pop_type="hybrid", location="usa",
                 beta=beta, pop_infected=n_wild, n_imports=0,
                 start_day=START_DAY, end_day=END_DAY,
                 variants=variants, use_waning=True,
                 interventions=interventions, rand_seed=seed, verbose=0)
    sim.run()
    return sim

def weekly(sim, key):
    """Sum a daily sim result over the real reporting weeks."""
    s = pd.Series(sim.results[key].values, index=pd.to_datetime(sim.datevec))
    return np.array([s.loc[a:b].sum() for a, b in zip(wk["week_start"], wk["week_end"])])

def growth(y, lo=2, hi=7):
    """Growth per day over weeks 3-7 (skips the start-up weeks)."""
    t = np.arange(len(y)) * 7.0
    return np.polyfit(t[lo:hi], np.log(np.asarray(y[lo:hi], float) + 1), 1)[0]

def summarise(sim, label):
    inf, dia = weekly(sim, "new_infections"), weekly(sim, "new_diagnoses")
    return dict(layer=label,
                growth_per_day=round(growth(inf), 3),
                peak_week=int(inf.argmax()) + 1,
                peak_infections_per_week=int(inf.max()),
                peak_diagnoses_per_week=int(dia.max()),
                share_infected=round(sim.results["cum_infections"].values[-1] / CATCHMENT_POP, 2))


#******************
#DATA
#*****************
#PARAMETERS
# --- County figures (source: county updates 30 Jun and 1 Sep 2021) ---
POP_SIZE      = 50000
CATCHMENT_POP = 2116170                     # 2,116,170
POP_SCALE     = CATCHMENT_POP / POP_SIZE

POP_COUNTY       = 3347827
POP_COUNTY_12P   = 2802581
POP_PL           = 2116170 

#VARIANT SHARE
VARIANT_SHARE_PL = {"delta": 0.78,
                 "alpha": 0.12,
                   "gamma": 0.06,
                     "wild_other": 0.04}

# County head counts, residents 12+
county_n_one_dose_0630 = 2190000   
county_n_full_0630   = 1870000
county_n_one_dose_0901 = 2407528
county_n_full_0901   = 2112936


# Share of ALL residents (Covasim needs everyone, not just 12+).
# ASSUMPTION: Point Loma coverage = county coverage (no zip-level data used yet)
pl_share_one_dose_0630 = county_n_one_dose_0630 / POP_COUNTY
print(f"Share of one dose 06/30: {pl_share_one_dose_0630:.3f}")

pl_share_full_0630     = county_n_full_0630     / POP_COUNTY
print(f"Share of fully vaccinated 06/30: {pl_share_full_0630:.3f}")

pl_share_one_dose_0901 = county_n_one_dose_0901 / POP_COUNTY
print(f"Share of one dose 09/01: {pl_share_one_dose_0901:.3f}")

pl_share_full_0901     = county_n_full_0901     / POP_COUNTY
print(f"Share of fully vaccinated 09/01: {pl_share_full_0901:.3f}")

# Uptake among the unvaccinated: fraction of people with no dose on 30 Jun who had a first dose by 1 Sep
#   numerator   = rise in the share of ALL residents with >=1 dose (0.719 - 0.654 = 0.065)
#   denominator = share still unvaccinated on 30 Jun (1 - 0.654 = 0.346)
pl_uptake_unvacc = (pl_share_one_dose_0901 - pl_share_one_dose_0630) / (1 - pl_share_one_dose_0630)
print(pl_uptake_unvacc)   # about 0.188

#*********************************************************************
# Real weekly cases in the Delta window

# ---- near the top, with the other constants ----
START_DAY = pd.Timestamp("2021-06-28")      # sim start
END_DAY   = pd.Timestamp("2021-11-30")      # last day of the window
DATA_FOLDER = Path.home() / "GitHub" / "covasim_fit_data"
OUT_DIR = Path("sim_results_basic"); OUT_DIR.mkdir(exist_ok=True)

# ---- all the function definitions stay here ----

if __name__ == "__main__":
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

    # Run
    results = [dict(layer="REAL DATA", growth_per_day=round(growth(real), 3), peak_week=int(real.argmax()) + 1,
                    peak_infections_per_week=None, peak_diagnoses_per_week=int(real.max()), share_infected=None)]

    sim0 = run_basic()
    results.append(summarise(sim0, "0: basic, defaults"))
    plot_basic(sim0, "Layer 0", save_to=OUT_DIR / "layer0.png")

    table = pd.DataFrame(results)
    print(table.to_string(index=False))
    table.to_csv(OUT_DIR / "layer_results.csv", index=False)
    print("Saved", OUT_DIR / "layer_results.csv")

