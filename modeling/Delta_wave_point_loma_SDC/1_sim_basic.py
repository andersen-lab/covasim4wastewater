import matplotlib
matplotlib.use("Agg")                       # no display on the server; the figure is saved as a file
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import covasim as cv

# Helper functions from your repo (vaccination campaigns, wastewater from viral shedding)
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from modeling.modeling_helper_functions import extract_wastewater_regression_data, make_vaccination

# =====================================================================
# SETTINGS: change these, then run the file
# =====================================================================
SEED_TOTAL = 3000            # infected agents on day 0
SHARES = {"delta": 0.78, "alpha": 0.12, "gamma": 0.06, "wild_other": 0.04}   # split of the seeded agents
BETA = 0.016                # Covasim default
VACCINATE = True            # known county first-dose coverage
VACC_START_DAYS = 120       # vaccination campaign starts this many days before 28 Jun (assumed)
PFIZER_SHARE = 0.60         # the rest is Moderna (assumed)
IMMUNE_FRAC = 0.0           # share of agents made unable to be infected (stand-in for earlier infections)
SYMP_PROB = 0.10            # daily test probability, symptomatic
ASYMP_PROB = 0.005          # daily test probability, asymptomatic
TEST_DELAY = 2              # days from test to diagnosis
RAND_SEED = 0

# =====================================================================
# FIXED INPUTS (data)
# =====================================================================
POP_SIZE = 50000
CATCHMENT_POP = 2116170
POP_SCALE = CATCHMENT_POP / POP_SIZE
START_STR, END_STR = "2021-06-28", "2021-11-30"

POP_COUNTY = 3347827
share_one_dose_0630 = 2190000 / POP_COUNTY
share_one_dose_0901 = 2407528 / POP_COUNTY
uptake_unvacc = (share_one_dose_0901 - share_one_dose_0630) / (1 - share_one_dose_0630)

DATA_FOLDER = Path.home() / "GitHub" / "covasim_fit_data"
OUT_DIR = Path(__file__).resolve().parent / "sim_results_simple"
OUT_DIR.mkdir(exist_ok=True)

# =====================================================================
# REAL DATA
# =====================================================================
start, end = pd.Timestamp(START_STR), pd.Timestamp(END_STR)

weekly_cases = pd.read_pickle(DATA_FOLDER / "df_cases_weekly_pl.pkl")
wk = weekly_cases[(weekly_cases["week_start"] >= start) & (weekly_cases["week_end"] <= end)].reset_index(drop=True)
real_cases = wk["weekly_total"].to_numpy(float)

ww = pd.read_pickle(DATA_FOLDER / "pl_wastewater.pkl")
ww = ww.rename(columns={"Mean viral gene copies/L": "ww_copies_per_L"})
ww["Sample_Date"] = pd.to_datetime(ww["Sample_Date"], format="%m/%d/%y")
real_ww = ww[(ww["Sample_Date"] >= start) & (ww["Sample_Date"] <= end)].set_index("Sample_Date")["ww_copies_per_L"].sort_index()

# =====================================================================
# THE SIMULATION
# =====================================================================
# Split the seeded agents by variant (wild/other is whatever is left over)
n_delta = int(round(SHARES["delta"] * SEED_TOTAL))
n_alpha = int(round(SHARES["alpha"] * SEED_TOTAL))
n_gamma = int(round(SHARES["gamma"] * SEED_TOTAL))
n_wild = SEED_TOTAL - n_delta - n_alpha - n_gamma
variants = [cv.variant("delta", days=0, n_imports=n_delta),
            cv.variant("alpha", days=0, n_imports=n_alpha),
            cv.variant("gamma", days=0, n_imports=n_gamma)]

interventions = [cv.test_prob(symp_prob=SYMP_PROB, asymp_prob=ASYMP_PROB, test_delay=TEST_DELAY)]
if VACCINATE:
    interventions += make_vaccination(VACC_START_DAYS, PFIZER_SHARE, share_one_dose_0630, uptake_unvacc)

sim = cv.Sim(
    pop_size=POP_SIZE,
    pop_scale=POP_SCALE,
    pop_type="hybrid",
    location="usa",
    beta=BETA,
    pop_infected=n_wild,
    n_imports=0,
    start_day=START_STR,
    end_day=END_STR,
    variants=variants,
    use_waning=True,
    interventions=interventions,
    rand_seed=RAND_SEED,
    verbose=0,
)
sim.initialize()
if IMMUNE_FRAC > 0:
    rng = np.random.default_rng(RAND_SEED + 1000)
    sim.people.rel_sus[rng.random(len(sim.people)) < IMMUNE_FRAC] = 0.0
sim.run()

# =====================================================================
# RESULTS
# =====================================================================
def weekly(key):
    s = pd.Series(sim.results[key].values, index=pd.to_datetime(sim.datevec))
    return np.array([s.loc[a:b].sum() for a, b in zip(wk["week_start"], wk["week_end"])])

infections, diagnoses = weekly("new_infections"), weekly("new_diagnoses")

d = extract_wastewater_regression_data(sim)
d["date"] = pd.to_datetime(d["date"])
ww_sim = d.set_index("date")["wastewater"]
scale = real_ww.mean() / ww_sim.reindex(real_ww.index, method="nearest").mean()   # one constant: units are arbitrary

cum_inf, cum_dia = sim.results["cum_infections"].values[-1], sim.results["cum_diagnoses"].values[-1]
print(f"seeded {SEED_TOTAL} (delta {n_delta}, alpha {n_alpha}, gamma {n_gamma}, wild/other {n_wild}), beta {BETA}, "
      f"vaccination {'ON' if VACCINATE else 'OFF'}, immune {IMMUNE_FRAC:.0%}")
print(f"peak infections {int(infections.max()):,} (week {infections.argmax() + 1})   "
      f"peak diagnoses {int(diagnoses.max()):,}   REAL peak {int(real_cases.max()):,} (week {real_cases.argmax() + 1})")
print(f"share of catchment infected {cum_inf / CATCHMENT_POP:.0%}   share of infections diagnosed {cum_dia / max(cum_inf, 1):.0%}")

# =====================================================================
# PLOT
# =====================================================================
fig, ax = plt.subplots(3, 1, figsize=(11, 11), sharex=True)
ax[0].plot(wk["week_mid"], infections, "o-", color="tab:gray", label="sim TRUE infections")
ax[0].set_ylabel("infections per week")
ax[1].plot(wk["week_mid"], diagnoses, "o-", color="tab:green", label="sim diagnoses")
ax[1].plot(wk["week_mid"], real_cases, "o-", color="#D95F02", lw=2.5, label="REAL reported cases")
ax[1].set_ylabel("cases per week")
ax[2].plot(real_ww.index, real_ww.values, ".", ms=8, color="tab:blue", label="REAL wastewater")
ax[2].plot(ww_sim.index, ww_sim * scale, color="tab:green", label="sim wastewater (one scale constant)")
ax[2].set_ylabel("copies / L")
ax[2].set_xlim(start, end)
for a in ax:
    a.legend(fontsize=9)
    a.ticklabel_format(axis="y", style="plain")
fig.suptitle(f"seeded={SEED_TOTAL}  beta={BETA}  vaccination={'ON' if VACCINATE else 'OFF'}  immune={IMMUNE_FRAC:.0%}  "
             f"symp_prob={SYMP_PROB}  seed={RAND_SEED}", fontsize=11)
fig.tight_layout()
fig.savefig(OUT_DIR / "sim_simple.png", dpi=150)
print("Saved", OUT_DIR / "sim_simple.png")