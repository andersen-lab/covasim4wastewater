#MAKE VARIANTS
"""
Extract daily (burden, cases, wastewater) time series from a Covasim
simulation, ready for the cases-only / wastewater-only / both linear-model
comparison.

Mapping used:
    burden     -> new_infections   (true, unobserved disease burden)
    cases      -> new_diagnoses    (observed via testing/surveillance)
    wastewater -> sum of viral_shedding_hist across agents, per day

IMPORTANT: `new_diagnoses` will be all zeros unless the sim includes a
testing intervention (e.g. cv.test_prob or cv.test_num). Add one if you
want non-trivial "cases" data.
"""
import pandas as pd
import numpy as np
import covasim as cv

#VARIANT SHARE
VARIANT_SHARE_PL = {"delta": 0.78,
                 "alpha": 0.12,
                   "gamma": 0.06,
                     "wild_other": 0.04}

def extract_wastewater_regression_data(sim) -> pd.DataFrame:
    """
    Parameters
    ----------
    sim : covasim.Sim
        A Sim object that has already been run (sim.run()).

    Returns
    -------
    pd.DataFrame with columns: day, date, burden, cases, wastewater
    """
    results = sim.results

    # --- True burden ------------------------------------------------------
    burden = np.asarray(results['new_infections'].values, dtype=float)

    # --- Case data ----------------------------------------------------------
    if 'new_diagnoses' not in results:
        raise KeyError("sim.results has no 'new_diagnoses' — did the sim run?")
    cases = np.asarray(results['new_diagnoses'].values, dtype=float)
    if cases.sum() == 0:
        print("Warning: 'cases' (new_diagnoses) is all zero. "
              "Add a testing intervention, e.g. cv.test_prob(...), to the sim.")

    # --- Wastewater signal ------------------------------------------------
    viral_shedding_hist = np.asarray(results['viral_shedding_hist'])  # (n_people, n_days)
    wastewater = viral_shedding_hist.sum(axis=0)

    n = len(burden)
    df = pd.DataFrame({
        'day': np.arange(n),
        'date': [sim.date(d) for d in range(n)],
        'burden': burden,
        'cases': cases,
        'wastewater': wastewater[:n],
    })
    return df

def make_variants(pop_infected):
    """Split one total number of starting infections into the four variant groups."""

    # Step 1: how many agents each variant gets (share x total, rounded to whole agents)
    n_delta      = int(round(VARIANT_SHARE_PL["delta"]      * pop_infected))   # 0.78 x 100 = 78
    n_alpha      = int(round(VARIANT_SHARE_PL["alpha"]      * pop_infected))   # 0.12 x 100 = 12
    n_gamma      = int(round(VARIANT_SHARE_PL["gamma"]      * pop_infected))   # 0.06 x 100 = 6
    n_wild_other = int(round(VARIANT_SHARE_PL["wild_other"] * pop_infected))   # 0.04 x 100 = 4

    # Step 2: rounding can leave the total 1-2 agents off, so wild/other absorbs the difference
    n_wild_other = pop_infected - (n_delta + n_alpha + n_gamma)                # 100 - 96 = 4

    # Step 3: one cv.variant object per variant, all imported on day 0
    delta = cv.variant("delta", days=0, n_imports=n_delta)    # e.g. n_imports=78
    alpha = cv.variant("alpha", days=0, n_imports=n_alpha)    # e.g. n_imports=12
    gamma = cv.variant("gamma", days=0, n_imports=n_gamma)    # e.g. n_imports=6

    # Wild/other has no cv.variant object: it is the sim's own pop_infected (e.g. 4)
    variants = [alpha, gamma, delta]

    # Print the table so the split is visible every time it runs
    print(f"total starting infections = {pop_infected}")
    print(f"  delta       : share {VARIANT_SHARE_PL['delta']:.2f} -> {n_delta:>4}  -> cv.variant('delta', days=0, n_imports={n_delta})")
    print(f"  alpha       : share {VARIANT_SHARE_PL['alpha']:.2f} -> {n_alpha:>4}  -> cv.variant('alpha', days=0, n_imports={n_alpha})")
    print(f"  gamma       : share {VARIANT_SHARE_PL['gamma']:.2f} -> {n_gamma:>4}  -> cv.variant('gamma', days=0, n_imports={n_gamma})")
    print(f"  wild/other  : share {VARIANT_SHARE_PL['wild_other']:.2f} -> {n_wild_other:>4}  -> cv.Sim(pop_infected={n_wild_other})")

    return variants, n_wild_other


def make_vaccination(vacc_start_days, pfizer_share,
                     share_before, #=pl_share_one_dose_0630,
                     uptake_unvacc, #=pl_uptake_unvacc,
                     days_during=65):
    """Historical first-dose campaigns for Point Loma.

    vacc_start_days : days before 28 Jun when the campaign begins (RANGE 75-170, timing unknown)
    pfizer_share    : share of doses that are Pfizer (RANGE 0.50-0.65); the rest is Moderna
    share_before    : share of ALL residents with >=1 dose on 30 Jun (county, 0.654)
    uptake_unvacc   : fraction of the unvaccinated first-dosed 28 Jun -> 1 Sep (0.188)
    days_during     : length of that period in days (65)
    """
    days_before = np.arange(-int(vacc_start_days), 0)
    days_after  = np.arange(0, days_during)

    campaigns = []
    for brand, share in (("pfizer", pfizer_share), ("moderna", 1 - pfizer_share)):
        p_before = cv.historical_vaccinate_prob.estimate_prob(len(days_before), share_before * share)
        p_during = cv.historical_vaccinate_prob.estimate_prob(len(days_after), uptake_unvacc * share)
        campaigns.append(cv.historical_vaccinate_prob(vaccine=brand, days=days_before, prob=float(p_before)))
        campaigns.append(cv.historical_vaccinate_prob(vaccine=brand, days=days_after, prob=float(p_during)))
    return campaigns