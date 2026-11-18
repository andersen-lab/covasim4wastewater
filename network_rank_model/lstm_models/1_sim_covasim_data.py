'''
8a. MAKE THE COVASIM RUNS (the slow part)
==========================================
Runs Covasim once for every seed and saves what the LSTM needs from each run. This script needs Covasim only (no torch), so it
can be started on its own, and the LSTM script (8b_train_lstm_stages.py) can be run later, even in a different Python environment.

WHAT IS SAVED FOR EVERY RUN (one file, run_seed_<seed>.npz, arrays with N_DAYS + 1 = 181 values, one per day)
  true_infections  infections per day (the truth), counted from Covasim's infection log
  reported_cases   people diagnosed per day (Covasim's new_diagnoses)
  load_gc_per_day  gene copies per day in the wastewater: every agent's stool shedding (gc per mL of stool) added up,
                   times the mL of stool per person per day
  n_agents         the number of people in the run (needed later to turn the load into a concentration)

ASSUMPTIONS
  1. The truth is Covasim's infection log: infections per day, including the 10 seed infections and any reinfections.
  2. Each run has N_AGENTS = 20,000 people on the 'rank' contact network with the San Diego maps (smaller than the 50,000 used
     elsewhere, to make the runs faster). ASSUMED, NOT CHECKED: that 20,000-agent runs are comparable with 50,000-agent runs once
     the signals are expressed per person.
  3. Each run (seed) has its own transmission rate beta, drawn between 0.010 and 0.016 (the same value for the same seed), so the
     epidemics differ in size and speed. 10 initial infections, no imported cases, uneven infectiousness (superspreading,
     dispersion 0.45), as in run_covasim.py.
  4. Reported cases are Covasim's new diagnoses. Only symptomatic people are tested (10% a day), so only part of the infections is
     ever reported (about 30% in the earlier runs), and each case is reported some days after infection.
  5. Each agent sheds virus in stool along the shedding_hub curve (Wolfel et al. 2020, gamma fit, gene copies per mL of stool).
     Covasim places each agent's curve from their infectious date.
  6. Load per day (gene copies) = the agents' shedding added up x 128 mL of stool per person per day (Rose et al. 2015; 1 g of
     stool is taken as 1 mL).
  7. Only the whole population is used (one site, no sewersheds). The wastewater concentration and the sewage flow are worked
     out in the LSTM script, not here.

HOW TO RUN
    python 8a_make_covasim_runs.py                              the seeds FIRST_SEED to LAST_SEED set below
    python 8a_make_covasim_runs.py --first-seed 1 --last-seed 250     choose the seeds on the command line instead
  Each run is saved the first time it is made and never run twice, so if the script stops it carries on from the runs already
  done. Time ONE run first (--first-seed 1 --last-seed 1), then choose how many to make: 30 seeds is a first test (the LSTM
  script needs at least 10 runs that took off), 200 to 1000 the real thing. Every file is written to RUNS_FOLDER.

  SEVERAL COPIES AT ONCE: to use more of the computer, start the script in several terminals with DIFFERENT seed ranges, e.g.
    python 8a_make_covasim_runs.py --first-seed 1   --last-seed 250
    python 8a_make_covasim_runs.py --first-seed 251 --last-seed 500
  They write different files, so they do not clash. Each copy prints the summary of its own seeds. Every copy also uses several
  processor cores to build the contact network, and needs its own memory, so start 2 or 3, not 10, and check that the computer
  is not running out of memory. A run is written under a temporary name and renamed when it is complete, so a copy that is
  stopped half-way never leaves a broken file behind.
'''
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import covasim as cv


# =====================================================================================================================
# SETTINGS
# =====================================================================================================================
SCRIPT_FOLDER = Path(__file__).resolve().parent            # the folder this script is in
OUTPUT_FOLDER = SCRIPT_FOLDER / 'lstm_stages_output'       # shared with the LSTM script
RUNS_FOLDER   = OUTPUT_FOLDER / 'covasim_runs'             # one file per Covasim run
REPO_ROOT     = SCRIPT_FOLDER.parents[1]                   # the root of the repository (same layout as run_covasim.py)

N_DAYS   = 180             # Covasim days. Each run has N_DAYS + 1 values, because Covasim also counts day 0
N_AGENTS = 20000           # people per run. Smaller than 50,000 to make the runs faster (the signals are scaled per person)
STOOL_ML_PER_PERSON_PER_DAY = 128    # mL of stool per person per day (Rose et al. 2015; 1 g = 1 mL)

FIRST_SEED = 1             # one Covasim run per seed, from FIRST_SEED to LAST_SEED (both included).
LAST_SEED  = 30            # 30 seeds is a first test; use 200 to 1000 for the real thing. The command line can override both.

# Covasim settings for the runs (same as run_covasim.py)
N_INITIAL_INFECTIONS           = 10      # people infected on day 0
INFECTIOUSNESS_DISPERSION      = 0.45    # how uneven infectiousness is between people (superspreading)
DAILY_TEST_PROB_IF_SYMPTOMATIC = 0.1     # each day a symptomatic person has this chance of being tested (and so reported)
SAN_DIEGO_DATA_FOLDER     = REPO_ROOT / 'docs' / 'tutorials' / 'san-diego-data'
POPULATION_DENSITY_RASTER = SAN_DIEGO_DATA_FOLDER / 'san_diego_county_ppp_2020.tif'       # where people live
ZIP_CODE_BOUNDARIES       = SAN_DIEGO_DATA_FOLDER / 'sandiego_zip_boundaries.geojson'     # one region per ZIP code
SEWERSHED_BOUNDARIES      = SAN_DIEGO_DATA_FOLDER / 'sandiego_sewersheds.geojson'         # one catchment per treatment plant


# =====================================================================================================================
# THE COVASIM RUNS
# =====================================================================================================================
def run_covasim_for_training(seed, beta):
    """
    Run Covasim once and keep only what the LSTM needs.

    Inputs:
        seed  the random seed of the run (a different seed gives a different epidemic)
        beta  the chance of transmission per infectious contact (a higher beta gives a bigger, faster epidemic)

    Returns a dictionary (arrays have N_DAYS + 1 values, one per day):
        true_infections  new infections on each day, counted from Covasim's infection log
        reported_cases   people diagnosed on each day (Covasim's new_diagnoses)
        load_gc_per_day  gene copies per day in the wastewater: every agent's stool shedding (gc/mL) added up,
                         times the mL of stool per person per day
        n_agents         the number of people (needed later to turn the load into a concentration)
    """
    # Only symptomatic people are tested, with a 10% chance each day. Without a testing intervention Covasim reports no cases.
    testing = cv.test_prob(symp_prob=DAILY_TEST_PROB_IF_SYMPTOMATIC, asymp_prob=0.0, start_day=0)
    sim = cv.Sim(pop_type='rank', pop_size=N_AGENTS, pop_infected=N_INITIAL_INFECTIONS, n_imports=0,
                 beta=beta, n_days=N_DAYS, rand_seed=seed, verbose=0,
                 beta_dist=dict(dist='neg_binomial', par1=1.0, par2=INFECTIOUSNESS_DISPERSION, step=0.01),   # superspreading
                 interventions=testing, density_raster=str(POPULATION_DENSITY_RASTER),
                 boundary_shapefile=str(ZIP_CODE_BOUNDARIES), sewershed_file=str(SEWERSHED_BOUNDARIES))
    sim.initialize(use_workers=True)                   # builds the population and its contact network (use_workers only makes it faster)
    sim.run()

    # The truth: every infection event in the log has the day it happened. Counting events per day gives infections per day.
    infection_days = np.array([event['date'] for event in sim.people.infection_log]).astype(int)
    # Each agent's stool concentration on each day (a table of agents x days, in gene copies per mL of stool)
    viral_shedding_hist = np.asarray(sim.results['viral_shedding_hist'], dtype=float)

    return {'true_infections': np.bincount(infection_days, minlength=N_DAYS + 1)[:N_DAYS + 1],         # a count for every day, even days with 0
            'reported_cases': np.array(sim.results['new_diagnoses'].values).astype(int),
            'load_gc_per_day': viral_shedding_hist.sum(axis=0) * STOOL_ML_PER_PERSON_PER_DAY,         # add up the agents, then gc/mL x mL = gc
            'n_agents': N_AGENTS}


def get_covasim_run(seed):
    """
    Get one Covasim run. It is run the first time and saved to disk, so it is never run twice. If the script is stopped
    and started again, the runs already done are loaded in a second.

    Input:    seed  the random seed of the run. Each seed also gets its own transmission rate beta (0.010 to 0.016).
    Returns:  the run dictionary described above run_covasim_for_training
    """
    path = RUNS_FOLDER / f'run_seed_{seed}.npz'
    if not path.exists():
        print(f'seed {seed}: running Covasim ...', flush=True)
        # A separate random generator seeded with this seed: the same seed always gives the same beta
        beta = np.random.RandomState(seed).uniform(0.010, 0.016)
        # Saved under a temporary name first, then renamed: a run that is stopped half-way never leaves a broken file behind
        unfinished_path = RUNS_FOLDER / f'unfinished_seed_{seed}.npz'
        np.savez(unfinished_path, **run_covasim_for_training(seed, beta))
        unfinished_path.replace(path)
    saved = np.load(path)
    return {'true_infections': saved['true_infections'], 'reported_cases': saved['reported_cases'],
            'load_gc_per_day': saved['load_gc_per_day'], 'n_agents': int(saved['n_agents'])}      # int(): the file stores it as an array


def get_seeds():
    """
    The seeds to make runs for: FIRST_SEED to LAST_SEED from the settings, unless --first-seed and --last-seed are given on the
    command line (so several copies of the script can each take a different range).

    Returns: a range of seeds, both ends included
    """
    parser = argparse.ArgumentParser(description='Make Covasim runs for the LSTM (each seed is only run once).')
    parser.add_argument('--first-seed', type=int, default=FIRST_SEED, help='the first seed to run')
    parser.add_argument('--last-seed', type=int, default=LAST_SEED, help='the last seed to run (included)')
    arguments = parser.parse_args()
    return range(arguments.first_seed, arguments.last_seed + 1)


def main():
    """Make (or load) the run for every seed, then print a one-line summary of each, so runs that did not take off are easy to see."""
    RUNS_FOLDER.mkdir(parents=True, exist_ok=True)
    seeds = get_seeds()
    print(f'seeds {seeds.start} to {seeds.stop - 1} ({len(seeds)} runs)')
    runs = {seed: get_covasim_run(seed) for seed in seeds}

    summary = pd.DataFrame([{'seed': seed,
                             'share_infected': run['true_infections'].sum() / run['n_agents'],        # the LSTM script drops runs below 0.01
                             'peak_day': int(run['true_infections'].argmax()),
                             'peak_infections': int(run['true_infections'].max()),
                             'share_reported': run['reported_cases'].sum() / max(run['true_infections'].sum(), 1)}
                            for seed, run in runs.items()])
    print('\nThe runs (share_infected below 0.01 = the epidemic did not take off and will not be used):')
    print(summary.round(3).to_string(index=False))
    print(f"\n{(summary['share_infected'] >= 0.01).sum()} of {len(summary)} runs took off. Saved in {RUNS_FOLDER}")


if __name__ == '__main__':
    main()