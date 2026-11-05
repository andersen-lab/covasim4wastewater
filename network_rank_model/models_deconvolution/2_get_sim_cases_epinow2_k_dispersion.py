'''
Aim: for each chosen pair of dispersions (k1 = number of friends, k2 = infectiousness), run the rank-population Covasim
until an epidemic takes off, and export everything we need to test whether EpiNow2 (in R) recovers the true infections
from the reported cases alone. One folder of csv files per scenario, so the R script can be pointed at any of them.

THE TWO DISPERSIONS (each is a negative binomial with a fixed mean; smaller k = more uneven)
    k1 : the number of friends per agent. None = Poisson. A small k gives hubs and many agents with few friends.
    k2 : infectiousness (beta_dist). Covasim default 0.45. A small k gives super-spreaders.

Outputs, in simulated_cases/<scenario>/ next to this script. The scenario name (e.g. k1_Poisson_k2_0.45) is in the
folder name AND at the end of every file name, so a file still says where it came from when it is copied elsewhere:
    covasim_cases_<scenario>.csv         one row per simulated day: date, confirm, true_infections, and other daily results
    covasim_delays_<scenario>.csv        one row per infected agent: the day each event happened, plus the delays between them
    covasim_transmission_<scenario>.csv  one row per infection event (who infected whom): used for the generation time
    covasim_run_info_<scenario>.csv      the seed and settings the run used
and one table over all scenarios:
    simulated_cases/scenarios_summary.csv

Run from anywhere:  python 2_get_simulated_cases_k.py       (try one scenario first: each run builds the network again)
'''
import time
from pathlib import Path
import numpy as np
import pandas as pd
import covasim as cv

# ---------------------------------------------------------------------------
# Data and checks (same layout as check_two_dispersions_outbreaks.py)
# ---------------------------------------------------------------------------
HERE = Path(__file__).resolve().parent            # network_rank_model/<folder>
REPO = HERE.parents[1]                            # the repo root
DATA       = REPO/'docs'/'tutorials'/'san-diego-data'
TIF        = DATA/'san_diego_county_ppp_2020.tif'            # people per ~100 m cell: where agents are placed
BOUNDARY   = DATA/'sandiego_zip_boundaries.geojson'          # one region per ZIP code
SEWERSHEDS = DATA/'sandiego_sewersheds.geojson'              # one catchment per treatment plant
OUT_ROOT   = HERE/'simulated_cases'

for f in (TIF, BOUNDARY, SEWERSHEDS):
    if not f.exists():
        raise FileNotFoundError(f'{f} not found. Run  git pull  on the network_rank_model branch to get docs/tutorials/san-diego-data.')

# ---------------------------------------------------------------------------
# Settings
# ---------------------------------------------------------------------------
N          = 50000     # agents
N_DAYS     = 180       # length of the epidemic
SEED       = 1         # first seed to try. Every scenario starts from the same seed, so the agents match and mostly only k differs
MAX_TRIES  = 20        # the most seeds to try per scenario before giving up
TOOK_OFF   = 0.01      # an epidemic counts as taken off if at least this share of the agents were infected (as in check_two_dispersions_outbreaks.py)
BETA       = 0.012     # transmission per contact (the Covasim default is 0.016)
N_INITIAL  = 10        # agents infected on day 0
TEST_PROB  = 0.1       # each day, the chance that a symptomatic agent is tested (and diagnosed); asymptomatic agents are never tested

# The scenarios to run, as (k1, k2) pairs: k1 = friend-count dispersion (None = Poisson), k2 = infectiousness dispersion.
# The baseline (None, 0.45) with the settings above is the run already analysed with EpiNow2, and it gives the same epidemic
# every time (same seed, same settings), so it is switched off here. Each scenario costs at least one network build, and more
# when the epidemic dies out (more likely for small k2): try one scenario first by commenting out the others.
SCENARIOS = [
    # (None, 0.45),  # baseline: Poisson friends, Covasim default infectiousness (already run)
    #(None, 0.1),     # strong super-spreading
    #(None, 10.0),    # almost no super-spreading
    (0.1,  0.45),    # hubs in the friend network
]


def contact_name(k):
    return 'Poisson' if k is None else f'{k:g}'


def scenario_tag(k_contacts, k_infect):
    return f'k1_{contact_name(k_contacts)}_k2_{k_infect:g}'


def run_one(k_contacts, k_infect, seed):
    ''' Build the rank population with the two dispersions, run the epidemic, and return the sim and each agent's friend count '''
    # Covasim only produces diagnoses if a testing intervention is added (the default sim has none, so
    # new_diagnoses would be all zeros). This is Covasim's built-in testing.
    testing = cv.test_prob(symp_prob=TEST_PROB, asymp_prob=0.0, start_day=0)

    sim = cv.Sim(pop_type='rank', pop_size=N, pop_infected=N_INITIAL, n_imports=0, beta=BETA, n_days=N_DAYS, rand_seed=seed, verbose=0,
                 beta_dist=dict(dist='neg_binomial', par1=1.0, par2=k_infect, step=0.01),   # infectiousness dispersion: all four entries are needed
                 interventions=testing,
                 density_raster=str(TIF), boundary_shapefile=str(BOUNDARY), sewershed_file=str(SEWERSHEDS))
    extra = dict(use_workers=True)                 # the workers change the speed of building the network, not the network
    if k_contacts is not None:
        extra['dispersion'] = k_contacts           # friend-count dispersion: passed through sim.initialize to make_rank_based_contacts
    sim.initialize(**extra)
    layer = sim.people.contacts['a']
    degree = np.bincount(np.concatenate([layer['p1'], layer['p2']]), minlength=N)
    sim.run()
    return sim, degree


def find_epidemic(k_contacts, k_infect):
    ''' Try SEED, SEED+1, ... and keep the FIRST run that took off.
    With few initial infections and overdispersed transmission, many epidemics die out in the first days (the N_INITIAL
    agents infect nobody), and EpiNow2 cannot be benchmarked on a few dozen infections. The rule is fixed in advance
    (not "the nicest-looking curve"), and the seed and the number of earlier seeds that died out are saved, so the run can
    be reproduced and the share that died out can be reported: it is a result in itself, and it rises as k2 falls. '''
    for tries in range(MAX_TRIES):
        seed = SEED + tries
        t0 = time.time()
        sim, degree = run_one(k_contacts, k_infect, seed)
        ever = int(np.sum(~np.isnan(np.array(sim.people.date_exposed))))      # agents infected at least once
        print(f'  seed {seed}: {ever:,} agents infected ({ever/N:.1%}), {time.time()-t0:.0f} s', flush=True)
        if ever >= TOOK_OFF*N:
            return sim, degree, seed, tries
    raise RuntimeError(f'No epidemic took off in {MAX_TRIES} seeds for k1 = {contact_name(k_contacts)}, k2 = {k_infect:g}. '
                       'Raise N_INITIAL (e.g. 20-50) or BETA, or MAX_TRIES.')


def export_scenario(sim, degree, seed, tries, k_contacts, k_infect, out_dir):
    ''' Write the csv files for one run and return a one-row summary '''
    out_dir.mkdir(parents=True, exist_ok=True)
    tag = scenario_tag(k_contacts, k_infect)          # goes at the end of every file name
    ppl = sim.people

    # -----------------------------------------------------------------------
    # File 1: covasim_cases_<scenario>.csv, the daily series (one row per simulated day, for the whole population)
    #   date             : calendar date of that simulated day (Covasim starts on 2020-03-01, so day 0 = 2020-03-01)
    #   confirm          : agents DIAGNOSED on that date = the "reported cases". This is the ONLY column EpiNow2 is given.
    #                      It is the report date, NOT the infection date: it comes after infection by the incubation
    #                      period plus the testing delay.
    #   true_infections  : agents INFECTED on that date (date of infection). The ground truth; EpiNow2 never sees it.
    # The other columns are extra daily results kept for later (e.g. other observation streams, or comparing EpiNow2's Rt
    # with the sim's r_eff). Any that this Covasim version does not have are skipped.
    # -----------------------------------------------------------------------
    df = pd.DataFrame({
        'date': pd.to_datetime(sim.datevec),
        'day': np.arange(len(sim.datevec)),
        'confirm': sim.results['new_diagnoses'].values.astype(int),
        'true_infections': sim.results['new_infections'].values.astype(int),
    })
    for key in ['new_symptomatic', 'new_infectious', 'new_tests', 'new_severe', 'new_critical', 'new_recoveries', 'new_deaths',
                'n_infectious', 'n_exposed', 'n_susceptible', 'cum_infections', 'cum_diagnoses', 'r_eff']:
        if key in sim.results:
            df[key] = np.array(sim.results[key].values)
    df.to_csv(out_dir/f'covasim_cases_{tag}.csv', index=False)

    # -----------------------------------------------------------------------
    # File 2: covasim_delays_<scenario>.csv, one row per agent who was ever infected
    # The date_ columns are SIMULATION DAYS (0 = the first day, i.e. 2020-03-01), not calendar dates.
    # NaN means the event never happened to that agent (e.g. no symptoms, never diagnosed).
    #   date_exposed     : day the agent was infected
    #   date_infectious  : day the agent became infectious (able to transmit)
    #   date_symptomatic : day symptoms started (NaN for asymptomatic agents)
    #   date_tested      : day the agent was tested (NaN if never)
    #   date_diagnosed   : day the agent was diagnosed (NaN if never tested positive)
    #   date_severe / date_critical / date_recovered / date_dead : later disease events (NaN if never)
    # The delay_ columns are the true delays in days, computed from the dates above (NaN if either date is missing):
    #   delay_latent                 = date_infectious  - date_exposed      latent period: infected but not yet infectious
    #   delay_incubation             = date_symptomatic - date_exposed      incubation period: infection -> symptoms (EpiNow2 input)
    #   delay_symptom_to_diagnosis   = date_diagnosed   - date_symptomatic  reporting delay: symptoms -> report (EpiNow2 input)
    #   delay_infection_to_diagnosis = date_diagnosed   - date_exposed      the full delay from infection to appearing in `confirm`
    # Agents infected on day 0 are the N_INITIAL seeds placed by pop_infected: infected by construction, not by transmission.
    # Caveat: date_exposed holds only an agent's most recent infection, so a reinfected agent's earlier infection is lost here
    # (reinfection is rare with default immunity; the transmission csv keeps every infection event).
    # Caveat: delays only exist for agents diagnosed before the run ended, so the observed delays are biased towards short
    # ones for agents infected late in the run.
    # -----------------------------------------------------------------------
    date_cols = ['date_exposed', 'date_infectious', 'date_symptomatic', 'date_tested', 'date_diagnosed',
                 'date_severe', 'date_critical', 'date_recovered', 'date_dead']
    delays = pd.DataFrame({'uid': np.arange(N)})
    for c in date_cols:
        if c in ppl.keys():
            delays[c] = np.array(ppl[c], dtype=float)
    delays['age'] = np.array(ppl.age)
    delays['sex'] = np.array(ppl.sex)
    delays['n_friends'] = degree                      # number of friends in the rank network
    delays['delay_latent'] = delays['date_infectious'] - delays['date_exposed']
    delays['delay_incubation'] = delays['date_symptomatic'] - delays['date_exposed']
    delays['delay_symptom_to_diagnosis'] = delays['date_diagnosed'] - delays['date_symptomatic']
    delays['delay_infection_to_diagnosis'] = delays['date_diagnosed'] - delays['date_exposed']
    delays = delays.dropna(subset=['date_exposed'])
    delays.to_csv(out_dir/f'covasim_delays_{tag}.csv', index=False)

    # -----------------------------------------------------------------------
    # File 3: covasim_transmission_<scenario>.csv, one row per infection event (who infected whom)
    # Covasim logs every transmission in sim.people.infection_log (source, target, date, layer). The generation time
    # needs this because it is a property of an infector-infectee PAIR, which the per-agent table above cannot give.
    #   source          : uid of the infector (NaN for the N_INITIAL seeds, who were not infected by anyone)
    #   target          : uid of the infectee
    #   date            : simulation day the infection happened
    #   layer           : which contact layer carried it (the rank network is layer 'a')
    #   generation_time = (infectee's infection day) - (infector's infection day), in days. EpiNow2 needs this
    #                     distribution for its renewal model. The infector's infection day is the latest one on or before this
    #                     event, taken from the log, so reinfection does not give negative values. NaN for seeds.
    #   source_infection_day, source_n_infections : the infector's infection day, and how many times they were infected
    #   serial_interval = (infectee's symptom onset) - (infector's symptom onset). NaN if either was asymptomatic.
    #   source_infectious_day = days after the INFECTOR became infectious that the transmission happened
    # -----------------------------------------------------------------------
    gt_mean = gt_min = np.nan
    log = getattr(ppl, 'infection_log', None)
    if log:
        trans = pd.DataFrame(log)
        trans['source'] = pd.to_numeric(trans['source'], errors='coerce')        # None for the seeds -> NaN
        src = trans['source'].to_numpy()
        has_src = ~np.isnan(src)
        s = np.where(has_src, src, 0).astype(int)
        tgt = trans['target'].to_numpy().astype(int)
        ev_date = trans['date'].to_numpy(dtype=float)
        # Every infection day of every agent, from the log itself. ppl.date_exposed only keeps an agent's LATEST infection,
        # so for a reinfected infector it would give a date AFTER the infection they caused (a negative generation time).
        inf_days = {}
        for t, d in zip(tgt, ev_date):
            inf_days.setdefault(int(t), []).append(d)

        def source_infection_day(u, d):
            prior = [x for x in inf_days.get(int(u), []) if x <= d]       # the infector's latest infection on or before day d
            return max(prior) if prior else np.nan

        src_inf = np.array([source_infection_day(u, d) if h else np.nan for u, d, h in zip(s, ev_date, has_src)])
        trans['source_infection_day'] = src_inf
        trans['source_n_infections'] = [len(inf_days.get(int(u), [])) if h else np.nan for u, h in zip(s, has_src)]  # >1 = reinfected infector
        trans['generation_time'] = ev_date - src_inf
        if 'date_symptomatic' in ppl.keys():
            sym = np.array(ppl.date_symptomatic, dtype=float)
            trans['serial_interval'] = np.where(has_src, sym[tgt] - sym[s], np.nan)      # affected by reinfection, so use with source_n_infections == 1
        if 'date_infectious' in ppl.keys():
            trans['source_infectious_day'] = np.where(has_src, ev_date - np.array(ppl.date_infectious, dtype=float)[s], np.nan)
        trans.to_csv(out_dir/f'covasim_transmission_{tag}.csv', index=False)
        gt_mean, gt_min = trans['generation_time'].mean(), trans['generation_time'].min()
        print(f"  generation time: mean {gt_mean:.2f} days, std {trans['generation_time'].std():.2f}, min {gt_min:.0f}, "
              f"from {trans['generation_time'].notna().sum():,} transmission pairs")
        if gt_min < 0:
            print('  WARNING: negative generation times, something is wrong with the infection log bookkeeping')
    else:
        print('  WARNING: no infection_log found on sim.people in this Covasim/fork, so the transmission csv was NOT written. '
              'Try sim.make_transtree() instead, or check how the rank fork stores transmissions.')

    # -----------------------------------------------------------------------
    # File 4: covasim_run_info_<scenario>.csv, the settings used, for reproducing the run
    # -----------------------------------------------------------------------
    pd.DataFrame({'seed': [seed], 'seeds_died_out_before': [tries], 'N': [N], 'N_DAYS': [N_DAYS], 'N_INITIAL': [N_INITIAL],
                  'BETA': [BETA], 'TEST_PROB': [TEST_PROB], 'K_CONTACTS': [contact_name(k_contacts)],
                  'K_INFECT': [k_infect]}).to_csv(out_dir/f'covasim_run_info_{tag}.csv', index=False)

    total_inf, total_rep = int(df['true_infections'].sum()), int(df['confirm'].sum())
    print(f"  total infections: {total_inf:,} ({total_inf/N:.0%} of {N:,} agents), reported cases: {total_rep:,} "
          f"(ratio {total_rep/total_inf:.2f})")
    print(f"  mean latent period {delays['delay_latent'].mean():.2f} d, incubation {delays['delay_incubation'].mean():.2f} d, "
          f"symptom-to-diagnosis {delays['delay_symptom_to_diagnosis'].mean():.2f} d; "
          f"friends mean {degree.mean():.2f}, std {degree.std():.1f}")

    return dict(scenario=scenario_tag(k_contacts, k_infect), k1=contact_name(k_contacts), k2=k_infect, seed=seed,
                seeds_died_out_before=tries, total_infections=total_inf, share_infected=round(total_inf/N, 3),
                reported_cases=total_rep, reported_per_infection=round(total_rep/total_inf, 3),
                peak_infections=int(df['true_infections'].max()), peak_day=int(df['true_infections'].values.argmax()),
                generation_time_mean=round(gt_mean, 2), generation_time_min=gt_min, friends_std=round(degree.std(), 2))


if __name__ == '__main__':
    OUT_ROOT.mkdir(exist_ok=True)
    rows = []
    for k_contacts, k_infect in SCENARIOS:
        tag = scenario_tag(k_contacts, k_infect)
        print(f'\n=== {tag} ===', flush=True)
        sim, degree, seed, tries = find_epidemic(k_contacts, k_infect)
        print(f'  using seed {seed} ({tries} earlier seeds died out)')
        rows.append(export_scenario(sim, degree, seed, tries, k_contacts, k_infect, OUT_ROOT/tag))
        pd.DataFrame(rows).to_csv(OUT_ROOT/'scenarios_summary.csv', index=False)   # rewritten after each scenario, so a crash keeps the earlier ones
    print(f'\nSaved the csv files in {OUT_ROOT}')
    print(pd.DataFrame(rows).to_string(index=False))