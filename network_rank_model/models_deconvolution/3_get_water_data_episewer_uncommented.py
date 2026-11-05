'''
Aim: run the rank-population Covasim ONCE (one seed) and export everything we may need later, so we can check
whether EpiNow2 (reported cases) and EpiSewer (wastewater concentrations), both in R, recover the true infections
from their observed stream alone.

Outputs (written next to this script):
  CLINICAL (for EpiNow2)
    covasim_cases.csv           -> one row per simulated day: date, confirm, true_infections, and other daily results
    covasim_delays.csv          -> one row per infected agent: the day each event happened, plus the delays between them
    covasim_transmission.csv    -> one row per infection event (who infected whom): used for the generation time
  WASTEWATER (for EpiSewer)
    covasim_ww_daily.csv        -> one row per site per day: true load, true concentration, noisy concentration, true infections
    covasim_ww_measurements.csv -> what EpiSewer is given: the sampled, noisy concentrations (site, date, concentration)
    covasim_ww_flows.csv        -> what EpiSewer is given: the daily flow at each site (site, date, flow)
    covasim_ww_shedding_dist.csv-> the TRUE shedding profile: share of an infection's shedding on day k after infection
  BOTH
    covasim_run_info.csv        -> the seed and settings the run used, plus the true load_per_case

Run from anywhere:  python run_covasim.py
'''
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

for f in (TIF, BOUNDARY, SEWERSHEDS):
    if not f.exists():
        raise FileNotFoundError(f'{f} not found. Run  git pull  on the network_rank_model branch to get docs/tutorials/san-diego-data.')

# ---------------------------------------------------------------------------
# Settings (same starting values as check_two_dispersions_outbreaks.py)
# ---------------------------------------------------------------------------
N          = 50000     # agents
N_DAYS     = 180       # length of the epidemic
SEED       = 1         # first seed to try (arbitrary). If that epidemic dies out, the next seeds are tried (see below)
MAX_TRIES  = 20        # the most seeds to try before giving up
TOOK_OFF   = 0.01      # an epidemic counts as taken off if at least this share of the agents were infected (as in check_two_dispersions_outbreaks.py)
BETA       = 0.012 #0.016     # transmission per contact (the Covasim default)
N_INITIAL  = 10        # agents infected on day 0
K_CONTACTS = None      # k1: friend-count dispersion. None = Poisson (the default)
K_INFECT   = 0.45      # k2: infectiousness dispersion. 0.45 is the Covasim default

# Wastewater settings. Covasim's viral shedding is in RELATIVE units (viral_load x rel_trans), not gene copies, so we
# multiply by a fixed, known constant to get gene copies. The constant is arbitrary but is saved, and the true
# load_per_case that follows from it is saved too (the wastewater analogue of the "true ascertainment" for EpiNow2).
WW_SCALE        = 1e9            # gene copies per unit of Covasim shedding
WW_FLOW         = 2e8            # litres of sewage per day, constant across days and sites
WW_NOISE_CV     = 0.2            # lab noise: multiplicative lognormal noise with mean 1 and this coefficient of variation
WW_SAMPLE_DOW   = None           # None = a sample every day (the best case). Covasim has no sampling schedule of its own, so this is our choice.
                                 # A tuple of weekdays (0 = Monday), e.g. (0, 3), gives sparse sampling as a stress test. covasim_ww_daily.csv
                                 # keeps the noisy concentration for EVERY day, so sparse sampling can also be applied later in R without re-running.
WW_SHED_DAYS    = 40             # length (days since infection) of the shedding profile that is exported
WW_MIN_INFECT   = 100            # warn if a sewershed had fewer infections than this (too few to benchmark an estimator on)
WW_MAX_REINFECT = 0.01           # warn if more than this share of infected agents were infected more than once


def run_one(k_contacts, k_infect, seed):
    ''' Build the rank population with the two dispersions, run the epidemic, and return the sim and each agent's friend count '''
    # Covasim only produces diagnoses if a testing intervention is added (the default sim has none, so
    # new_diagnoses would be all zeros). This is Covasim's built-in testing: each day, every symptomatic
    # agent has a 10% chance of being tested (and diagnosed), asymptomatic agents are never tested.
    testing = cv.test_prob(symp_prob=0.1, asymp_prob=0.0, start_day=0)

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


if __name__ == '__main__':
    # With few initial infections and overdispersed transmission, many epidemics die out in the first days (the 5 seeds
    # infect nobody). EpiNow2 can't be benchmarked on 15 infections, so we try SEED, SEED+1, ... and keep the FIRST run
    # that took off. The rule is fixed in advance (not "the nicest-looking curve"), and the seed used is printed and
    # saved so the run can be reproduced. Report the share of seeds that died out if you use this in a write-up.
    for tries in range(MAX_TRIES):
        seed = SEED + tries
        sim, degree = run_one(K_CONTACTS, K_INFECT, seed)
        ever = int(np.sum(~np.isnan(np.array(sim.people.date_exposed))))      # agents infected at least once
        print(f'seed {seed}: {ever:,} agents infected ({ever/N:.1%})', flush=True)
        if ever >= TOOK_OFF*N:
            break
    else:
        raise RuntimeError(f'No epidemic took off in {MAX_TRIES} seeds. Raise N_INITIAL (e.g. 20-50) or BETA, or lower K_INFECT overdispersion.')
    print(f'Using seed {seed} ({tries} earlier seeds died out)')
    ppl = sim.people

    # -----------------------------------------------------------------------
    # File 1: covasim_cases.csv, the daily series (one row per simulated day, for the whole population)
    #   date             : calendar date of that simulated day (Covasim starts on 2020-03-01, so day 0 = 2020-03-01)
    #   confirm          : agents DIAGNOSED on that date = the "reported cases". This is the ONLY column EpiNow2 is given.
    #                      It is the report date, NOT the infection date: it comes after infection by the incubation
    #                      period plus the testing delay.
    #   true_infections  : agents INFECTED on that date (date of infection). The ground truth; EpiNow2 never sees it.
    # The other columns are extra daily results kept for later (e.g. other observation streams like hospitalisations,
    # or comparing EpiNow2's Rt with the sim's). Any that this Covasim version does not have are skipped.
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
    df.to_csv(HERE/'covasim_cases.csv', index=False)

    # -----------------------------------------------------------------------
    # File 2: covasim_delays.csv, one row per agent who was ever infected
    # The date_ columns are SIMULATION DAYS (0 = the first day, i.e. 2020-03-01), not calendar dates.
    # NaN means the event never happened to that agent (e.g. no symptoms, never diagnosed).
    #   date_exposed     : day the agent was infected
    #   date_infectious  : day the agent became infectious (able to transmit)
    #   date_symptomatic : day symptoms started (NaN for asymptomatic agents)
    #   date_tested      : day the agent was tested (NaN if never)
    #   date_diagnosed   : day the agent was diagnosed (NaN if never tested positive)
    #   date_severe / date_critical / date_recovered / date_dead : later disease events (NaN if never)
    # The delay_ columns are the true delays in days, computed from the dates above (NaN if either date is missing):
    #   delay_latent               = date_infectious  - date_exposed     latent period: infected but not yet infectious
    #   delay_incubation           = date_symptomatic - date_exposed     incubation period: infection -> symptoms (EpiNow2 input)
    #   delay_symptom_to_diagnosis = date_diagnosed   - date_symptomatic reporting delay: symptoms -> report (EpiNow2 input)
    #   delay_infection_to_diagnosis = date_diagnosed - date_exposed     the full delay from infection to appearing in `confirm`
    # Agents infected on day 0 are the N_INITIAL seeds placed by pop_infected, infected by construction, not by transmission.
    # Caveat: date_exposed holds only an agent's most recent infection, so a reinfected agent's earlier infection is lost here
    # (reinfection is rare with default immunity; covasim_transmission.csv keeps every infection event).
    # The sewershed column is the integer ID in people.sewershed (0 = no sewershed file, -1 = outside every sewershed).
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
    delays['sewershed'] = np.asarray(ppl.sewershed)   # which catchment the agent lives in
    delays['delay_latent'] = delays['date_infectious'] - delays['date_exposed'] #Latent period: infected but not yet infectious
    delays['delay_incubation'] = delays['date_symptomatic'] - delays['date_exposed'] #Incubation period: infection -> symptoms
    delays['delay_symptom_to_diagnosis'] = delays['date_diagnosed'] - delays['date_symptomatic'] #Reporting delay: symptoms -> report
    delays['delay_infection_to_diagnosis'] = delays['date_diagnosed'] - delays['date_exposed'] 
    delays = delays.dropna(subset=['date_exposed'])
    delays.to_csv(HERE/'covasim_delays.csv', index=False)

    # -----------------------------------------------------------------------
    # File 3: covasim_transmission.csv, one row per infection event (who infected whom)
    # Covasim logs every transmission in sim.people.infection_log (source, target, date, layer). The generation time
    # needs this because it is a property of an infector-infectee PAIR, which the per-agent table above cannot give.
    #   source          : uid of the infector (NaN for the N_INITIAL seeds, who were not infected by anyone)
    #   target          : uid of the infectee
    #   date            : simulation day the infection happened
    #   layer           : which contact layer carried it (the rank network is layer 'a')
    #   generation_time = (infectee's infection day) - (infector's infection day), in days. EpiNow2 and EpiSewer need this
    #                     distribution for their renewal models. The infector's infection day is the latest one on or before this
    #                     event, taken from the log, so reinfection does not give negative values. NaN for seeds.
    #   source_infection_day, source_n_infections : the infector's infection day, and how many times they were infected
    #   serial_interval = (infectee's symptom onset) - (infector's symptom onset). NaN if either was asymptomatic.
    #   source_infectious_day = days after the INFECTOR became infectious that the transmission happened
    #                     (the infector's date_infectious is the earliest this can be 0).
    # -----------------------------------------------------------------------
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
        trans.to_csv(HERE/'covasim_transmission.csv', index=False)
        print(f"Generation time: mean {trans['generation_time'].mean():.2f} days, std {trans['generation_time'].std():.2f}, "
              f"from {trans['generation_time'].notna().sum():,} transmission pairs")
    else:
        print('WARNING: no infection_log found on sim.people in this Covasim/fork, so covasim_transmission.csv was NOT written. '
              'Try sim.make_transtree() instead, or check how the rank fork stores transmissions.')

    # -----------------------------------------------------------------------
    # WASTEWATER. Covasim gives every agent a viral shedding value every day: sim.results['viral_shedding_hist'] has one row per
    # agent and one column per day. A treatment plant sees the SUM over the agents that live in its catchment (people.sewershed).
    # We build that signal for the whole population ('ALL') and for each sewershed, then turn it into what a lab would report:
    #   load          = sum of shedding over the site's agents x WW_SCALE            gene copies per day arriving at the plant
    #   concentration = load / WW_FLOW                                               gene copies per litre (the noise-free truth)
    #   measurement   = concentration x lognormal noise, on the sampled days only    what EpiSewer is given
    # Real measurements are not available on every day, so only the days of the week in WW_SAMPLE_DOW are kept.
    # -----------------------------------------------------------------------
    shed = np.asarray(sim.results['viral_shedding_hist'], dtype=float)
    if shed.shape[0] != N and shed.shape[1] == N:
        shed = shed.T                                                      # make it (agents, days) whichever way this fork stores it
    dates = pd.to_datetime(sim.datevec)
    if shed.shape != (N, len(dates)):
        raise ValueError(f'viral_shedding_hist has shape {shed.shape}, expected ({N}, {len(dates)}) = (agents, days).')
    n_days = len(dates)
    sew = np.asarray(ppl.sewershed)
    labels = getattr(sim, 'sewershed_labels', None) or {}                  # {integer ID: plant name}

    # The sites: the whole population, plus one per sewershed (0 = no sewershed file, so nothing to add)
    sites = {'ALL': np.ones(N, dtype=bool)}
    for sid in np.unique(sew):
        if sid == 0:
            continue
        name = 'outside_sewersheds' if sid == -1 else str(labels.get(int(sid), f'sewershed_{int(sid)}'))
        sites[name] = (sew == sid)

    # Every infection event (agent, day), used for the TRUE infections per site. The log also holds the seeds.
    if log:
        ev_target = np.array([int(e['target']) for e in log])
        ev_day = np.array([int(e['date']) for e in log])
    else:
        ev_target = delays['uid'].to_numpy().astype(int)
        ev_day = delays['date_exposed'].to_numpy().astype(int)

    # Reinfection check. date_exposed (and so the shedding profile below) only holds an agent's LATEST infection, so agents who
    # appear as a target more than once in the log are the ones whose profile is ambiguous. Restarting with another seed would not
    # change this share (it depends on the settings, e.g. immunity and run length), so we measure it, save it and warn if it is high.
    infection_counts = np.bincount(ev_target, minlength=N)
    n_infected_ever = int((infection_counts > 0).sum())
    n_reinfected = int((infection_counts > 1).sum())
    reinfected = infection_counts > 1
    print(f'Reinfected agents: {n_reinfected} of {n_infected_ever} infected ({n_reinfected/max(n_infected_ever, 1):.2%})')
    if n_reinfected/max(n_infected_ever, 1) > WW_MAX_REINFECT:
        print(f'WARNING: more than {WW_MAX_REINFECT:.0%} of infected agents were reinfected. Reinfected agents are left out of the '
              f'shedding profile, but treat the true load_per_case as approximate, or shorten N_DAYS / change the immunity settings.')

    rng = np.random.default_rng(seed)
    sampled_day = np.ones(n_days, dtype=bool) if WW_SAMPLE_DOW is None else np.isin(dates.dayofweek, WW_SAMPLE_DOW)
    ww_rows = []
    for name, mask in sites.items():
        load = shed[mask].sum(axis=0)*WW_SCALE                                              # gene copies per day
        conc = load/WW_FLOW                                                                 # gene copies per litre, no noise
        sigma = np.sqrt(np.log(1 + WW_NOISE_CV**2))                                         # lognormal sigma that gives a CV of WW_NOISE_CV
        noise = rng.lognormal(mean=-sigma**2/2, sigma=sigma, size=n_days)                   # the -sigma^2/2 shift makes the mean exactly 1
        inf_here = np.bincount(ev_day[mask[ev_target]], minlength=n_days)[:n_days]          # true infections per day in this site
        ww_rows.append(pd.DataFrame({
            'site': name, 'date': dates, 'day': np.arange(n_days), 'n_people': int(mask.sum()),
            'true_infections': inf_here, 'load_gc_per_day': load, 'flow_L_per_day': WW_FLOW,
            'concentration_true': conc, 'concentration_noisy': conc*noise, 'sampled': sampled_day,
        }))
        if inf_here.sum() < WW_MIN_INFECT:
            print(f'WARNING: site {name!r} had only {inf_here.sum()} infections, too few to benchmark an estimator on. '
                  f'Use the ALL site, or raise N.')
    ww = pd.concat(ww_rows, ignore_index=True)
    ww.to_csv(HERE/'covasim_ww_daily.csv', index=False)

    # What EpiSewer is given. Days with zero concentration (no one shedding yet) are left out of the measurements, because EpiSewer
    # models log concentrations and cannot take a 0. Check where each site's first measurement is when you set the start date in R.
    meas = ww[ww['sampled']][['site', 'date', 'concentration_noisy']].rename(columns={'concentration_noisy': 'concentration'})
    n_zero = int((meas['concentration'] <= 0).sum())
    meas = meas[meas['concentration'] > 0]
    meas.to_csv(HERE/'covasim_ww_measurements.csv', index=False)
    ww[['site', 'date', 'flow_L_per_day']].rename(columns={'flow_L_per_day': 'flow'}).to_csv(HERE/'covasim_ww_flows.csv', index=False)

    # The TRUE shedding profile and load per case, taken from the simulation (the best case, like the true delays for EpiNow2).
    #   shedding_dist = share of an infection's total shedding that happens k days after the infection (k = 0 ... WW_SHED_DAYS-1)
    #   load_per_case = gene copies shed per infection, on average
    # Only agents infected early enough to have their whole shedding window inside the run are used, so the end of the run
    # does not cut the profile short. Reinfected agents are left out: their shedding row mixes two infections, and date_exposed
    # only marks the latest one. (Their infections still count in the true infections per site, which come from the log.)
    ok = delays[(delays['date_exposed'] <= n_days - WW_SHED_DAYS) & ~reinfected[delays['uid'].to_numpy().astype(int)]]
    uids = ok['uid'].to_numpy().astype(int)
    day0 = ok['date_exposed'].to_numpy().astype(int)
    profile = np.zeros(WW_SHED_DAYS)
    for u, d0 in zip(uids, day0):
        profile += shed[u, d0:d0 + WW_SHED_DAYS]
    coverage = profile.sum()/shed[uids].sum()                                  # share of total shedding inside the exported window
    shedding_dist = profile/profile.sum()
    load_per_case = profile.sum()/len(uids)*WW_SCALE
    pd.DataFrame({'day': np.arange(WW_SHED_DAYS), 'pmf': shedding_dist}).to_csv(HERE/'covasim_ww_shedding_dist.csv', index=False)
    if coverage < 0.99:
        print(f'WARNING: the {WW_SHED_DAYS}-day shedding window holds only {coverage:.1%} of total shedding. Raise WW_SHED_DAYS.')

    print(df.head(10))
    print(f"\nFirst date: {df['date'].iloc[0].date()}, last date: {df['date'].iloc[-1].date()}")
    print(f"Total infections: {df['true_infections'].sum():,} ({df['true_infections'].sum()/N:.0%} of the {N:,} agents)")
    print(f"Total reported cases: {df['confirm'].sum():,}")
    print(f"Mean latent period: {delays['delay_latent'].mean():.2f} days, incubation: {delays['delay_incubation'].mean():.2f} days, "
          f"symptom-to-diagnosis: {delays['delay_symptom_to_diagnosis'].mean():.2f} days")
    print(f"Mean friend count: {degree.mean():.2f}, std {degree.std():.1f}")
    print(f"\nWastewater sites: {', '.join(f'{k} ({v.sum():,} people)' for k, v in sites.items())}")
    print(f"Measurements kept: {len(meas):,} ({n_zero} zero-concentration sampled days dropped)")
    print(f"True load per case: {load_per_case:.3e} gene copies; mean shedding delay: {(np.arange(WW_SHED_DAYS)*shedding_dist).sum():.2f} days; "
          f"window holds {coverage:.1%} of shedding")
    pd.DataFrame({'seed': [seed], 'seeds_died_out_before': [tries], 'N': [N], 'N_INITIAL': [N_INITIAL], 'BETA': [BETA],
                  'K_CONTACTS': [K_CONTACTS], 'K_INFECT': [K_INFECT],
                  'WW_SCALE': [WW_SCALE], 'WW_FLOW': [WW_FLOW], 'WW_NOISE_CV': [WW_NOISE_CV], 'WW_SAMPLE_DOW': [str(WW_SAMPLE_DOW)],
                  'WW_SHED_DAYS': [WW_SHED_DAYS], 'LOAD_PER_CASE': [load_per_case],
                  'N_INFECTED': [n_infected_ever], 'N_REINFECTED': [n_reinfected]}).to_csv(HERE/'covasim_run_info.csv', index=False)   # settings used, for reproducing the run
    print(f"Saved the CSVs in {HERE}")