'''
WHAT THIS SCRIPT DOES
---------------------
Runs the rank-population Covasim model ONCE (one random seed) and saves everything we need to test two estimation tools
in R against the TRUE answer:

    EpiNow2   is given only the reported (diagnosed) cases      -> does it recover the true daily infections?
    EpiSewer  is given only the wastewater concentrations       -> does it recover the true daily infections?

The simulation knows exactly who was infected and when, so it can judge both tools.

FILES WRITTEN (next to this script)
-----------------------------------
  Clinical stream (for EpiNow2)
    covasim_cases.csv             one row per day: reported cases (`confirm`), true infections, and other daily results
    covasim_delays.csv            one row per ever-infected agent: the day of each event and the delays between events
    covasim_transmission.csv      one row per infection event (who infected whom): used for the generation time

  Wastewater stream (for EpiSewer)
    covasim_ww_daily.csv          one row per site per day: true load, true and noisy concentration, true infections
    covasim_ww_measurements.csv   what EpiSewer is given: sampled noisy concentrations   (site, date, concentration)
    covasim_ww_flows.csv          what EpiSewer is given: daily sewage flow               (site, date, flow)
    covasim_ww_shedding_dist.csv  the TRUE shedding profile (share of an infection's shedding k days after infection)

  Both
    covasim_run_info.csv          the seed and settings used, plus the TRUE load per case

HOW TO RUN
----------
    python run_covasim.py

A NOTE ON DAYS AND DATES
------------------------
Covasim counts days from 0. Day 0 is 2020-03-01 by default. Columns called `date_...` in covasim_delays.csv are these
simulation day numbers. Columns called `date` in the other files are real calendar dates.
'''
from pathlib import Path
import numpy as np
import pandas as pd
import covasim as cv


# =====================================================================================================================
# 1. FILE LOCATIONS
# =====================================================================================================================
OUTPUT_FOLDER = Path(__file__).resolve().parent         # the folder this script is in: all the CSVs are written here
REPO_ROOT = OUTPUT_FOLDER.parents[1]                    # the root of the repository
SAN_DIEGO_DATA_FOLDER = REPO_ROOT/'docs'/'tutorials'/'san-diego-data'

POPULATION_DENSITY_RASTER = SAN_DIEGO_DATA_FOLDER/'san_diego_county_ppp_2020.tif'    # people per ~100 m cell: decides where agents live
ZIP_CODE_BOUNDARIES       = SAN_DIEGO_DATA_FOLDER/'sandiego_zip_boundaries.geojson'  # one region per ZIP code
SEWERSHED_BOUNDARIES      = SAN_DIEGO_DATA_FOLDER/'sandiego_sewersheds.geojson'      # one catchment per treatment plant


# =====================================================================================================================
# 2. SETTINGS (everything you might want to change is here)
# =====================================================================================================================

# ---- The epidemic -------------------------------------------------------------------------------------------------
N_AGENTS                  = 50000      # number of simulated people
N_DAYS                    = 180        # length of the simulated epidemic, in days
TRANSMISSION_PROB_PER_CONTACT = 0.012  # Covasim's `beta`: chance of transmission per infectious contact (Covasim default is 0.016)
N_INITIAL_INFECTIONS      = 10         # agents infected on day 0 (the "seeds": infected by construction, not by anyone)
FRIEND_COUNT_DISPERSION   = None       # "k1": how uneven the number of friends is. None = Poisson, i.e. no extra unevenness (default)
INFECTIOUSNESS_DISPERSION = 0.45       # "k2": how uneven infectiousness is between people (superspreading). 0.45 is Covasim's default
DAILY_TEST_PROB_IF_SYMPTOMATIC = 0.1   # each day, every symptomatic agent has this chance of being tested (and so diagnosed)

# ---- Finding an epidemic that takes off ---------------------------------------------------------------------------
# With few initial infections and uneven infectiousness, many runs die out within days, and a tool cannot be tested on
# 15 infections. So we try FIRST_SEED_TO_TRY, then +1, then +2 ... and keep the FIRST run that takes off. The rule is fixed in
# advance (not "the nicest-looking curve"), and the seed that was used is saved so the run can be reproduced exactly.
FIRST_SEED_TO_TRY               = 1      # arbitrary starting seed
MAX_SEEDS_TO_TRY                = 20     # give up after this many seeds
MIN_SHARE_INFECTED_FOR_TAKEOFF  = 0.01   # a run "took off" if at least this share of all agents were infected at some point

# ---- Wastewater ---------------------------------------------------------------------------------------------------
# WHERE COVASIM'S SHEDDING VALUES COME FROM. Covasim draws each agent's daily shedding from the `shedding_hub` Python package
# (Woelfel et al. 2020, stool, gamma fit; see Sim.compute_viral_shedding_lookup). We checked the catalog entry: its unit is
# gc/mL, i.e. gene copies per mL of STOOL, so `viral_shedding_hist` is a stool concentration, not an arbitrary relative number.
# (The curve is defined from symptom onset, but Covasim applies it from each agent's date_infectious. Our shedding profile is
# measured from the matrix itself, in days since infection, so this does not affect the export.)
#
# To get from a stool concentration to what a person puts into the sewer, multiply by how much stool they produce per day.
# Median fecal wet mass is about 128 g per person per day (as used in wastewater models; 1 g is taken as 1 mL).
STOOL_ML_PER_PERSON_PER_DAY = 128      # mL of stool per person per day. load (gc/day) = stool concentration (gc/mL) x this

# Sewage flow is set PER PERSON, because concentration depends on how much water each person's waste is diluted into, not on
# the plant's total size. San Diego's Point Loma plant treats about 136-175 million gallons per day for ~2.2 million residents
# (City of San Diego fact sheet and annual reports), i.e. roughly 235-300 litres per person per day. The flow of a site is
# (number of agents in the site) x this number, so a bigger sewershed gets proportionally more flow. Constant over days.
SEWAGE_FLOW_LITRES_PER_PERSON_PER_DAY = 270

# Lab measurement noise has no Covasim equivalent: it is our own placeholder choice.
LAB_NOISE_CV                  = 0.2    # lab measurement noise: multiplicative, mean 1, with this coefficient of variation (sd/mean)

# Which days a sample is taken. Covasim has no sampling schedule of its own, so this is also our choice.
#   None   -> a sample every day (the best case)
#   (0, 3) -> only Mondays and Thursdays (0 = Monday ... 6 = Sunday): a sparse-sampling stress test
# covasim_ww_daily.csv keeps the noisy concentration for EVERY day, so a sparse schedule can also be applied later in R
# without re-running Covasim.
SAMPLED_WEEKDAYS = None

SHEDDING_PROFILE_DAYS = 40             # how many days after infection are kept in the exported shedding profile
WARN_IF_SITE_HAS_FEWER_INFECTIONS_THAN = 100   # a site with fewer infections than this is too small to test an estimator on
WARN_IF_REINFECTED_SHARE_ABOVE = 0.01          # warn if more than this share of infected agents were infected more than once


# =====================================================================================================================
# 3. RUNNING THE SIMULATION
# =====================================================================================================================
def check_input_files_exist():
    ''' Stop early, with a clear message, if the San Diego map files are missing '''
    for required_file in (POPULATION_DENSITY_RASTER, ZIP_CODE_BOUNDARIES, SEWERSHED_BOUNDARIES):
        if not required_file.exists():
            raise FileNotFoundError(f'{required_file} not found. Run  git pull  on the network_rank_model branch '
                                    f'to get docs/tutorials/san-diego-data.')


def run_simulation(seed):
    '''
    Build the rank population (people mostly contact people who live near them), run the epidemic, and return
    the finished simulation together with the number of friends each agent has.
    '''
    # Covasim only produces diagnoses if a testing intervention is added. Without one, `new_diagnoses` would be all zeros and
    # there would be no reported cases. Here: every symptomatic agent has a 10% chance per day of being tested and diagnosed.
    # Asymptomatic agents are never tested.
    testing = cv.test_prob(symp_prob=DAILY_TEST_PROB_IF_SYMPTOMATIC, asymp_prob=0.0, start_day=0)

    simulation = cv.Sim(
        pop_type='rank',                                   # contacts depend on how close people live
        pop_size=N_AGENTS,
        pop_infected=N_INITIAL_INFECTIONS,
        n_imports=0,                                       # no infections arrive from outside after day 0
        beta=TRANSMISSION_PROB_PER_CONTACT,
        n_days=N_DAYS,
        rand_seed=seed,
        verbose=0,
        # Uneven infectiousness (superspreading). All four entries are needed.
        beta_dist=dict(dist='neg_binomial', par1=1.0, par2=INFECTIOUSNESS_DISPERSION, step=0.01),
        interventions=testing,
        density_raster=str(POPULATION_DENSITY_RASTER),
        boundary_shapefile=str(ZIP_CODE_BOUNDARIES),
        sewershed_file=str(SEWERSHED_BOUNDARIES),
    )

    # `use_workers` only changes how fast the contact network is built, not the network itself.
    extra_initialize_arguments = dict(use_workers=True)
    if FRIEND_COUNT_DISPERSION is not None:
        # passed through sim.initialize to make_rank_based_contacts
        extra_initialize_arguments['dispersion'] = FRIEND_COUNT_DISPERSION
    simulation.initialize(**extra_initialize_arguments)

    # Count each agent's friends: how many times their id appears in either end of a friendship.
    friendships = simulation.people.contacts['a']          # the rank network is contact layer 'a'
    friend_count_per_agent = np.bincount(np.concatenate([friendships['p1'], friendships['p2']]), minlength=N_AGENTS)

    simulation.run()
    return simulation, friend_count_per_agent


def run_until_epidemic_takes_off():
    ''' Try seeds one after another and return the first run that took off (see the note in the settings) '''
    for n_seeds_died_out in range(MAX_SEEDS_TO_TRY):
        seed = FIRST_SEED_TO_TRY + n_seeds_died_out
        simulation, friend_count_per_agent = run_simulation(seed)

        n_agents_ever_infected = int(np.sum(~np.isnan(np.array(simulation.people.date_exposed))))
        print(f'seed {seed}: {n_agents_ever_infected:,} agents infected ({n_agents_ever_infected/N_AGENTS:.1%})', flush=True)
        if n_agents_ever_infected >= MIN_SHARE_INFECTED_FOR_TAKEOFF*N_AGENTS:
            print(f'Using seed {seed} ({n_seeds_died_out} earlier seeds died out)')
            return simulation, friend_count_per_agent, seed, n_seeds_died_out

    raise RuntimeError(f'No epidemic took off in {MAX_SEEDS_TO_TRY} seeds. Raise N_INITIAL_INFECTIONS (e.g. 20-50) or '
                       f'TRANSMISSION_PROB_PER_CONTACT, or lower the INFECTIOUSNESS_DISPERSION unevenness.')


# =====================================================================================================================
# 4. CLINICAL STREAM (for EpiNow2)
# =====================================================================================================================
def build_daily_cases_table(simulation):
    '''
    covasim_cases.csv: one row per simulated day, for the whole population.

      date             calendar date of that day (day 0 = 2020-03-01)
      day              simulation day number, starting at 0
      confirm          agents DIAGNOSED on that date = the "reported cases". This is the ONLY column EpiNow2 is given.
                       It is the REPORT date, not the infection date: a case appears after the incubation period plus the
                       testing delay, so reported cases are lower than, and later than, true infections.
      true_infections  agents INFECTED on that date. The ground truth. EpiNow2 never sees it.

    The other columns are extra daily results kept for later (other data streams, or comparing EpiNow2's Rt with
    Covasim's own `r_eff`). Any column this Covasim version does not have is skipped.
    '''
    daily_cases = pd.DataFrame({
        'date': pd.to_datetime(simulation.datevec),
        'day': np.arange(len(simulation.datevec)),
        'confirm': simulation.results['new_diagnoses'].values.astype(int),
        'true_infections': simulation.results['new_infections'].values.astype(int),
    })
    extra_daily_results = ['new_symptomatic', 'new_infectious', 'new_tests', 'new_severe', 'new_critical', 'new_recoveries',
                           'new_deaths', 'n_infectious', 'n_exposed', 'n_susceptible', 'cum_infections', 'cum_diagnoses', 'r_eff']
    for result_name in extra_daily_results:
        if result_name in simulation.results:
            daily_cases[result_name] = np.array(simulation.results[result_name].values)
    return daily_cases


def build_per_agent_delays_table(simulation, friend_count_per_agent):
    '''
    covasim_delays.csv: one row per agent who was ever infected.

    The `date_...` columns are SIMULATION DAYS (0 = the first day), not calendar dates. NaN means that event never
    happened to that agent (for example no symptoms, or never diagnosed).
      date_exposed      day the agent was infected
      date_infectious   day the agent became able to infect others
      date_symptomatic  day symptoms started (NaN for asymptomatic agents)
      date_tested       day the agent was tested (NaN if never)
      date_diagnosed    day the agent was diagnosed (NaN if never tested positive)
      date_severe, date_critical, date_recovered, date_dead   later disease events (NaN if they never happened)

    The `delay_...` columns are the TRUE delays in days, worked out from the dates (NaN if either date is missing):
      delay_latent                  infectious  - exposed      latent period: infected but not yet infectious
      delay_incubation              symptomatic - exposed      incubation period (an EpiNow2 input)
      delay_symptom_to_diagnosis    diagnosed   - symptomatic  reporting delay (an EpiNow2 input)
      delay_infection_to_diagnosis  diagnosed   - exposed      the whole delay from infection to appearing in `confirm`

    Other columns: age, sex, n_friends (friends in the rank network) and sewershed (integer id of the catchment the agent
    lives in: 0 = no sewershed file, -1 = outside every sewershed).

    Two things to know:
      * Agents infected on day 0 are the initial seeds, infected by construction and not by transmission.
      * `date_exposed` only holds an agent's MOST RECENT infection. A reinfected agent's earlier infection is lost here.
        This is NOT negligible: in the first run (seed 1) about 7.6% of infected agents were infected more than once,
        mostly 3 to 8 weeks after the previous infection (find_reinfected_agents() prints the share for each run).
        covasim_transmission.csv keeps every infection event.
    '''
    people = simulation.people
    date_columns = ['date_exposed', 'date_infectious', 'date_symptomatic', 'date_tested', 'date_diagnosed',
                    'date_severe', 'date_critical', 'date_recovered', 'date_dead']

    per_agent = pd.DataFrame({'uid': np.arange(N_AGENTS)})      # uid = the agent's id (its position in the population)
    for column in date_columns:
        if column in people.keys():
            per_agent[column] = np.array(people[column], dtype=float)
    per_agent['age'] = np.array(people.age)
    per_agent['sex'] = np.array(people.sex)
    per_agent['n_friends'] = friend_count_per_agent
    per_agent['sewershed'] = np.asarray(people.sewershed)

    per_agent['delay_latent']                  = per_agent['date_infectious']  - per_agent['date_exposed']
    per_agent['delay_incubation']              = per_agent['date_symptomatic'] - per_agent['date_exposed']
    per_agent['delay_symptom_to_diagnosis']    = per_agent['date_diagnosed']   - per_agent['date_symptomatic']
    per_agent['delay_infection_to_diagnosis']  = per_agent['date_diagnosed']   - per_agent['date_exposed']

    return per_agent.dropna(subset=['date_exposed'])            # keep only agents who were ever infected


def build_transmission_table(simulation):
    '''
    covasim_transmission.csv: one row per infection event (who infected whom). Returns None if Covasim kept no log.

    The generation time is a property of an infector-infectee PAIR, so the per-agent table cannot give it. It needs the
    transmission log, `people.infection_log`, where each entry has a source, target, date and layer.

      source           uid of the infector (NaN for the initial seeds, who were not infected by anyone)
      target           uid of the infectee
      date             simulation day the infection happened
      layer            contact layer that carried it (the rank network is layer 'a')
      generation_time  (infectee's infection day) - (infector's infection day), in days. NaN for seeds. EpiNow2 and
                       EpiSewer both need this distribution for their renewal models.
      source_infection_day, source_n_infections
                       the infector's infection day, and how many times they were infected (more than 1 = reinfected infector)
      serial_interval  (infectee's symptom onset) - (infector's symptom onset). NaN if either was asymptomatic.
                       Distorted by reinfection, so use it only where source_n_infections == 1.
      source_infectious_day
                       days after the INFECTOR became infectious that the transmission happened. The earliest this can be is 0.
    '''
    people = simulation.people
    infection_log = getattr(people, 'infection_log', None)
    if not infection_log:
        print('WARNING: no infection_log found on sim.people in this Covasim/fork, so covasim_transmission.csv was NOT written. '
              'Try sim.make_transtree() instead, or check how the rank fork stores transmissions.')
        return None

    transmissions = pd.DataFrame(infection_log)
    transmissions['source'] = pd.to_numeric(transmissions['source'], errors='coerce')    # None for the seeds becomes NaN
    infector_uid_with_nan = transmissions['source'].to_numpy()
    has_infector = ~np.isnan(infector_uid_with_nan)
    infector_uid = np.where(has_infector, infector_uid_with_nan, 0).astype(int)          # placeholder 0 for seeds, masked out below
    infectee_uid = transmissions['target'].to_numpy().astype(int)
    event_day = transmissions['date'].to_numpy(dtype=float)

    # Every infection day of every agent, taken from the log itself. We cannot use people.date_exposed for this: it only
    # keeps an agent's LATEST infection, so for a reinfected infector it could be AFTER the infection they caused, which
    # would give a negative generation time.
    infection_days_by_agent = {}
    for uid, day in zip(infectee_uid, event_day):
        infection_days_by_agent.setdefault(int(uid), []).append(day)

    def infectors_infection_day(uid, day_of_transmission):
        ''' The infector's latest infection on or before the day they passed it on '''
        earlier_infections = [d for d in infection_days_by_agent.get(int(uid), []) if d <= day_of_transmission]
        return max(earlier_infections) if earlier_infections else np.nan

    source_infection_day = np.array([
        infectors_infection_day(uid, day) if has_one else np.nan
        for uid, day, has_one in zip(infector_uid, event_day, has_infector)
    ])
    transmissions['source_infection_day'] = source_infection_day
    transmissions['source_n_infections'] = [
        len(infection_days_by_agent.get(int(uid), [])) if has_one else np.nan
        for uid, has_one in zip(infector_uid, has_infector)
    ]
    transmissions['generation_time'] = event_day - source_infection_day

    if 'date_symptomatic' in people.keys():
        symptom_onset_day = np.array(people.date_symptomatic, dtype=float)
        transmissions['serial_interval'] = np.where(has_infector, symptom_onset_day[infectee_uid] - symptom_onset_day[infector_uid], np.nan)
    if 'date_infectious' in people.keys():
        became_infectious_day = np.array(people.date_infectious, dtype=float)
        transmissions['source_infectious_day'] = np.where(has_infector, event_day - became_infectious_day[infector_uid], np.nan)

    return transmissions


# =====================================================================================================================
# 5. WASTEWATER STREAM (for EpiSewer)
#
# Covasim gives every agent a viral shedding value on every day: simulation.results['viral_shedding_hist'] has one row per
# agent and one column per day. A treatment plant sees the SUM of the shedding of the agents who live in its catchment.
# We build that signal for each "site" and turn it into what a lab would report:
#
#   load           = sum of shedding (gc/mL stool) over the site's agents x STOOL_ML_PER_PERSON_PER_DAY     gene copies (gc) per day
#   concentration  = load / (agents in site x SEWAGE_FLOW_LITRES_PER_PERSON_PER_DAY)                        gc per litre (noise-free truth)
#   measurement    = concentration x lab noise, on the sampled days only                                    what EpiSewer is given
# =====================================================================================================================
def get_shedding_by_agent_and_day(simulation):
    ''' Each agent's stool concentration (gc/mL) on each day, as a table with one row per agent and one column per day '''
    shedding = np.asarray(simulation.results['viral_shedding_hist'], dtype=float)
    if shedding.shape[0] != N_AGENTS and shedding.shape[1] == N_AGENTS:
        shedding = shedding.T                                    # this fork stored it as days x agents: flip it
    expected_shape = (N_AGENTS, len(simulation.datevec))
    if shedding.shape != expected_shape:
        raise ValueError(f'viral_shedding_hist has shape {shedding.shape}, expected {expected_shape} = (agents, days).')
    return shedding


def get_wastewater_sites(simulation):
    '''
    The places a wastewater sample could be taken: the whole population ('ALL'), plus one per sewershed.
    Returns {site name: True/False for each agent saying whether they live in that site}.
    '''
    sewershed_id_per_agent = np.asarray(simulation.people.sewershed)
    sewershed_names = getattr(simulation, 'sewershed_labels', None) or {}      # {integer id: plant name}

    sites = {'ALL': np.ones(N_AGENTS, dtype=bool)}
    for sewershed_id in np.unique(sewershed_id_per_agent):
        if sewershed_id == 0:
            continue                                             # 0 = no sewershed file, so there is nothing to add
        if sewershed_id == -1:
            site_name = 'outside_sewersheds'
        else:
            site_name = str(sewershed_names.get(int(sewershed_id), f'sewershed_{int(sewershed_id)}'))
        sites[site_name] = (sewershed_id_per_agent == sewershed_id)
    return sites


def get_infection_events(simulation, per_agent_delays):
    '''
    Every infection event as two arrays: which agent was infected, and on which day. Used for the TRUE number of
    infections per site per day. The log is preferred because it keeps reinfections and also lists the initial seeds.
    '''
    infection_log = getattr(simulation.people, 'infection_log', None)
    if infection_log:
        infected_agent = np.array([int(event['target']) for event in infection_log])
        infection_day = np.array([int(event['date']) for event in infection_log])
    else:
        infected_agent = per_agent_delays['uid'].to_numpy().astype(int)
        infection_day = per_agent_delays['date_exposed'].to_numpy().astype(int)
    return infected_agent, infection_day


def find_reinfected_agents(infected_agent):
    '''
    Agents who appear as infected more than once. Their `date_exposed` only marks the latest infection, so their
    shedding is ambiguous. Restarting with another seed would not change the share: it depends on the settings (immunity,
    run length), so we measure it, save it, and warn if it is high.
    '''
    n_infections_per_agent = np.bincount(infected_agent, minlength=N_AGENTS)
    is_reinfected = n_infections_per_agent > 1
    n_ever_infected = int((n_infections_per_agent > 0).sum())
    n_reinfected = int(is_reinfected.sum())

    reinfected_share = n_reinfected/max(n_ever_infected, 1)
    print(f'Reinfected agents: {n_reinfected} of {n_ever_infected} infected ({reinfected_share:.2%})')
    if reinfected_share > WARN_IF_REINFECTED_SHARE_ABOVE:
        print(f'WARNING: more than {WARN_IF_REINFECTED_SHARE_ABOVE:.0%} of infected agents were reinfected. They are left out of '
              f'the shedding profile, but treat the true load per case as approximate, or shorten N_DAYS / change the immunity settings.')
    return is_reinfected, n_ever_infected, n_reinfected


def build_wastewater_daily_table(shedding_by_agent_and_day, sites, infected_agent, infection_day, dates, seed):
    '''
    covasim_ww_daily.csv: one row per site per day, everything about the wastewater signal in one table.

      site                 'ALL' or a sewershed name
      date, day            calendar date and simulation day
      n_people             how many agents live in the site
      true_infections      TRUE new infections that day among the site's agents (from the infection log)
      load_gc_per_day      TRUE gene copies per day arriving at the plant
      flow_L_per_day       the site's sewage flow, litres per day (= n_people x SEWAGE_FLOW_LITRES_PER_PERSON_PER_DAY)
      concentration_true   load / flow, gc per litre, no lab noise
      concentration_noisy  the same with lab noise applied (what a lab would report)
      sampled              whether a sample is taken on that day (see SAMPLED_WEEKDAYS)
    '''
    n_days = len(dates)
    random_generator = np.random.default_rng(seed)
    is_sampling_day = np.ones(n_days, dtype=bool) if SAMPLED_WEEKDAYS is None else np.isin(dates.dayofweek, SAMPLED_WEEKDAYS)

    # THE TABLES USED BELOW, AND THEIR SHAPES (rows x columns)
    #   shedding_by_agent_and_day    (N_AGENTS, n_days), e.g. (50000, 181)   [agent, day] = that agent's stool concentration that day (gc/mL)
    #   agents_in_site               (N_AGENTS,), e.g. (50000,)              True or False per agent: do they live in this site?
    #   shedding_of_agents_in_site   (n_agents_in_site, n_days), e.g. (3200, 181)   the same table, keeping only the rows of agents in the site
    #   total_shedding_per_day       (n_days,), e.g. (181,)                  one total per day, after adding up the agents
    #   load_gc_per_day              (n_days,)                               the same, converted to gene copies per day
    #   concentration_true           (n_days,)                               load divided by flow
    #   * The 181 is N_DAYS + 1, because Covasim also counts day 0

    # Calculate the log-scale standard deviation (sigma) needed to produce the specified
    # coefficient of variation (CV = 0.2, or 20%) on the original measurement scale.
    lognormal_sigma = np.sqrt(np.log(1 + LAB_NOISE_CV**2))
    tables_per_site = []
    for site_name, agents_in_site in sites.items():

        shedding_of_agents_in_site = shedding_by_agent_and_day[agents_in_site]                    # step 1: keep only the agents who live in this site
        total_shedding_per_day     = shedding_of_agents_in_site.sum(axis=0)                       # step 2: add the agents up, leaving one total per day
        load_gc_per_day            = total_shedding_per_day * STOOL_ML_PER_PERSON_PER_DAY         # step 3: gc/mL of stool x mL of stool per day = gc per day

        # This site's sewage volume per day: each resident's waste is diluted into this much water, so a bigger sewershed gets more flow.
        flow_litres_per_day = agents_in_site.sum() * SEWAGE_FLOW_LITRES_PER_PERSON_PER_DAY
        concentration_true  = load_gc_per_day / flow_litres_per_day                               # gc per litre, no lab noise

        # Generate daily multiplicative measurement-noise factors from a lognormal distribution.
        # The -sigma^2/2 adjustment to the log-scale mean ensures the multipliers have an expected
        # value of 1, so noise adds 20% relative variability without systematically biasing
        # concentrations upwards or downwards.
        lab_noise_multiplier = random_generator.lognormal(mean=-lognormal_sigma**2/2, sigma=lognormal_sigma, size=n_days)

        # True infections per day in this site: count the infection events whose infected agent lives in the site.
        infected_here = agents_in_site[infected_agent]
        true_infections_per_day = np.bincount(infection_day[infected_here], minlength=n_days)[:n_days]

        tables_per_site.append(pd.DataFrame({
            'site': site_name,
            'date': dates,
            'day': np.arange(n_days),
            'n_people': int(agents_in_site.sum()),
            'true_infections': true_infections_per_day,
            'load_gc_per_day': load_gc_per_day,
            'flow_L_per_day': flow_litres_per_day,
            'concentration_true': concentration_true,
            'concentration_noisy': concentration_true * lab_noise_multiplier,
            'sampled': is_sampling_day,
        }))
        if true_infections_per_day.sum() < WARN_IF_SITE_HAS_FEWER_INFECTIONS_THAN:
            print(f'WARNING: site {site_name!r} had only {true_infections_per_day.sum()} infections, too few to test an '
                  f'estimator on. Use the ALL site, or raise N_AGENTS.')
    return pd.concat(tables_per_site, ignore_index=True)


def build_measurements_and_flows_tables(wastewater_daily):
    '''
    The two tables EpiSewer is given.
      covasim_ww_measurements.csv   site, date, concentration (noisy, sampled days only, gc per litre)
      covasim_ww_flows.csv          site, date, flow (litres per day, every day)

    Sampled days with a concentration of exactly 0 (nobody is shedding yet) are LEFT OUT of the measurements. EpiSewer works
    with log concentrations and cannot take a 0. Check where each site's first measurement is when setting the start date in R.
    '''
    sampled_days = wastewater_daily[wastewater_daily['sampled']]
    measurements = sampled_days[['site', 'date', 'concentration_noisy']].rename(columns={'concentration_noisy': 'concentration'})
    n_zero_concentration = int((measurements['concentration'] <= 0).sum())
    measurements = measurements[measurements['concentration'] > 0]

    flows = wastewater_daily[['site', 'date', 'flow_L_per_day']].rename(columns={'flow_L_per_day': 'flow'})
    return measurements, flows, n_zero_concentration


def estimate_true_shedding_profile(shedding_by_agent_and_day, per_agent_delays, is_reinfected, n_days):
    '''
    covasim_ww_shedding_dist.csv and the TRUE load per case, both taken from the simulation. This is the best case, like
    handing EpiNow2 the true delays.

      shedding profile  the share of one infection's total shedding that happens k days after the infection (k = 0, 1, ...).
                        It sums to 1.
      load per case     the average number of gene copies shed by one infection in total.

    Which agents are used: only agents infected early enough that their whole shedding window fits inside the run (so the
    end of the run does not cut the profile short), and not reinfected ones (their shedding row mixes two infections).
    Their infections still count in the true infections per site, because those come from the infection log.

    Returns: the profile, the load per case, and the share of total shedding that falls inside the exported window.
    '''
    infected_early_enough = per_agent_delays['date_exposed'] <= n_days - SHEDDING_PROFILE_DAYS
    infected_once = ~is_reinfected[per_agent_delays['uid'].to_numpy().astype(int)]
    used_agents = per_agent_delays[infected_early_enough & infected_once]

    used_agent_ids = used_agents['uid'].to_numpy().astype(int)
    used_infection_days = used_agents['date_exposed'].to_numpy().astype(int)

    # Add up each agent's shedding in the days right after THEIR infection, so all agents are lined up by "days since infection".
    shedding_by_days_since_infection = np.zeros(SHEDDING_PROFILE_DAYS)
    for agent_id, infection_day in zip(used_agent_ids, used_infection_days):
        shedding_by_days_since_infection += shedding_by_agent_and_day[agent_id, infection_day:infection_day + SHEDDING_PROFILE_DAYS]

    total_shedding_in_window = shedding_by_days_since_infection.sum()
    total_shedding_all_days = shedding_by_agent_and_day[used_agent_ids].sum()
    share_of_shedding_in_window = total_shedding_in_window / total_shedding_all_days

    shedding_profile = shedding_by_days_since_infection / total_shedding_in_window
    # Average total shedding of one infection (gc/mL summed over days) x stool per day = total gene copies shed per infection.
    load_per_case_gene_copies = total_shedding_in_window / len(used_agent_ids) * STOOL_ML_PER_PERSON_PER_DAY
    return shedding_profile, load_per_case_gene_copies, share_of_shedding_in_window


# =====================================================================================================================
# 6. MAIN
# =====================================================================================================================
def main():
    check_input_files_exist()
    simulation, friend_count_per_agent, seed, n_seeds_died_out = run_until_epidemic_takes_off()

    # ---- Clinical stream -----------------------------------------------------------------------------------------
    daily_cases = build_daily_cases_table(simulation)
    daily_cases.to_csv(OUTPUT_FOLDER/'covasim_cases.csv', index=False)

    per_agent_delays = build_per_agent_delays_table(simulation, friend_count_per_agent)
    per_agent_delays.to_csv(OUTPUT_FOLDER/'covasim_delays.csv', index=False)

    transmissions = build_transmission_table(simulation)
    if transmissions is not None:
        transmissions.to_csv(OUTPUT_FOLDER/'covasim_transmission.csv', index=False)
        print(f"Generation time: mean {transmissions['generation_time'].mean():.2f} days, "
              f"std {transmissions['generation_time'].std():.2f}, "
              f"from {transmissions['generation_time'].notna().sum():,} transmission pairs")

    # ---- Wastewater stream ---------------------------------------------------------------------------------------
    shedding_by_agent_and_day = get_shedding_by_agent_and_day(simulation)
    dates = pd.to_datetime(simulation.datevec)
    sites = get_wastewater_sites(simulation)
    infected_agent, infection_day = get_infection_events(simulation, per_agent_delays)
    is_reinfected, n_ever_infected, n_reinfected = find_reinfected_agents(infected_agent)

    wastewater_daily = build_wastewater_daily_table(shedding_by_agent_and_day, sites, infected_agent, infection_day, dates, seed)
    wastewater_daily.to_csv(OUTPUT_FOLDER/'covasim_ww_daily.csv', index=False)

    measurements, flows, n_zero_concentration = build_measurements_and_flows_tables(wastewater_daily)
    measurements.to_csv(OUTPUT_FOLDER/'covasim_ww_measurements.csv', index=False)
    flows.to_csv(OUTPUT_FOLDER/'covasim_ww_flows.csv', index=False)

    shedding_profile, load_per_case, share_of_shedding_in_window = estimate_true_shedding_profile(
        shedding_by_agent_and_day, per_agent_delays, is_reinfected, n_days=len(dates))
    pd.DataFrame({'day': np.arange(SHEDDING_PROFILE_DAYS), 'pmf': shedding_profile}).to_csv(
        OUTPUT_FOLDER/'covasim_ww_shedding_dist.csv', index=False)
    if share_of_shedding_in_window < 0.99:
        print(f'WARNING: the {SHEDDING_PROFILE_DAYS}-day shedding window holds only {share_of_shedding_in_window:.1%} of total '
              f'shedding. Raise SHEDDING_PROFILE_DAYS.')

    # ---- Run info: the settings used, so the run can be reproduced ----------------------------------------------
    # `LOAD_PER_CASE` is the column the R script reads. The two wastewater settings that changed are now called
    # WW_STOOL_ML_PER_DAY (was WW_SCALE) and WW_FLOW_PER_PERSON (was WW_FLOW, a constant per plant).
    pd.DataFrame({
        'seed': [seed], 'seeds_died_out_before': [n_seeds_died_out], 'N': [N_AGENTS], 'N_INITIAL': [N_INITIAL_INFECTIONS],
        'BETA': [TRANSMISSION_PROB_PER_CONTACT], 'K_CONTACTS': [FRIEND_COUNT_DISPERSION], 'K_INFECT': [INFECTIOUSNESS_DISPERSION],
        'WW_STOOL_ML_PER_DAY': [STOOL_ML_PER_PERSON_PER_DAY], 'WW_FLOW_PER_PERSON': [SEWAGE_FLOW_LITRES_PER_PERSON_PER_DAY],
        'WW_NOISE_CV': [LAB_NOISE_CV],
        'WW_SAMPLE_DOW': [str(SAMPLED_WEEKDAYS)], 'WW_SHED_DAYS': [SHEDDING_PROFILE_DAYS],
        'LOAD_PER_CASE': [load_per_case], 'N_INFECTED': [n_ever_infected], 'N_REINFECTED': [n_reinfected],
    }).to_csv(OUTPUT_FOLDER/'covasim_run_info.csv', index=False)

    # ---- Summary printed to the screen ---------------------------------------------------------------------------
    print(daily_cases.head(10))
    print(f"\nFirst date: {daily_cases['date'].iloc[0].date()}, last date: {daily_cases['date'].iloc[-1].date()}")
    total_infections = daily_cases['true_infections'].sum()
    print(f"Total infections: {total_infections:,} ({total_infections/N_AGENTS:.0%} of the {N_AGENTS:,} agents)")
    print(f"Total reported cases: {daily_cases['confirm'].sum():,}")
    print(f"Mean latent period: {per_agent_delays['delay_latent'].mean():.2f} days, "
          f"incubation: {per_agent_delays['delay_incubation'].mean():.2f} days, "
          f"symptom-to-diagnosis: {per_agent_delays['delay_symptom_to_diagnosis'].mean():.2f} days")
    print(f"Mean friend count: {friend_count_per_agent.mean():.2f}, std {friend_count_per_agent.std():.1f}")
    print(f"\nWastewater sites: {', '.join(f'{name} ({agents.sum():,} people)' for name, agents in sites.items())}")
    print(f"Measurements kept: {len(measurements):,} ({n_zero_concentration} zero-concentration sampled days left out)")
    mean_shedding_delay = (np.arange(SHEDDING_PROFILE_DAYS) * shedding_profile).sum()
    print(f"True load per case: {load_per_case:.3e} gene copies; mean shedding delay: {mean_shedding_delay:.2f} days; "
          f"the window holds {share_of_shedding_in_window:.1%} of shedding")
    print(f"Saved the CSVs in {OUTPUT_FOLDER}")


if __name__ == '__main__':
    main()