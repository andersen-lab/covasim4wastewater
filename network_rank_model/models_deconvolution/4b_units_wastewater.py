'''
WALK-THROUGH: where every wastewater number and unit comes from, step by step
-----------------------------------------------------------------------------
Run this and read the printed output from top to bottom. It does NOT use reported cases.
It follows one quantity, the viral shedding of each agent, from the shedding_hub package to the inputs EpiSewer needs.

Every printed line is tagged with how we know it:
    [printed here]   this script just computed or printed it, so you can see it yourself
    [sim.py]         read in Covasim's sim.py (Sim.compute_viral_shedding_lookup and Sim.step)
    [assumption]     OUR choice, not a Covasim output (these are the numbers to defend or replace)
    [EpiSewer docs]  taken from EpiSewer's documentation (check the help pages of your installed version)
    [NOT CHECKED]    we believe it but have not confirmed it

Needs: covasim (your fork) and shedding_hub.   Run:  python walk_through_units.py
'''
import numpy as np
import pandas as pd
import covasim as cv
import shedding_hub as sh

# ---- Our two assumptions (same names and values as run_covasim.py) -------------------------------------------------
STOOL_ML_PER_PERSON_PER_DAY = 128              # mL of stool per person per day (Rose et al. 2015; 1 g taken as 1 mL)
SEWAGE_FLOW_LITRES_PER_PERSON_PER_DAY = 270    # litres per person per day (Point Loma flow / residents, 235-300 L)

# ---- A small simulation, just to look at the numbers ---------------------------------------------------------------
N_AGENTS = 20000
N_DAYS = 90
N_INITIAL_INFECTIONS = 50
SEED = 1
SHEDDING_PROFILE_DAYS = 40


def step(number, title):
    print('\n' + '=' * 110)
    print(f'STEP {number}: {title}')
    print('=' * 110)


def show(label, value, unit='', source=''):
    ''' One printed line: what it is, its value, its unit, and how we know it '''
    print(f'  {label:<38} {value}   {unit}')
    if source:
        print(f'  {"":<38} -> {source}')


# =====================================================================================================================
step(1, 'What we ask shedding_hub for (this is where the unit is declared)')
# =====================================================================================================================
# Covasim does exactly this in Sim.compute_viral_shedding_lookup() in sim.py.
catalog = sh.load_shedding_catalog()
source = catalog.select(dataset_id='woelfel2020virological', analyte='stool', model='gamma')

show('dataset_id', 'woelfel2020virological', '', 'the Wölfel et al. 2020 study; the id is set in sim.py [sim.py]')
show('analyte', 'stool', '', 'so the value is a concentration IN STOOL, not in sewage [sim.py]')
show('model', 'gamma', '', 'a fitted gamma curve of shedding over time [sim.py]')
show('unit', getattr(source, 'unit', '?'), '', 'the unit is a LABEL stored in the catalog entry (source.unit) [printed here]')
show('reference event', getattr(source, 'reference_event', '?'), '',
     'day 0 of the curve = this event (Covasim applies it from date_infectious: checked in step 3) [printed here]')
print('\n  NOTE: Covasim does not define this unit. It copies the numbers and the label stays in shedding_hub.')
print('  [NOT CHECKED] the original paper may report gene copies per GRAM of stool; we treat 1 g as 1 mL.')

# =====================================================================================================================
step(2, 'What shedding_hub returns for a few individuals (the raw numbers)')
# =====================================================================================================================
raw = sh.simulate_shedding(source, n_individuals=5, times=np.arange(1, 31), dispersion=0.1)
show('columns returned', list(raw.columns), '', '[printed here]')
print('\n  First rows:')
print(raw.head(6).to_string(index=False))
biggest = float(raw['value'].max())
print()
show('largest value', f'{biggest:,.1f}', 'gc/mL (if the label is right)', '[printed here]')
if biggest > 100:
    print('  -> thousands or more: these look like LINEAR concentrations, not log10 values. [printed here]')
else:
    print('  -> single or double digits: these could be log10 values. CHECK before multiplying by 128 mL. [printed here]')

# =====================================================================================================================
step(3, 'What Covasim does with them: viral_shedding_hist (agents x days)')
# =====================================================================================================================
sim = cv.Sim(pop_type='random', pop_size=N_AGENTS, pop_infected=N_INITIAL_INFECTIONS, n_days=N_DAYS, rand_seed=SEED, verbose=0)
sim.initialize()
lookup = np.array(sim.shedding_lookup, dtype=float).copy()   # copy NOW: Covasim redraws sim.shedding_lookup after the run
sim.run()
hist = np.asarray(sim.results['viral_shedding_hist'], dtype=float)

show('shedding_lookup (from shedding_hub)', lookup.shape, '(agents, days)', 'one curve per agent [sim.py]')
show('viral_shedding_hist', hist.shape, '(agents, days)', 'Covasim also counts day 0, so one more column [printed here]')
show('largest value in viral_shedding_hist', f'{hist.max():,.1f}', 'gc/mL of stool', 'same unit as the shedding_hub label [printed here]')
print('\n  Covasim also has people.viral_load. That is a different, relative quantity (transmissibility) and is NOT used here.')

# Check 1: are the numbers copied unchanged from shedding_hub?
shedders = np.where((hist > 0).any(axis=1))[0]
agent = int(shedders[0])
values = hist[agent][hist[agent] > 0]
relative_gap = np.min(np.abs(lookup[agent][None, :] - values[:, None]) / values[:, None], axis=1)
copied_unchanged = bool((relative_gap < 1e-5).all())
show('values copied unchanged?', copied_unchanged, '',
     f'agent {agent}: every non-zero value of viral_shedding_hist is in its shedding_lookup row [printed here]')
if not copied_unchanged:
    print('  -> NOT unchanged: Covasim rescales somewhere (read compute_viral_shedding in utils.py) and the unit may differ.')

# Check 2: on which day does each agent's curve start?
date_exposed = np.asarray(sim.people.date_exposed, dtype=float)
date_infectious = np.asarray(sim.people.date_infectious, dtype=float)
print('\n  Where does the curve start? (day numbers, 0 = first day of the run)')
print(f'  {"agent":>6} {"date_exposed":>14} {"date_infectious":>16} {"first day with shedding":>26}')
for a in shedders[:6]:
    first_day = int(np.argmax(hist[a] > 0))
    print(f'  {a:>6} {date_exposed[a]:>14.0f} {date_infectious[a]:>16.0f} {first_day:>26}')
print('  -> if "first day with shedding" equals date_infectious (or +1), the curve starts at date_infectious. [printed here]')

# =====================================================================================================================
step(4, 'From shedding to what reaches the plant: load, flow, concentration (our arithmetic + 2 assumptions)')
# =====================================================================================================================
total_shedding_per_day = hist.sum(axis=0)                 # add up the agents of the site (here: everyone)
peak_day = int(np.argmax(total_shedding_per_day))
n_shedding_that_day = int((hist[:, peak_day] > 0).sum())

load_gc_per_day = total_shedding_per_day[peak_day] * STOOL_ML_PER_PERSON_PER_DAY
flow_L_per_day = N_AGENTS * SEWAGE_FLOW_LITRES_PER_PERSON_PER_DAY
concentration_gc_per_L = load_gc_per_day / flow_L_per_day

print(f'  Worked example for day {peak_day} (the day with the most shedding), site = all {N_AGENTS:,} agents:\n')
show('agents shedding that day', n_shedding_that_day, 'agents', '[printed here]')
show('sum of viral_shedding_hist', f'{total_shedding_per_day[peak_day]:,.1f}', 'gc/mL of stool (still a concentration)',
     'adding concentrations is only a total if every agent makes the same stool volume [assumption]')
show('x STOOL_ML_PER_PERSON_PER_DAY', STOOL_ML_PER_PERSON_PER_DAY, 'mL of stool per person per day',
     'median fecal wet mass 128 g (Rose et al. 2015), 1 g taken as 1 mL [assumption]')
show('= load_gc_per_day', f'{load_gc_per_day:,.3e}', 'gene copies per day', 'gc/mL x mL = gc per day [printed here]')
show('flow_L_per_day', f'{flow_L_per_day:,}', 'litres per day',
     f'agents x {SEWAGE_FLOW_LITRES_PER_PERSON_PER_DAY} L; Point Loma flow / residents; Covasim has no flow [assumption]')
show('concentration = load / flow', f'{concentration_gc_per_L:,.2f}', 'gene copies per litre', '[printed here]')
print('\n  Lab noise is applied on top of this: measurement = concentration x noise (mean 1, CV 0.2). [assumption]')

# =====================================================================================================================
step(5, 'Units EpiSewer wants: same volume unit, ideally gc/mL and mL/day')
# =====================================================================================================================
concentration_gc_per_mL = concentration_gc_per_L / 1000
flow_mL_per_day = flow_L_per_day * 1000
load_check_litres = concentration_gc_per_L * flow_L_per_day
load_check_mL = concentration_gc_per_mL * flow_mL_per_day

show('concentration (gc/L -> gc/mL)', f'{concentration_gc_per_mL:,.4f}', 'gc/mL', 'divide by 1000 [printed here]')
show('flow (L/day -> mL/day)', f'{flow_mL_per_day:,}', 'mL/day', 'multiply by 1000 [printed here]')
show('load = concentration x flow, both ways', f'{load_check_litres:,.3e} vs {load_check_mL:,.3e}', 'gc/day',
     'identical, so the conversion does not change the load [printed here]')
print('\n  EpiSewer wants concentration and flow in the same volume unit, ideally gc/mL and mL/day. [EpiSewer docs]')

# =====================================================================================================================
step(6, 'The two other EpiSewer inputs that come from the shedding table')
# =====================================================================================================================
# Line each infected agent's shedding up by days since THEIR infection (date_exposed), then average.
used = np.where(~np.isnan(date_exposed) & (date_exposed <= N_DAYS - SHEDDING_PROFILE_DAYS))[0]
profile_sum = np.zeros(SHEDDING_PROFILE_DAYS)
for a in used:
    start = int(date_exposed[a])
    profile_sum += hist[a, start:start + SHEDDING_PROFILE_DAYS]

if len(used) == 0 or profile_sum.sum() == 0:
    print('  No agents were infected early enough in this small run. Raise N_DAYS or N_INITIAL_INFECTIONS.')
else:
    shedding_distribution = profile_sum / profile_sum.sum()
    load_per_case = profile_sum.sum() / len(used) * STOOL_ML_PER_PERSON_PER_DAY
    mean_delay = float((np.arange(SHEDDING_PROFILE_DAYS) * shedding_distribution).sum())

    print(f'  Using {len(used):,} agents infected early enough that their whole {SHEDDING_PROFILE_DAYS}-day window fits in the run.\n')
    show('shedding distribution', 'share of one infection\'s shedding on each day after date_exposed', '(no unit, sums to 1)',
         'EpiSewer input: shedding_dist [printed here]')
    print('  first 12 days since infection:', np.round(shedding_distribution[:12], 3))
    show('  sums to', f'{shedding_distribution.sum():.6f}', '', '[printed here]')
    show('  mean day of shedding', f'{mean_delay:.2f}', 'days after infection', '[printed here]')
    print()
    show('load per case', f'{load_per_case:,.3e}', 'gene copies per infection',
         'mean total of viral_shedding_hist per infection x 128 mL; EpiSewer input: load_per_case [printed here] [assumption: 128 mL]')
    print('\n  EpiSewer takes load per case in the same unit as the numerator of the concentration (gc). [EpiSewer docs]')
    print('  It must be passed explicitly (load_per_case_assume), otherwise EpiSewer calibrates it itself. [EpiSewer docs]')
    print('  [NOT CHECKED] this simple version does not remove reinfected agents; run_covasim.py does.')

# =====================================================================================================================
step(7, 'Summary: every number, its unit, and how we know it')
# =====================================================================================================================
summary = pd.DataFrame([
    ['viral_shedding_hist',          'gc per mL of stool',   'shedding_hub catalog label, copied by Covasim',         'Covasim output'],
    ['STOOL_ML_PER_PERSON_PER_DAY',  'mL stool / person / day', 'Rose et al. 2015 (128 g, 1 g = 1 mL)',               'OUR ASSUMPTION'],
    ['load_gc_per_day',              'gc per day',           'sum of viral_shedding_hist x 128 mL',                  'our arithmetic'],
    ['SEWAGE_FLOW_L_PER_PERSON',     'L / person / day',     'Point Loma flow / residents (235-300 L)',               'OUR ASSUMPTION'],
    ['flow_L_per_day',               'L per day',            'agents in site x 270 L',                                'our arithmetic'],
    ['concentration',                'gc per L (gc/mL in R)', 'load / flow',                                          'our arithmetic'],
    ['lab noise',                    'none (multiplier)',    'mean 1, CV 0.2, our placeholder',                       'OUR ASSUMPTION'],
    ['shedding distribution',        'none (sums to 1)',     'viral_shedding_hist lined up by days since date_exposed', 'from the Covasim table'],
    ['load per case',                'gc per infection',     'mean total shedding of one infection x 128 mL',          'Covasim output x assumption'],
], columns=['quantity', 'unit', 'where it comes from', 'status'])
print(summary.to_string(index=False))

print('\nNOT covered here: the generation time (it needs the transmission log), the reported cases, and the EpiSewer fit itself.')
print('None of the assumption values is validated against measured San Diego wastewater data.')