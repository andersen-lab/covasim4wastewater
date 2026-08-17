'''Run a small spatial wastewater-enabled Covasim simulation.'''
import numpy as np

import pylab as pl

import covasim as cv



def main(shedding_model='gamma', cross_region_contact_factor=1.0):
    max_day = 300
    sample_days = np.arange(0, max_day, 1)
    wastewater = cv.WastewaterSampler(
        days=sample_days,
        label='regional wastewater',
        regions='all',
        capture_rates={1: 1.0, 2: 0.9, 3: 0.8, 4: 0.7},
    )

    sim = cv.Sim(
        pop_size=20000,
        pop_infected=50,
        n_days=max_day,
        verbose=0.1,
        region_pars=dict(
            enable=True,
            labels=[1, 2, 3, 4],
            probabilities=[0.5, 0.2, 0.2, 0.1],
            cross_region_contact_factor=cross_region_contact_factor, #1.0 preserves normal Covasim behavior; 0.0 prevents transmission across regions; any value in between will reduce cross-region transmission by that factor
        ),
        wastewater_pars=dict(
            shedding_model=shedding_model, # 'viral_load' or 'shedding_hub' or gamma or any callable function
            # shedding_lookup=
            # date_infectious=
            gamma_scale=3.0,
            duration=21,
            max_individual_multiplier=8.0,
            pathogen_scale=10000.0,
        ),
        evo_pars=dict(
            enable=True,
            L=1000,
            reference=None,
            mol_clock_rate=1e-5,
            sub_model='JC',
            fitness_model=None,
        ),
        analyzers=[wastewater],
    )

    sim.run()
    wastewater = sim.get_analyzer('regional wastewater')

    for day in sample_days:
        pooled = wastewater.get_sample(day)
        if pooled is None:
            print(f'Day {day}: no measurable shedding')
            continue

        print(f'\nDay {day}: pooled wastewater load={pooled.total_load:.2f}')
        for region, sample in wastewater.region_samples[day].items():
            if sample is not None:
                print(
                    f'  region {region}: infectious={sample.n_infectious}, '
                    f'load={sample.total_load:.2f}, genotypes={sample.n_genotypes}'
                )
   

    regions = wastewater._regions

    for region in regions:
        loads = []

        for day in sample_days:
            day_samples = wastewater.region_samples.get(day, {})
            sample = day_samples.get(region)

            if sample is None:
                loads.append(np.nan)
            else:
                loads.append(sample.total_load)

        pl.plot(
            sample_days,
            loads,
            marker='',
            label=f'Region {region}',
        )

    pl.xlabel('Day of simulation')
    pl.ylabel('Total viral load')
    pl.title(f'Wastewater viral load by region \n (shedding model={shedding_model}, cross-region contact factor={cross_region_contact_factor})')
    pl.legend()
    pl.grid(alpha=0.2)
    pl.tight_layout()
    pl.savefig(f'wastewater_load_by_region_{shedding_model}_{cross_region_contact_factor}.png', dpi=300)

    # sim.plot()

    del sim
    del wastewater
    pl.close('all')


if __name__ == '__main__':
    viral_shedding_models = ['viral_load', 'gamma', 'shedding_hub']
    cross_region_contact_factors = [1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.0]

    for shedding_model in viral_shedding_models:
        for cross_region_contact_factor in cross_region_contact_factors:
            try:
                main(shedding_model=shedding_model, cross_region_contact_factor=cross_region_contact_factor)
            except Exception as e:
                print(f'Error occurred with shedding_model={shedding_model}, cross_region_contact_factor={cross_region_contact_factor}: {e}')