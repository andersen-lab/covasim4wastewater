# spatial wastewater integration

This update primarily adds three new features to Covasim:

**Configurable regions or sewersheds.** `region_pars` assigns every agent to an integer region using probabilities or an explicit assignment array.

**Lightweight spatial mixing.** `cross_region_contact_factor` scales transmission on contacts that cross region boundaries. A value of `1.0` preserves normal Covasim behavior; `0.0` prevents transmission across regions.

**Separate wastewater shedding and regional sampling.** `wastewater_pars` selects the existing viral-load model, an optional gamma-shaped shedding curve, or a custom callable. `WastewaterSampler` can produce pooled and per-region haplotype mixtures.

These files are updated as part of this update:

1. [parameters.py](covasim/parameters.py): `region_pars` and `wastewater_pars` are added to the default parameters.
2. [population.py](covasim/population.py): `assign_regions` assigns agents to regions using either probabilities or an explicit assignment array; uses `region_pars` to configure the population;
3. [sim.py](covasim/sim.py): `viral_shedding` is updated by `cvu.compute_wastewater_shedding`; uses `wastewater_pars` to configure the shedding model; also use `region_pars` and `cross_region_contact_factor` to scale transmission on cross-region contacts;
4. [utils.py](covasim/utils.py): Renames `compute_viral_shedding` to `compute_viral_shedding_sh` to be used for integrating shedding hubs. Implements `compute_wastewater_shedding` to compute shedding for wastewater sampling; implements
5. [wastewater.py](covasim/wastewater.py): Add support for region-specific wastewater sampling. Adds `region` and `total_load` to `WastewaterSample`, adds `regions` and `capture_rates` configuration, validates requested regions and capture rates, applies region-specific capture weights to shedding, stores regional samples separately in `region_samples`, and allows `get_sample`, `to_fasta`, and `simulate_sample` to operate on either pooled or region-specific samples while preserving the original pooled behavior.
6. [main.py](covasim/main.py): Adds a new example for spatial wastewater integration and visualization of the results.

## Example

```python
import numpy as np
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
            cross_region_contact_factor=cross_region_contact_factor,
        ),
        wastewater_pars=dict(
            shedding_model=shedding_model,
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
```
