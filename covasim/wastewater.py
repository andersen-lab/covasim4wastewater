'''
Wastewater sampling analyzer for Covasim.

WastewaterSampler snapshots circulating viral haplotypes and their relative
proportions weighted by individual wastewater shedding. It can return a pooled
sample and optional region/sewershed-specific samples.

Requires pars['evo_pars']['enable'] = True.
'''

import csv
import dataclasses
import os
import tempfile
from collections import defaultdict

import numpy as np

from .analysis import Analyzer
from .sequence_evolution import agent_mutations as _agent_mutations
from .sequence_evolution import decode_mutations as _decode_mutations

__all__ = ['WastewaterSampler', 'WastewaterSample']


@dataclasses.dataclass
class WastewaterSample:
    '''Snapshot of a viral haplotype mixture in wastewater.'''
    day: int
    date: str
    mutation_sets: list
    variant_labels: list
    proportions: list
    raw_loads: list
    n_infectious: int
    n_genotypes: int
    variant_shedding: dict = None
    region: int = None
    total_load: float = 0.0


class WastewaterSampler(Analyzer):
    '''
    Snapshot pooled and optional region-specific wastewater mixtures.

    Args:
        days          (list): simulation days or calendar dates to sample
        label          (str): optional analyzer label
        regions        (list/str): region IDs to sample, or ``'all'``
        capture_rates  (dict): optional ``{region: fraction_entering_sample}``

    ``self.samples[day]`` retains the original pooled-sample interface.
    Regional samples are available at ``self.region_samples[day][region]``.
    '''

    def __init__(self, days, label=None, regions=None, capture_rates=None):
        super().__init__(label=label)
        self._days_input = days
        self._regions_input = regions
        self.capture_rates = capture_rates or {}
        self.samples = {}
        self.region_samples = {}
        self._reference = None
        self._regions = []

    def initialize(self, sim):
        super().initialize(sim)
        if not sim['evo_pars'].get('enable', False):
            raise ValueError(
                "WastewaterSampler requires pars['evo_pars']['enable'] = True"
            )

        self._reference = sim.people.sequence_tracker.reference
        self._day_set = {
            sim.day(day) if isinstance(day, str) else int(day)
            for day in self._days_input
        }

        available_regions = [int(r) for r in np.unique(sim.people.region)]
        if self._regions_input is None:
            self._regions = []
        elif isinstance(self._regions_input, str) and self._regions_input == 'all':
            self._regions = available_regions
        else:
            self._regions = [int(r) for r in self._regions_input]
            unknown = sorted(set(self._regions) - set(available_regions))
            if unknown:
                raise ValueError(f'Unknown wastewater region IDs: {unknown}')

        normalized_rates = {}
        for region, rate in self.capture_rates.items():
            rate = float(rate)
            if rate < 0:
                raise ValueError('Wastewater capture rates must be non-negative')
            normalized_rates[int(region)] = rate
        self.capture_rates = normalized_rates

    def apply(self, sim):
        if sim.t in self._day_set:
            self._take_sample(sim)

    def _capture_weights(self, sim, inds):
        '''Return the fraction of each agent's shedding entering the sample.'''
        weights = np.ones(len(inds), dtype=float)
        if self.capture_rates:
            regions = sim.people.region[inds]
            for region, rate in self.capture_rates.items():
                weights[regions == region] = rate
        return weights

    def _take_sample(self, sim):
        infectious = np.nonzero(sim.people.infectious)[0]

        self.samples[sim.t] = self._build_sample(
            sim=sim,
            inds=infectious,
            region=None,
        )

        regional = {}
        for region in self._regions:
            region_inds = infectious[sim.people.region[infectious] == region]
            regional[region] = self._build_sample(
                sim=sim,
                inds=region_inds,
                region=region,
            )
        self.region_samples[sim.t] = regional

    def _build_sample(self, sim, inds, region=None):
        '''Build one pooled or regional sample.'''
        if len(inds) == 0:
            return None

        loads = np.asarray(sim.people.viral_shedding[inds], dtype=float)
        loads = loads * self._capture_weights(sim, inds)
        total = float(loads.sum())
        if total <= 0:
            return None

        tracker = sim.people.sequence_tracker
        ref = self._reference
        variant_map = sim.pars.get('variant_map', {})

        load_by_muts = defaultdict(float)
        label_by_muts = {}
        load_by_variant = defaultdict(float)

        for idx, load in zip(inds, loads):
            if load <= 0:
                continue

            i = int(idx)
            muts = _agent_mutations(tracker, i, ref)
            load_by_muts[muts] += float(load)

            variant_index = sim.people.infectious_variant[i]
            if np.isnan(variant_index):
                variant_label = 'wild'
            else:
                variant_label = variant_map.get(
                    int(variant_index),
                    f'variant_{int(variant_index)}',
                )

            label_by_muts.setdefault(muts, variant_label)
            load_by_variant[variant_label] += float(load)

        mutation_sets = list(load_by_muts)
        raw_loads = [load_by_muts[muts] for muts in mutation_sets]
        proportions = [load / total for load in raw_loads]
        if proportions:
            proportions[-1] = 1.0 - sum(proportions[:-1])

        return WastewaterSample(
            day=sim.t,
            date=sim.date(sim.t),
            mutation_sets=mutation_sets,
            variant_labels=[label_by_muts[muts] for muts in mutation_sets],
            proportions=proportions,
            raw_loads=raw_loads,
            n_infectious=int(len(inds)),
            n_genotypes=len(mutation_sets),
            variant_shedding=dict(load_by_variant),
            region=region,
            total_load=total,
        )

    def get_sample(self, day, region=None):
        '''Return a pooled sample or a sample for one region.'''
        day = int(day)
        if region is None:
            return self.samples[day]
        return self.region_samples[day][int(region)]

    def to_fasta(self, day, region=None):
        '''Return a multi-FASTA string for a pooled or regional sample.'''
        sample = self.get_sample(day, region=region)
        if sample is None:
            return ''

        region_text = 'pooled' if sample.region is None else str(sample.region)
        lines = []
        for i, (muts, prop, variant_label) in enumerate(zip(
            sample.mutation_sets,
            sample.proportions,
            sample.variant_labels,
        )):
            lines.append(
                f'>genotype_{i}  proportion={prop:.8f}  day={day}  '
                f'date={sample.date}  region={region_text}  variant={variant_label}'
            )
            lines.append(_decode_mutations(muts, self._reference))
        return '\n'.join(lines)

    def simulate_sample(
        self,
        day,
        primers,
        reference,
        outdir='./reads/',
        readcnt=500,
        redo=False,
        region=None,
        **bygul_kwargs,
    ):
        '''Generate synthetic amplicon reads for a pooled or regional sample.'''
        try:
            from bygul._cli import simulate_proportions
        except ImportError as exc:
            raise ImportError(
                "The 'bygul' package is required for WastewaterSampler.simulate_sample(). "
                'Install it from the Andersen Lab Bygul repository.'
            ) from exc

        sample = self.get_sample(day, region=region)
        if sample is None:
            where = 'pooled sample' if region is None else f'region {region}'
            raise ValueError(
                f'No measurable infectious shedding was present in the {where} on day {day}'
            )

        with tempfile.TemporaryDirectory() as tmpdir:
            suffix = 'pooled' if region is None else f'region{int(region)}'
            multifasta_path = os.path.join(tmpdir, f'wastewater_day{day}_{suffix}.fasta')
            csv_path = os.path.join(tmpdir, f'proportions_day{day}_{suffix}.csv')

            rounded = [round(p, 10) for p in sample.proportions[:-1]]
            rounded.append(round(1.0 - sum(rounded), 10))

            with open(multifasta_path, 'w') as fasta_file, open(
                csv_path,
                'w',
                newline='',
            ) as csv_file:
                csv_writer = csv.writer(csv_file)
                csv_writer.writerow(['sample_name', 'proportion'])

                for i, muts in enumerate(sample.mutation_sets):
                    sequence_name = f'genotype_{i}'
                    sequence = _decode_mutations(muts, self._reference)
                    fasta_file.write(f'>{sequence_name}\n{sequence}\n')
                    csv_writer.writerow([sequence_name, rounded[i]])

            args = [
                '--multifasta', multifasta_path,
                '--csv', csv_path,
                '--primers', primers,
                '--reference', reference,
                '--outdir', outdir,
                '--readcnt', str(readcnt),
            ]
            for key, value in bygul_kwargs.items():
                args.extend([f'--{key}', str(value)])
            if redo:
                args.append('--redo')

            return simulate_proportions.main(args=args, standalone_mode=False)
